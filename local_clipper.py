#!/usr/bin/env python3
"""Local-first YouTube video clipping prototype.

Downloads a target YouTube video (defaulting to https://youtu.be/_nSjfe3k6Ss),
extracts lightweight mono 16kHz audio, transcribes with faster-whisper,
selects high-retention viral segments via Gemini 2.5 Flash, renders 9:16
vertical Shorts with FFmpeg, burns subtitles with Windows path-escaping,
and presents a detailed inspection summary.
"""

import argparse
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional load_dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional third-party imports (handled gracefully if missing)
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local_clipper")

DEFAULT_TARGET_URL = "https://youtu.be/_nSjfe3k6Ss"


# ==============================================================================
# 1. FFmpeg & Environment Utilities
# ==============================================================================

def get_ffmpeg_binary() -> str:
    """Locate an available FFmpeg binary on Windows/Linux."""
    # 1. Check system PATH
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        return ffmpeg_cmd

    # 2. Check imageio-ffmpeg package fallback
    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception:
            pass

    # 3. Check common Windows install locations
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\tools\ffmpeg\bin\ffmpeg.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    raise RuntimeError(
        "FFmpeg binary was not found on your system.\n"
        "Please install it using one of the following methods:\n"
        "  1) pip install imageio-ffmpeg  (quickest, bundled python ffmpeg)\n"
        "  2) winget install Gyan.FFmpeg   (system-wide Windows ffmpeg)"
    )


def discover_cookies(explicit_path: Optional[str] = None, work_dir: Path = Path("./work")) -> Optional[str]:
    """Search for valid YouTube Netscape cookie files in local folder, Downloads, or env var."""
    # 1. User-specified explicit path
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.is_file() and p.stat().st_size > 0:
            logger.info("Using specified cookies file: %s", p)
            return str(p)
        logger.warning("Specified cookie file not found or empty: %s", explicit_path)

    # 2. Check local workspace files
    for name in ["cookies.txt", "www.youtube.com_cookies.txt"]:
        p = Path(name).resolve()
        if p.is_file() and p.stat().st_size > 0:
            logger.info("Found local cookies file: %s", p.name)
            return str(p)

    # 3. Check user's Downloads folder
    try:
        downloads = Path.home() / "Downloads"
        for name in ["cookies.txt", "www.youtube.com_cookies.txt"]:
            p = downloads / name
            if p.is_file() and p.stat().st_size > 0:
                logger.info("Found cookies in Downloads folder: %s", p)
                return str(p)
    except Exception:
        pass

    # 4. Check YT_COOKIES_B64 environment variable
    b64_val = os.environ.get("YT_COOKIES_B64")
    if b64_val and b64_val.strip():
        try:
            clean_b64 = b64_val.strip().strip('"\'')
            rem = len(clean_b64) % 4
            if rem:
                clean_b64 += "=" * (4 - rem)
            decoded_text = base64.b64decode(clean_b64).decode("utf-8", errors="ignore")
            if decoded_text.strip():
                work_dir.mkdir(parents=True, exist_ok=True)
                temp_file = (work_dir / "temp_cookies.txt").resolve()
                temp_file.write_text(decoded_text, encoding="utf-8")
                logger.info("Decoded YT_COOKIES_B64 into temporary cookies file: %s", temp_file)
                return str(temp_file)
        except Exception as exc:
            logger.warning("Could not decode YT_COOKIES_B64: %s", exc)

    logger.info("No cookies found. Proceeding with public yt-dlp download...")
    return None


# ==============================================================================
# 2. Video Downloader (yt-dlp)
# ==============================================================================

def download_video(url: str, output_path: Path, cookie_file: Optional[str] = None) -> Dict[str, Any]:
    """Download target YouTube video into output_path using yt-dlp."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_tmpl = str(output_path.parent / "temp_source.%(ext)s")

    # Clean prior temp files
    for f in output_path.parent.glob("temp_source.*"):
        f.unlink(missing_ok=True)
    if output_path.exists():
        output_path.unlink(missing_ok=True)

    logger.info("Starting download for: %s", url)

    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    ydl_opts: Dict[str, Any] = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": temp_tmpl,
        "quiet": False,
        "no_warnings": False,
        "retries": 5,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    # Locate downloaded mp4 and move to final output_path
    candidate_mp4 = Path(filename).with_suffix(".mp4")
    if not candidate_mp4.exists():
        found = list(output_path.parent.glob("temp_source*.mp4"))
        if found:
            candidate_mp4 = found[0]

    if not candidate_mp4.exists():
        raise FileNotFoundError(f"Failed to find downloaded MP4 at {candidate_mp4}")

    if candidate_mp4.resolve() != output_path.resolve():
        shutil.move(str(candidate_mp4), str(output_path))

    duration = float(info.get("duration") or 0.0)
    width = info.get("width")
    height = info.get("height")
    title = info.get("title", "YouTube Video")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        "Downloaded Video: '%s' | Duration: %.1fs (%02d:%02d) | Res: %sx%s | Size: %.2f MB",
        title,
        duration,
        int(duration // 60),
        int(duration % 60),
        width or "?",
        height or "?",
        file_size_mb,
    )

    return {
        "title": title,
        "duration": duration,
        "width": width,
        "height": height,
        "file_size_mb": file_size_mb,
        "file_path": output_path,
    }


# ==============================================================================
# 3. Audio Extraction & Transcription
# ==============================================================================

def extract_audio_16k(video_path: Path, audio_path: Path, ffmpeg_bin: str) -> Path:
    """Extract mono 16kHz WAV track for optimal fast Whisper processing without OOM."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.exists():
        audio_path.unlink()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path.resolve()),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path.resolve()),
    ]

    logger.info("Extracting mono 16kHz audio track to %s...", audio_path.name)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        logger.error("FFmpeg audio extraction error:\n%s", res.stderr)
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction produced empty or missing file: {audio_path}")

    logger.info("Audio track extracted: %.2f MB", audio_path.stat().st_size / (1024 * 1024))
    return audio_path


def transcribe_audio(
    audio_path: Path,
    transcript_json: Path,
    model_size: str = "base.en",
) -> List[Dict[str, Any]]:
    """Transcribe extracted audio using faster-whisper and write transcript.json."""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

    logger.info("Initializing faster-whisper model '%s' (CPU, int8)...", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)

    logger.info("Transcribing audio with VAD silence filtering...")
    segments, info = model.transcribe(
        str(audio_path.resolve()),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
    )

    results: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if text:
            results.append({
                "id": idx,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
            })

    # Save to transcript.json
    transcript_json.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "duration": info.duration,
                "language": info.language,
                "segments_count": len(results),
                "segments": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Transcription complete: %d speech segments identified. Saved to %s",
        len(results),
        transcript_json.name,
    )
    return results


# ==============================================================================
# 4. Viral Segment Selection (Gemini 2.5 Flash + Fallbacks)
# ==============================================================================

def generate_fallback_segments(
    total_duration: float,
    max_clips: int,
    min_sec: int,
    max_sec: int,
) -> List[Dict[str, Any]]:
    """Provide evenly spaced fallback highlight segments when LLM is unavailable or speech is sparse."""
    logger.warning("Generating evenly-spaced fallback highlight segments across %.1fs video...", total_duration)
    target_len = float(min(max_sec, max(min_sec, 35)))
    clips = []

    if total_duration <= target_len:
        clips.append({
            "start_sec": 0.0,
            "end_sec": round(total_duration, 1),
            "hook_title": "INSANE GTA 6 MOMENT! (Watch Till End)",
            "reason": "Complete video highlight slice.",
        })
        return clips

    # Generate evenly distributed segments
    step = (total_duration - target_len) / max(1, max_clips)
    for i in range(max_clips):
        start = round(min(total_duration - target_len, i * step + 5.0), 1)
        end = round(min(total_duration, start + target_len), 1)
        clips.append({
            "start_sec": start,
            "end_sec": end,
            "hook_title": f"CRAZY GTA 6 HIGHLIGHT #{i+1} 🚨",
            "reason": f"Evenly-spaced action moment at {start}s.",
        })

    return clips


def find_viral_moments_gemini(
    segments: List[Dict[str, Any]],
    total_duration: float,
    video_title: str,
    max_clips: int = 3,
    min_sec: int = 20,
    max_sec: int = 55,
) -> List[Dict[str, Any]]:
    """Query Gemini 2.5 Flash to detect high-retention gaming clips."""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.error(
            "\n" + "=" * 70 + "\n"
            "[ERROR] GEMINI_API_KEY environment variable is missing!\n"
            "To enable AI hook detection, set your key in PowerShell:\n"
            "  $env:GEMINI_API_KEY = 'your_actual_api_key_here'\n"
            "or in CMD:\n"
            "  set GEMINI_API_KEY=your_actual_api_key_here\n"
            "or create a .env file with GEMINI_API_KEY=...\n"
            + "=" * 70
        )
        return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)

    if genai is None:
        logger.warning("google-genai package not installed. Run: pip install google-genai. Using fallbacks.")
        return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)

    # Format transcript summary
    lines = []
    for s in segments:
        m_s, s_s = int(s["start"] // 60), int(s["start"] % 60)
        m_e, s_e = int(s["end"] // 60), int(s["end"] % 60)
        lines.append(f"[{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}] {s['text']}")
    transcript_text = "\n".join(lines)

    if not transcript_text.strip():
        logger.warning("Transcript is empty or has no spoken dialogue. Using fallbacks.")
        return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)

    prompt = (
        f"You are an elite short-form video editor specializing in viral YouTube Shorts for US gaming audiences (GTA 6 / GTA Online).\n"
        f"Video Title: {video_title}\n"
        f"Total Video Duration: {total_duration:.1f} seconds.\n\n"
        f"Analyze the following timestamped transcript and identify the top {max_clips} most viral, intense, funny, or climactic moments.\n"
        f"CRITICAL CONSTRAINTS:\n"
        f"1. Each clip duration MUST be between {min_sec} and {max_sec} seconds (end_sec - start_sec).\n"
        f"2. start_sec and end_sec MUST be positive numbers within 0 and {total_duration:.1f}.\n"
        f"3. Return ONLY a valid JSON array of objects with keys:\n"
        f"   - 'start_sec': (float) start timestamp in seconds\n"
        f"   - 'end_sec': (float) end timestamp in seconds\n"
        f"   - 'hook_title': (string) punchy, click-worthy hook title in uppercase or titlecase for Shorts (e.g., 'HE COULDN'T BELIEVE THIS GLITCH 💀')\n"
        f"   - 'reason': (string) concise explanation of the viral retention hook\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    try:
        client = genai.Client(api_key=api_key)
        logger.info("Sending transcript (duration: %.1fs) to Gemini 2.5 Flash...", total_duration)

        config_args = {"temperature": 0.7}
        if types and hasattr(types, "GenerateContentConfig"):
            config = types.GenerateContentConfig(**config_args)
        else:
            config = config_args

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        raw_text = getattr(response, "text", "") or str(response)

        # Regex cleaner: extract content between first '[' and last ']'
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in Gemini response. Using fallbacks.")
            return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)

        json_str = match.group(0).strip()
        parsed = json.loads(json_str)

        if not isinstance(parsed, list) or not parsed:
            logger.warning("Gemini returned empty or non-list JSON. Using fallbacks.")
            return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)

        # Sanitize and validate every segment
        valid_segments: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("start_sec", 0.0))
                end = float(item.get("end_sec", start + 35.0))
                title = str(item.get("hook_title", "INSANE GTA 6 MOMENT")).strip()
                reason = str(item.get("reason", "High-energy gameplay moment.")).strip()

                # Clamp within boundaries
                start = max(0.0, min(total_duration - min_sec, start))
                end = min(total_duration, max(start + min_sec, end))
                clip_dur = end - start

                if clip_dur < min_sec:
                    end = min(total_duration, start + min_sec)
                elif clip_dur > max_sec:
                    end = start + max_sec

                valid_segments.append({
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "hook_title": title,
                    "reason": reason,
                })
            except (ValueError, TypeError) as conv_err:
                logger.warning("Skipping malformed segment candidate: %s", conv_err)

        if not valid_segments:
            return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)

        return valid_segments[:max_clips]

    except Exception as exc:
        logger.error("Gemini 2.5 Flash hook analysis failed: %s. Reverting to fallbacks.", exc)
        return generate_fallback_segments(total_duration, max_clips, min_sec, max_sec)


# ==============================================================================
# 5. Vertical 9:16 Rendering & Subtitle Generation
# ==============================================================================

def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into standard SRT timestamp HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def create_clip_srt(
    transcript_segments: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_srt: Path,
) -> bool:
    """Generate an SRT subtitle file relative to the clip's local start time."""
    clip_duration = clip_end - clip_start
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    cues = []
    cue_idx = 1
    for seg in transcript_segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        text = seg["text"].strip()

        # Check overlap
        if s_end > clip_start and s_start < clip_end and text:
            rel_start = max(0.0, s_start - clip_start)
            rel_end = min(clip_duration, s_end - clip_start)

            if rel_end > rel_start:
                cues.append(
                    f"{cue_idx}\n"
                    f"{format_srt_timestamp(rel_start)} --> {format_srt_timestamp(rel_end)}\n"
                    f"{text}\n"
                )
                cue_idx += 1

    if cues:
        output_srt.write_text("\n".join(cues), encoding="utf-8")
        return True

    return False


def render_vertical_short(
    source_video: Path,
    start_sec: float,
    end_sec: float,
    output_clip: Path,
    srt_file: Optional[Path],
    ffmpeg_bin: str,
) -> Path:
    """Render a 9:16 vertical short (1080x1920) with subtitle burn-in and robust fallback."""
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    if output_clip.exists():
        output_clip.unlink()

    duration = end_sec - start_sec
    base_filter = "crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0,scale=1080:1920"

    # Try burning subtitles first if an SRT file is present
    burned_with_subs = False
    if srt_file and srt_file.is_file() and srt_file.stat().st_size > 0:
        # Windows Bug Prevention: Escape drive letters and colons for FFmpeg subtitle filter
        srt_clean = str(srt_file.resolve()).replace("\\", "/").replace(":", r"\:")
        # Bold yellow text with black outline, bottom-center positioned
        sub_style = "FontSize=20,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2.5,Alignment=2,MarginV=60"
        vf_with_subs = f"{base_filter},subtitles='{srt_clean}':force_style='{sub_style}'"

        cmd_subs = [
            ffmpeg_bin,
            "-y",
            "-ss", str(round(start_sec, 2)),
            "-i", str(source_video.resolve()),
            "-t", str(round(duration, 2)),
            "-vf", vf_with_subs,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_clip.resolve()),
        ]

        logger.info("Attempting 9:16 render with burned subtitles for %s...", output_clip.name)
        res = subprocess.run(cmd_subs, capture_output=True, text=True)
        if res.returncode == 0 and output_clip.exists() and output_clip.stat().st_size > 0:
            burned_with_subs = True
            logger.info("Successfully rendered Short with burned subtitles!")
        else:
            logger.warning(
                "FFmpeg subtitle filter failed (likely missing libass/font on Windows). Falling back to clean 9:16 crop."
            )
            if output_clip.exists():
                output_clip.unlink()

    # Fallback: render without subtitles if subtitle burn failed or no SRT
    if not burned_with_subs:
        cmd_clean = [
            ffmpeg_bin,
            "-y",
            "-ss", str(round(start_sec, 2)),
            "-i", str(source_video.resolve()),
            "-t", str(round(duration, 2)),
            "-vf", base_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_clip.resolve()),
        ]

        logger.info("Executing clean 9:16 center-crop render for %s...", output_clip.name)
        res = subprocess.run(cmd_clean, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("FFmpeg rendering error:\n%s", res.stderr)
            raise subprocess.CalledProcessError(res.returncode, cmd_clean, output=res.stdout, stderr=res.stderr)

    # Validate output exists and is non-empty
    if not output_clip.exists() or output_clip.stat().st_size == 0:
        raise RuntimeError(f"Rendered clip file is missing or empty (0 bytes): {output_clip}")

    return output_clip


# ==============================================================================
# 6. Main Local CLI Orchestrator
# ==============================================================================

def slugify(text: str) -> str:
    """Generate safe alphanumeric file slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return clean[:40] or "clip"


def run_local_clipper(
    url: str,
    max_clips: int = 3,
    min_sec: int = 20,
    max_sec: int = 55,
    keep_work: bool = False,
    explicit_cookies: Optional[str] = None,
    work_dir_path: str = "./work",
    clips_dir_path: str = "./clips",
):
    """Execute the end-to-end local clipping pipeline."""
    start_total_time = time.time()
    work_dir = Path(work_dir_path).resolve()
    clips_dir = Path(clips_dir_path).resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GTA 6 LOCAL CLIPPING PROTOTYPE")
    print(f"Target URL: {url}")
    print(f"Max Clips: {max_clips} | Min Duration: {min_sec}s | Max Duration: {max_sec}s")
    print("=" * 80)

    # 1. Locate FFmpeg binary early
    ffmpeg_bin = get_ffmpeg_binary()
    logger.info("Using FFmpeg binary: %s", ffmpeg_bin)

    # 2. Check cookies
    cookie_file = discover_cookies(explicit_cookies, work_dir=work_dir)

    # 3. Download target video
    source_mp4 = work_dir / "source.mp4"
    video_meta = download_video(url=url, output_path=source_mp4, cookie_file=cookie_file)
    total_duration = video_meta["duration"]

    # 4. Extract mono 16kHz audio
    audio_wav = work_dir / "audio_16k.wav"
    extract_audio_16k(video_path=source_mp4, audio_path=audio_wav, ffmpeg_bin=ffmpeg_bin)

    # 5. Local Transcription with faster-whisper
    transcript_json = work_dir / "transcript.json"
    transcript_segments = transcribe_audio(
        audio_path=audio_wav,
        transcript_json=transcript_json,
        model_size="base.en",
    )

    # 6. Select viral moments via Gemini 2.5 Flash
    viral_segments = find_viral_moments_gemini(
        segments=transcript_segments,
        total_duration=total_duration,
        video_title=video_meta["title"],
        max_clips=max_clips,
        min_sec=min_sec,
        max_sec=max_sec,
    )

    logger.info("Found %d viral moment(s) to render.", len(viral_segments))

    # 7. Render 9:16 vertical clips
    saved_clips_summary: List[Dict[str, Any]] = []

    for idx, seg in enumerate(viral_segments, start=1):
        s_sec = seg["start_sec"]
        e_sec = seg["end_sec"]
        dur = e_sec - s_sec
        title = seg["hook_title"]
        reason = seg["reason"]
        slug = slugify(title)

        clip_filename = f"clip_{idx:02d}_{slug}.mp4"
        output_clip_path = clips_dir / clip_filename
        srt_path = work_dir / f"clip_{idx:02d}.srt"

        # Generate SRT
        has_subtitles = create_clip_srt(
            transcript_segments=transcript_segments,
            clip_start=s_sec,
            clip_end=e_sec,
            output_srt=srt_path,
        )

        # Render clip
        render_vertical_short(
            source_video=source_mp4,
            start_sec=s_sec,
            end_sec=e_sec,
            output_clip=output_clip_path,
            srt_file=srt_path if has_subtitles else None,
            ffmpeg_bin=ffmpeg_bin,
        )

        size_mb = output_clip_path.stat().st_size / (1024 * 1024)
        saved_clips_summary.append({
            "index": idx,
            "filename": clip_filename,
            "full_path": str(output_clip_path.resolve()),
            "start_sec": s_sec,
            "end_sec": e_sec,
            "duration": round(dur, 2),
            "size_mb": round(size_mb, 2),
            "hook_title": title,
            "reason": reason,
        })

    # 8. Cleanup work directory if not --keep-work
    if not keep_work:
        logger.info("Cleaning up large intermediate files in %s...", work_dir)
        source_mp4.unlink(missing_ok=True)
        audio_wav.unlink(missing_ok=True)
        temp_cookie = work_dir / "temp_cookies.txt"
        temp_cookie.unlink(missing_ok=True)
        logger.info("Intermediate video/audio files removed. Kept: %s", transcript_json.name)
    else:
        logger.info("Flag --keep-work enabled. All files in %s preserved.", work_dir)

    # 9. Local Inspection Summary Printout
    elapsed_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print("LOCAL CLIPPING PROTOTYPE COMPLETE")
    print(f"Total processing time: {elapsed_time:.1f}s")
    print(f"Output Directory: {clips_dir.resolve()}")
    print("=" * 80)

    for item in saved_clips_summary:
        print(f"\n[{item['index']}] {item['filename']} ({item['size_mb']} MB)")
        print(f"    Path:       {item['full_path']}")
        print(f"    Timestamps: {item['start_sec']}s -> {item['end_sec']}s (Duration: {item['duration']}s)")
        print(f"    Title:      {item['hook_title']}")
        print(f"    Reason:     {item['reason']}")

    print("\n" + "=" * 80)
    print(f"Successfully generated {len(saved_clips_summary)} vertical 9:16 Short(s)!")
    print("=" * 80 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Local-first YouTube 9:16 Video Clipper Prototype")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_TARGET_URL,
        help=f"Target YouTube video URL (default: {DEFAULT_TARGET_URL})",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=3,
        help="Maximum number of viral clips to produce (default: 3)",
    )
    parser.add_argument(
        "--min-sec",
        type=int,
        default=20,
        help="Minimum clip duration in seconds (default: 20)",
    )
    parser.add_argument(
        "--max-sec",
        type=int,
        default=55,
        help="Maximum clip duration in seconds (default: 55)",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate large files in ./work/ (source.mp4, audio_16k.wav)",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Optional path to a custom cookies.txt file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        run_local_clipper(
            url=args.url,
            max_clips=args.max_clips,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            keep_work=args.keep_work,
            explicit_cookies=args.cookies,
        )
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error("\n[FATAL ERROR] %s", e, exc_info=True)
        sys.exit(1)
