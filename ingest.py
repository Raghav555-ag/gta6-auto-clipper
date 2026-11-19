"""High-Resolution Video Ingest and Real Word-Level Transcription Module.

Downloads source footage in high-resolution (up to 4K 2160p or 1440p) using yt-dlp
so vertical 9:16 crops downscale cleanly without upscaling blur. Extracts 16kHz mono
audio and generates authentic word-level transcripts via faster-whisper without ever
falling back to mock or fake subtitles.
"""

import argparse
import base64
import binascii
import contextlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

DEFAULT_TARGET_URL = "https://youtu.be/_nSjfe3k6Ss"
DEFAULT_WORK_DIR = Path("./work")


# ==============================================================================
# 1. FFmpeg Binary Discovery & Media Probing
# ==============================================================================

def get_ffmpeg_binary() -> str:
    """Locate a functional FFmpeg executable across PATH, imageio_ffmpeg, or environment.

    Returns:
        str: Absolute or system path to ffmpeg binary.

    Raises:
        RuntimeError: If FFmpeg is not installed or discoverable.
    """
    env_ffmpeg = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG_BINARY")
    if env_ffmpeg and Path(env_ffmpeg).is_file():
        return str(Path(env_ffmpeg).resolve())

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    # Windows standard paths check
    win_paths = [
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    ]
    for p in win_paths:
        if p.is_file():
            return str(p.resolve())

    raise RuntimeError(
        "FFmpeg binary not found on PATH or via imageio-ffmpeg. "
        "Please install FFmpeg or install imageio-ffmpeg via: pip install imageio-ffmpeg"
    )


def probe_video_metadata(video_path: Path, ffmpeg_bin: Optional[str] = None) -> Dict[str, Any]:
    """Inspect video file to extract actual resolution, FPS, codecs, and duration.

    Args:
        video_path: Path to the media file.
        ffmpeg_bin: Optional path to ffmpeg executable.

    Returns:
        Dict[str, Any]: Metadata containing width, height, fps, vcodec, acodec, duration.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found for probing: {path}")

    bin_path = ffmpeg_bin or get_ffmpeg_binary()
    cmd = [bin_path, "-hide_banner", "-i", str(path)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = proc.stderr

    duration = 0.0
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
    if dur_match:
        h, m, s = dur_match.groups()
        duration = round(int(h) * 3600 + int(m) * 60 + float(s), 2)

    width = 0
    height = 0
    fps = 0.0
    vcodec = "unknown"
    acodec = "unknown"

    # Search video stream info: e.g. "Stream #0:0[0x1](und): Video: h264 (High) (avc1 / ...), yuv420p(tv,...), 1920x1080 [SAR 1:1 DAR 16:9], 60 fps"
    video_stream_match = re.search(r"Stream #\d+:\d+.*?: Video: ([^,\s]+).*?,\s*(\d{3,4})x(\d{3,4})", stderr)
    if video_stream_match:
        vcodec = video_stream_match.group(1)
        width = int(video_stream_match.group(2))
        height = int(video_stream_match.group(3))

    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    if fps_match:
        fps = float(fps_match.group(1))

    # Search audio stream info: e.g. "Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / ...), 48000 Hz, stereo, fltp"
    audio_stream_match = re.search(r"Stream #\d+:\d+.*?: Audio: ([^,\s]+)", stderr)
    if audio_stream_match:
        acodec = audio_stream_match.group(1)

    file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)

    meta = {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "vcodec": vcodec,
        "acodec": acodec,
        "duration": duration,
        "file_size_mb": file_size_mb,
        "file_path": str(path),
    }

    logger.info(
        "Downloaded Video Verified: %s | %s @ %.1ffps | Codecs: video=%s, audio=%s | Duration: %.1fs | Size: %.2f MB",
        path.name,
        meta["resolution"],
        meta["fps"],
        meta["vcodec"],
        meta["acodec"],
        meta["duration"],
        meta["file_size_mb"],
    )
    return meta


# ==============================================================================
# 2. Secure Netscape Cookie Decoding & Context Management
# ==============================================================================

def decode_and_validate_cookies(raw_b64: Optional[str]) -> str:
    """Validate and decode a Base64-encoded Netscape cookie string.

    Args:
        raw_b64: Base64 string from environment variable or file.

    Returns:
        Decoded UTF-8 string containing Netscape cookies.

    Raises:
        ValueError: If raw_b64 is invalid, empty, or lacks Netscape YouTube domain entries.
    """
    if not raw_b64 or not raw_b64.strip():
        raise ValueError("Cookie string is missing or empty.")

    cleaned_b64 = raw_b64.strip().strip('"\'')
    padding_needed = len(cleaned_b64) % 4
    if padding_needed:
        cleaned_b64 += "=" * (4 - padding_needed)

    try:
        decoded_bytes = base64.b64decode(cleaned_b64, validate=False)
        decoded_text = decoded_bytes.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to decode Base64 cookies (invalid Base64 or UTF-8): {exc}") from exc

    if not decoded_text.strip():
        raise ValueError("Decoded cookies file is empty.")

    lines = [line.strip() for line in decoded_text.splitlines() if line.strip()]
    has_header = any("# Netscape HTTP Cookie File" in line for line in lines)
    valid_domain_entries = [
        line for line in lines
        if (
            not line.startswith("#")
            and len(line.split("\t")) >= 6
            and any(dom in line.split("\t")[0] for dom in ["youtube.com", "google.com"])
        )
        or (
            line.startswith("#HttpOnly_")
            and any(dom in line for dom in ["youtube.com", "google.com"])
        )
    ]

    if not (has_header or valid_domain_entries):
        raise ValueError(
            "Decoded cookies do not appear to contain valid Netscape domain lines for YouTube (.youtube.com / google.com)."
        )

    return decoded_text


@contextlib.contextmanager
def secure_cookie_context(cookie_hint: Optional[Union[str, Path]] = None):
    """Context manager for securely setting up cookies and guaranteeing deletion.

    Checks:
    1. Explicit file path passed in `cookie_hint`
    2. Local files `cookies.txt` or `www.youtube.com_cookies.txt`
    3. `YT_COOKIES_B64` environment variable

    Yields:
        Optional[str]: Path to active cookie file, or None if public access is used.
    """
    temp_cookie_path: Optional[Path] = None
    active_cookie_file: Optional[str] = None

    try:
        # Check explicit path
        if cookie_hint:
            hint_path = Path(cookie_hint).resolve()
            if hint_path.is_file() and hint_path.stat().st_size > 0:
                active_cookie_file = str(hint_path)

        # Check local standard files
        if not active_cookie_file:
            for candidate in [Path("cookies.txt"), Path("www.youtube.com_cookies.txt")]:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    active_cookie_file = str(candidate.resolve())
                    break

        # Check Base64 environment variable
        if not active_cookie_file and os.environ.get("YT_COOKIES_B64"):
            raw_b64 = os.environ.get("YT_COOKIES_B64", "")
            try:
                decoded = decode_and_validate_cookies(raw_b64)
                temp_cookie_path = Path("./work/temp_cookies.txt").resolve()
                temp_cookie_path.parent.mkdir(parents=True, exist_ok=True)
                temp_cookie_path.write_text(decoded, encoding="utf-8")
                try:
                    os.chmod(temp_cookie_path, stat.S_IRUSR | stat.S_IWUSR)
                except Exception:
                    pass
                active_cookie_file = str(temp_cookie_path)
                logger.info("Decoded Netscape cookies written securely to %s", temp_cookie_path.name)
            except Exception as e:
                logger.warning("Could not parse YT_COOKIES_B64: %s. Continuing without cookies.", e)

        yield active_cookie_file

    finally:
        # Guarantee shredding of temporary cookie file
        if temp_cookie_path and temp_cookie_path.exists():
            try:
                length = temp_cookie_path.stat().st_size
                with open(temp_cookie_path, "wb") as f:
                    f.write(b"\x00" * length)
                temp_cookie_path.unlink()
                logger.info("Securely shredded and removed temporary cookie file: %s", temp_cookie_path.name)
            except Exception as err:
                logger.warning("Error during cookie file cleanup: %s", err)


# ==============================================================================
# 3. High-Resolution Video Download (yt-dlp)
# ==============================================================================

def download_high_res_video(
    url: str,
    output_path: Path,
    max_height: int = 2160,
    cookie_file: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Download highest resolution source video up to 2160p (4K) or 1440p.

    Ensures that subsequent 9:16 vertical cropping downscales from sharp 4K/1440p
    footage instead of upscaling blurry 720p/1080p frames.

    Args:
        url: YouTube video URL.
        output_path: Target video file path (e.g. ./work/source.mp4).
        max_height: Maximum vertical resolution (default: 2160 for 4K).
        cookie_file: Path to Netscape cookie file.
        ffmpeg_bin: Path to FFmpeg binary for stream merging.

    Returns:
        Dict[str, Any]: Probed video metadata dictionary.

    Raises:
        RuntimeError: If yt-dlp is missing or download fails.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Install via: pip install yt-dlp")

    bin_path = ffmpeg_bin or get_ffmpeg_binary()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and is non-empty, probe and return cached metadata
    if out.is_file() and out.stat().st_size > 1024 * 1024:
        logger.info(
            "Found existing source video at %s (%.2f MB). Skipping re-download.",
            out.name,
            out.stat().st_size / (1024 * 1024),
        )
        return probe_video_metadata(out, ffmpeg_bin=bin_path)

    # Format selector preferring highest resolution up to 4K (2160p) or 1440p
    # bv*[height<=2160]+ba/b merges best video stream up to 2160p with best audio stream
    format_selector = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/best[height<={max_height}]/best"
    temp_tmpl = str(out.parent / f"download_{out.stem}.%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": format_selector,
        "merge_output_format": "mp4",
        "outtmpl": temp_tmpl,
        "ffmpeg_location": bin_path,
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if cookie_file and Path(cookie_file).is_file():
        ydl_opts["cookiefile"] = str(Path(cookie_file).resolve())
        logger.info("Using authenticated cookies for yt-dlp download: %s", Path(cookie_file).name)
    else:
        logger.info("No cookie file provided. Attempting public yt-dlp download...")

    logger.info("Downloading high-resolution video (max height: %dp) from: %s", max_height, url)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            if info_dict is None:
                raise RuntimeError(f"yt-dlp could not fetch video metadata for URL: {url}")

            downloaded_file = Path(ydl.prepare_filename(info_dict))
            merged_mp4 = downloaded_file.with_suffix(".mp4")

            actual_file = merged_mp4 if merged_mp4.is_file() else downloaded_file
            if not actual_file.is_file():
                # Locate any file matching download_source.*
                matches = list(out.parent.glob(f"download_{out.stem}.*"))
                if matches:
                    actual_file = matches[0]
                else:
                    raise FileNotFoundError(f"yt-dlp output file not found in {out.parent}")

            # Rename to final clean output destination
            if actual_file.resolve() != out.resolve():
                if out.exists():
                    out.unlink()
                shutil.move(str(actual_file), str(out))

    except Exception as exc:
        logger.error("Failed to download video with yt-dlp: %s", exc)
        raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

    # Probe and return final verified metadata
    return probe_video_metadata(out, ffmpeg_bin=bin_path)


# ==============================================================================
# 4. Audio Extraction (16kHz Mono PCM WAV)
# ==============================================================================

def extract_audio_16k_mono(
    video_path: Path,
    output_audio_path: Path,
    ffmpeg_bin: Optional[str] = None,
) -> Path:
    """Extract lightweight 16kHz mono WAV audio from raw video via FFmpeg.

    Prevents Whisper memory OOM by extracting only an uncompressed mono audio
    track tuned specifically for speech recognition models.

    Args:
        video_path: Path to source video file.
        output_audio_path: Path for output WAV file (e.g. ./work/audio_16k.wav).
        ffmpeg_bin: Path to FFmpeg binary.

    Returns:
        Path: Path to the generated 16kHz WAV file.

    Raises:
        RuntimeError: If FFmpeg fails or output audio is zero-byte.
    """
    v_path = Path(video_path).resolve()
    a_path = Path(output_audio_path).resolve()
    a_path.parent.mkdir(parents=True, exist_ok=True)

    if not v_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {v_path}")

    # If audio already extracted and non-empty, use existing
    if a_path.is_file() and a_path.stat().st_size > 1024:
        logger.info(
            "Found existing extracted 16kHz audio at %s (%.2f MB). Using cached file.",
            a_path.name,
            a_path.stat().st_size / (1024 * 1024),
        )
        return a_path

    bin_path = ffmpeg_bin or get_ffmpeg_binary()

    cmd = [
        bin_path,
        "-y",
        "-i", str(v_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(a_path),
    ]

    logger.info("Extracting 16kHz mono audio from %s -> %s...", v_path.name, a_path.name)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        logger.error("FFmpeg audio extraction failed:\n%s", proc.stderr)
        raise RuntimeError(f"FFmpeg audio extraction failed (exit code {proc.returncode}): {proc.stderr[:400]}")

    if not a_path.is_file() or a_path.stat().st_size == 0:
        raise RuntimeError(f"Extracted audio file is missing or empty at {a_path}")

    logger.info(
        "Extracted 16kHz mono audio successfully: %s (%.2f MB)",
        a_path.name,
        a_path.stat().st_size / (1024 * 1024),
    )
    return a_path


# ==============================================================================
# 5. Real Word-Level Transcription (Multilingual faster-whisper)
# ==============================================================================

def transcribe_real_audio(
    audio_path: Path,
    output_json_path: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe audio using faster-whisper with real word-level timestamps.

    VULNERABILITY PREVENTION:
    - Strictly prohibits fake or mock subtitles. If faster-whisper is unavailable
      or returns 0 words, aborts immediately with an actionable error.
    - Uses multilingual models ('small', 'medium', etc.) to handle English,
      Hinglish, and creator languages naturally.
    - Saves output in standard Whisper JSON format readable directly by pycaps.

    Args:
        audio_path: Path to extracted 16kHz WAV audio.
        output_json_path: Destination path for transcript.json.
        model_size: Model size ('small' or 'medium' - NOT '.en' models).
        device: 'cpu' or 'cuda'.
        compute_type: Quantization mode (default 'int8' for CPU memory efficiency).
        language: Optional language code (e.g. 'en', 'hi', or None for auto-detect).

    Returns:
        Dict[str, Any]: Whisper-compliant transcript dictionary containing segments and words.

    Raises:
        RuntimeError: If faster-whisper is missing, or transcription yields 0 words.
    """
    a_path = Path(audio_path).resolve()
    out_json = Path(output_json_path).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not a_path.is_file():
        raise FileNotFoundError(f"Audio file not found for transcription: {a_path}")

    # Enforce multilingual model selection (NOT .en models)
    if model_size.endswith(".en"):
        raise ValueError(
            f"Model '{model_size}' is an English-only model. "
            "Ingest requires a multilingual model ('small' or 'medium') to properly transcribe "
            "multilingual content and creator banter (e.g. English, Hinglish, Hindi)."
        )

    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Real word-level transcription requires faster-whisper. "
            "Install it via: pip install faster-whisper (mock subtitles are strictly prohibited)."
        )

    logger.info(
        "Initializing multilingual faster-whisper model '%s' (device=%s, compute_type=%s)...",
        model_size,
        device,
        compute_type,
    )

    try:
        model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
        )
    except Exception as exc:
        logger.error("Failed to load faster-whisper model '%s': %s", model_size, exc)
        raise RuntimeError(f"Could not initialize faster-whisper model: {exc}") from exc

    logger.info(
        "Transcribing '%s' with vad_filter=True, word_timestamps=True (language=%s)...",
        a_path.name,
        language or "auto-detect",
    )

    try:
        transcribe_kwargs: Dict[str, Any] = {
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=500),
            "word_timestamps": True,
            "beam_size": 5,
        }
        if language:
            transcribe_kwargs["language"] = language

        segments_iter, info = model.transcribe(str(a_path), **transcribe_kwargs)

        collected_segments: List[Dict[str, Any]] = []
        full_text_list: List[str] = []
        total_words_count = 0

        for seg in segments_iter:
            words_data: List[Dict[str, Any]] = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    word_str = w.word.strip()
                    if word_str:
                        words_data.append({
                            "word": word_str,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(getattr(w, "probability", 1.0), 3),
                        })
                        total_words_count += 1

            seg_text = seg.text.strip()
            if seg_text:
                full_text_list.append(seg_text)

            collected_segments.append({
                "id": seg.id,
                "seek": getattr(seg, "seek", 0),
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg_text,
                "tokens": getattr(seg, "tokens", []),
                "temperature": getattr(seg, "temperature", 0.0),
                "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                "compression_ratio": getattr(seg, "compression_ratio", 0.0),
                "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                "words": words_data,
            })

    except Exception as exc:
        logger.error("Error during faster-whisper transcription: %s", exc)
        raise RuntimeError(f"faster-whisper transcription failed: {exc}") from exc

    # VULNERABILITY PREVENTION: Zero Words Check
    if total_words_count == 0 or not collected_segments:
        err_msg = (
            f"Transcription aborted: audio at '{a_path.name}' produced 0 transcribed words. "
            "The video may be completely silent, contain non-speech audio, or the audio stream is corrupted. "
            "Falling back to mock or fake subtitles is strictly prohibited."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    detected_lang = getattr(info, "language", language or "en")
    lang_prob = round(getattr(info, "language_probability", 1.0), 3)
    duration = round(getattr(info, "duration", 0.0), 2)
    full_text = " ".join(full_text_list)

    # Standard Whisper JSON format compatible with pycaps
    whisper_json_data = {
        "text": full_text,
        "language": detected_lang,
        "language_probability": lang_prob,
        "duration": duration,
        "segments_count": len(collected_segments),
        "total_words_count": total_words_count,
        "segments": collected_segments,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(whisper_json_data, f, indent=2, ensure_ascii=False)

    logger.info(
        "Transcription verified: %d segments, %d words (detected language: %s, duration: %.1fs). "
        "Saved real word-level transcript to %s",
        len(collected_segments),
        total_words_count,
        detected_lang,
        duration,
        out_json.name,
    )

    return whisper_json_data


# ==============================================================================
# 6. End-to-End Ingest Pipeline
# ==============================================================================

def run_ingest(
    url: str = DEFAULT_TARGET_URL,
    work_dir: Path = DEFAULT_WORK_DIR,
    max_height: int = 2160,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    cookies_path: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete high-resolution ingestion and authentic transcription.

    Args:
        url: YouTube video URL.
        work_dir: Intermediate working directory (e.g. ./work).
        max_height: Preferred vertical resolution (default: 2160 for 4K).
        model_size: Multilingual faster-whisper model ('small' or 'medium').
        device: 'cpu' or 'cuda'.
        compute_type: 'int8' or 'float16'.
        cookies_path: Optional path to cookies.txt.
        language: Optional language code.

    Returns:
        Dict[str, Any]: Ingest summary containing video metadata and transcript info.
    """
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = get_ffmpeg_binary()
    logger.info("Using FFmpeg binary: %s", ffmpeg_bin)

    source_video = work / "source.mp4"
    audio_wav = work / "audio_16k.wav"
    transcript_json = work / "transcript.json"

    with secure_cookie_context(cookies_path) as active_cookie_file:
        # Step 1: High-Resolution Video Download
        video_meta = download_high_res_video(
            url=url,
            output_path=source_video,
            max_height=max_height,
            cookie_file=active_cookie_file,
            ffmpeg_bin=ffmpeg_bin,
        )

        # Step 2: 16kHz Mono Audio Extraction
        extract_audio_16k_mono(
            video_path=source_video,
            output_audio_path=audio_wav,
            ffmpeg_bin=ffmpeg_bin,
        )

        # Step 3: Real Word-Level Transcription
        transcript_data = transcribe_real_audio(
            audio_path=audio_wav,
            output_json_path=transcript_json,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )

    summary = {
        "video": video_meta,
        "audio_path": str(audio_wav),
        "transcript_path": str(transcript_json),
        "language": transcript_data["language"],
        "segments_count": transcript_data["segments_count"],
        "total_words_count": transcript_data["total_words_count"],
    }

    logger.info("=" * 70)
    logger.info("INGESTION & WORD-LEVEL TRANSCRIPTION COMPLETED")
    logger.info("Video File:       %s (%s, %.1ffps)", source_video.name, video_meta["resolution"], video_meta["fps"])
    logger.info("Extracted Audio:  %s (16kHz mono WAV)", audio_wav.name)
    logger.info("Transcript JSON:  %s (%d words, %d segments)", transcript_json.name, summary["total_words_count"], summary["segments_count"])
    logger.info("Language:         %s", summary["language"])
    logger.info("=" * 70)

    return summary


# ==============================================================================
# 7. CLI Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest high-resolution video (up to 4K 2160p) and generate authentic word-level transcripts"
    )
    parser.add_argument("--url", type=str, default=DEFAULT_TARGET_URL, help="Target YouTube URL")
    parser.add_argument("--work-dir", type=str, default="./work", help="Working directory for media & transcripts")
    parser.add_argument("--max-height", type=int, default=2160, help="Max video height in pixels (e.g. 2160 for 4K, 1440)")
    parser.add_argument("--model", type=str, default="small", help="Multilingual Whisper model ('small' or 'medium')")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device ('cpu' or 'cuda')")
    parser.add_argument("--compute-type", type=str, default="int8", help="Quantization compute type ('int8', 'float16')")
    parser.add_argument("--cookies", type=str, default=None, help="Path to Netscape cookies.txt")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g. 'en', 'hi', or omit for auto-detect)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_ingest(
            url=args.url,
            work_dir=Path(args.work_dir),
            max_height=args.max_height,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cookies_path=args.cookies,
            language=args.language,
        )
    except Exception as exc:
        logger.error("[FATAL ERROR] Ingest failed: %s", exc)
        sys.exit(1)
