"""Viral Hotspot & Editorial Director Engine (viral_hotspot.py).

Combines:
  1. YouTube's "Most Replayed" heatmap markers (inspired by cheat-clip) via yt-dlp.
  2. Librosa audio loudness & onset strength peak detection for reaction/laugh/crash spikes.
  3. CPU-optimized English word-level transcription with faster-whisper (small.en, int8).
  4. Gemini 2.5 Flash editorial director grounded in skills/MOOD_EDITOR_SKILL.md.

Vulnerability Prevention:
  - Strictly prohibits dummy or mock subtitles; fails fast if transcription yields 0 words.
  - Smooth fallback to librosa energy spikes when YouTube heatmap is unavailable.
  - Strict programmatic word boundary snapping (-0.25s lead-in, +0.35s tail padding).
  - Robust JSON regex sanitization to eliminate JSONDecodeError crashes.
"""

import argparse
import base64
import binascii
import contextlib
import json
import logging
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import wave
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
    import librosa
    import numpy as np
except ImportError:
    librosa = None
    np = None

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

from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viral_hotspot")

DEFAULT_TARGET_URL = "https://youtu.be/_nSjfe3k6Ss"
DEFAULT_WORK_DIR = Path("./work")
DEFAULT_SKILL_PATH = Path("./skills/MOOD_EDITOR_SKILL.md")


# ==============================================================================
# 1. Pydantic Schemas for Strict Editorial Output
# ==============================================================================

class HighlightWord(BaseModel):
    word: str = Field(description="Word to visually emphasize")
    color_tag: str = Field(
        default="yellow",
        description="'red' (danger/crazy), 'yellow' (hype/money/key), or 'green' (win/calm)",
    )

    @field_validator("color_tag")
    @classmethod
    def validate_color(cls, v: str) -> str:
        clean = str(v).lower().strip()
        if clean not in ("red", "yellow", "green"):
            return "yellow"
        return clean


class MoodEvent(BaseModel):
    time_sec: float = Field(
        description="Timestamp in seconds relative to clip start (0.0s = start of clip)"
    )
    mood: str = Field(
        default="laugh",
        description="Visual reaction mood: 'laugh', 'sigma_crazy', or 'shock_impact'",
    )
    sfx: str = Field(
        default="pop",
        description="Subtle SFX: 'pop', 'boom', 'whoosh', or 'hit'",
    )
    use_pixelfx: bool = Field(
        default=False,
        description="Whether to trigger subtle spark/impact visual effect",
    )

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, v: str) -> str:
        clean = str(v).lower().strip()
        if clean not in ("laugh", "sigma_crazy", "shock_impact"):
            return "laugh"
        return clean

    @field_validator("sfx")
    @classmethod
    def validate_sfx(cls, v: str) -> str:
        clean = str(v).lower().strip()
        if clean not in ("pop", "boom", "whoosh", "hit"):
            return "pop"
        return clean


class HotspotClip(BaseModel):
    start_sec: float = Field(
        description="Start timestamp in seconds within source video (aligned to word boundary)"
    )
    end_sec: float = Field(
        description="End timestamp in seconds within source video (duration: 18s - 48s)"
    )
    hook_kicker: str = Field(
        description="Short urgent top hook headline (e.g., 'WAIT FOR THE END! 💀' or 'HOW IS THIS POSSIBLE?!')"
    )
    main_title: str = Field(
        description="Punchy 4-to-7 word title + 1 emoji (e.g., 'GTA 6 Jet Ski Physics Is Broken 🔥')"
    )
    overall_mood: str = Field(
        default="hype",
        description="Dominant clip vibe: 'funny', 'sigma_crazy', or 'hype'",
    )
    viral_score: int = Field(
        ge=1,
        le=100,
        description="Predicted retention and viral score from 1 to 100",
    )
    reason: str = Field(
        description="Editorial justification detailing the Setup -> Escalation -> Payoff arc"
    )
    highlight_words: List[HighlightWord] = Field(
        default_factory=list,
        description="List of words to emphasize with mood colors",
    )
    mood_events: List[MoodEvent] = Field(
        default_factory=list,
        description="Max 3 timed visual mood panels & SFX cues relative to clip start",
    )


# ==============================================================================
# 2. FFmpeg Discovery & Cookie Management
# ==============================================================================

def get_ffmpeg_binary() -> str:
    """Locate functional FFmpeg executable across PATH, imageio-ffmpeg, or standard locations."""
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

    win_paths = [
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    ]
    for p in win_paths:
        if p.is_file():
            return str(p.resolve())

    raise RuntimeError("FFmpeg binary not found. Please install FFmpeg or imageio-ffmpeg.")


def decode_and_validate_cookies(raw_b64: Optional[str]) -> str:
    """Decode and validate a Base64-encoded Netscape cookie string."""
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
        raise ValueError(f"Failed to decode Base64 cookies: {exc}") from exc

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
        raise ValueError("Decoded cookies do not contain valid Netscape YouTube domain entries.")

    return decoded_text


@contextlib.contextmanager
def secure_cookie_context(cookie_hint: Optional[Union[str, Path]] = None, work_dir: Optional[Path] = None):
    """Context manager for securely mounting and guaranteed shredding of temporary cookies."""
    temp_cookie_path: Optional[Path] = None
    active_cookie_file: Optional[str] = None
    target_work = work_dir or DEFAULT_WORK_DIR

    try:
        if cookie_hint:
            hint_path = Path(cookie_hint).resolve()
            if hint_path.is_file() and hint_path.stat().st_size > 0:
                active_cookie_file = str(hint_path)

        if not active_cookie_file:
            for candidate in [Path("cookies.txt"), Path("www.youtube.com_cookies.txt")]:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    active_cookie_file = str(candidate.resolve())
                    break

        if not active_cookie_file and os.environ.get("YT_COOKIES_B64"):
            raw_b64 = os.environ.get("YT_COOKIES_B64", "")
            try:
                decoded = decode_and_validate_cookies(raw_b64)
                temp_cookie_path = target_work / "temp_cookies.txt"
                temp_cookie_path.parent.mkdir(parents=True, exist_ok=True)
                temp_cookie_path.write_text(decoded, encoding="utf-8")
                try:
                    os.chmod(temp_cookie_path, stat.S_IRUSR | stat.S_IWUSR)
                except Exception:
                    pass
                active_cookie_file = str(temp_cookie_path.resolve())
                logger.info("Decoded Netscape cookies written securely to %s", temp_cookie_path.name)
            except Exception as e:
                logger.warning("Could not parse YT_COOKIES_B64: %s. Continuing without cookies.", e)

        yield active_cookie_file

    finally:
        if temp_cookie_path and temp_cookie_path.exists():
            try:
                length = temp_cookie_path.stat().st_size
                with open(temp_cookie_path, "wb") as f:
                    f.write(b"\x00" * length)
                temp_cookie_path.unlink()
                logger.info("Securely shredded temporary cookie file: %s", temp_cookie_path.name)
            except Exception as err:
                logger.warning("Error during cookie file cleanup: %s", err)


# ==============================================================================
# 3. High-Resolution Video Download & Heatmap Extraction
# ==============================================================================

def parse_youtube_heatmap(raw_heatmap: Optional[List[Dict[str, Any]]], top_n: int = 6) -> List[Dict[str, Any]]:
    """Extract YouTube's 'Most Replayed' hotspot markers into structured peak intervals.

    Args:
        raw_heatmap: List of raw heatmap dicts from yt-dlp metadata with start_time, end_time, value.
        top_n: Number of top replayed hotspots to extract.

    Returns:
        List[Dict[str, Any]]: Ranked list of peak replayed intervals with intensity.
    """
    if not raw_heatmap or not isinstance(raw_heatmap, list):
        return []

    valid_windows = []
    for entry in raw_heatmap:
        if not isinstance(entry, dict) or "value" not in entry:
            continue
        val = float(entry.get("value", 0.0))
        if val <= 0.0:
            continue
        start_t = float(entry.get("start_time", 0.0))
        end_t = float(entry.get("end_time", start_t + 2.0))
        valid_windows.append({
            "start_sec": round(start_t, 2),
            "end_sec": round(end_t, 2),
            "intensity": round(val, 3),
        })

    if not valid_windows:
        return []

    # Sort descending by replay intensity
    sorted_windows = sorted(valid_windows, key=lambda w: w["intensity"], reverse=True)
    top_peaks = sorted_windows[:top_n]

    # Re-sort chronologically for editorial clarity
    top_peaks.sort(key=lambda w: w["start_sec"])
    return top_peaks


def download_source_and_metadata(
    url: str,
    output_video_path: Path,
    output_metadata_path: Path,
    max_height: int = 2160,
    cookie_file: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Download highest resolution video and extract YouTube 'Most Replayed' heatmap metadata.

    Args:
        url: Target YouTube video URL.
        output_video_path: Destination path for video (e.g. ./work/source.mp4).
        output_metadata_path: Destination path for metadata (e.g. ./work/metadata.json).
        max_height: Maximum vertical resolution (default: 2160 for 4K/UHD).
        cookie_file: Optional path to cookies.txt.
        ffmpeg_bin: Optional path to FFmpeg binary.

    Returns:
        Dict[str, Any]: Consolidated metadata including heatmap, title, and duration.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Install via: pip install yt-dlp")

    bin_path = ffmpeg_bin or get_ffmpeg_binary()
    out_video = Path(output_video_path).resolve()
    out_meta = Path(output_metadata_path).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    # Format selector preferring crisp wide gameplay up to 1440p/2160p
    format_selector = (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bestvideo+bestaudio/best"
    )
    temp_tmpl = str(out_video.parent / f"download_{out_video.stem}.%(ext)s")

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
        logger.info("Using authenticated cookies for yt-dlp: %s", Path(cookie_file).name)

    # Check if video already downloaded to save bandwidth
    download_needed = not (out_video.is_file() and out_video.stat().st_size > 1024 * 1024)

    logger.info("Fetching video info & heatmap for: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=download_needed)
        if info_dict is None:
            raise RuntimeError(f"Could not extract info for URL: {url}")

        if download_needed:
            downloaded_file = Path(ydl.prepare_filename(info_dict))
            merged_mp4 = downloaded_file.with_suffix(".mp4")
            actual_file = merged_mp4 if merged_mp4.is_file() else downloaded_file

            if not actual_file.is_file():
                matches = list(out_video.parent.glob(f"download_{out_video.stem}.*"))
                if matches:
                    actual_file = matches[0]
                else:
                    raise FileNotFoundError(f"yt-dlp output file not found in {out_video.parent}")

            if actual_file.resolve() != out_video.resolve():
                if out_video.exists():
                    out_video.unlink()
                shutil.move(str(actual_file), str(out_video))
        else:
            logger.info("Using existing source video at %s (%.2f MB)", out_video.name, out_video.stat().st_size / (1024 * 1024))

    # Parse and extract heatmap data
    raw_heatmap = info_dict.get("heatmap") or []
    hotspots = parse_youtube_heatmap(raw_heatmap)
    if hotspots:
        logger.info("Extracted %d YouTube 'Most Replayed' heatmap hotspot(s).", len(hotspots))
    else:
        logger.warning("YouTube heatmap data unavailable for this video. Smoothly falling back to audio spikes.")

    metadata = {
        "video_id": info_dict.get("id", ""),
        "title": info_dict.get("title", "YouTube Gaming Video"),
        "duration": float(info_dict.get("duration") or 0.0),
        "uploader": info_dict.get("uploader", ""),
        "view_count": info_dict.get("view_count", 0),
        "has_heatmap": bool(hotspots),
        "hotspots": hotspots,
        "video_path": str(out_video),
    }

    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved video metadata and heatmap to %s", out_meta.name)
    return metadata


# ==============================================================================
# 4. CPU-Optimized English Word-Level Transcription
# ==============================================================================

def extract_16k_mono_audio(
    video_path: Path,
    output_audio_path: Path,
    ffmpeg_bin: Optional[str] = None,
) -> Path:
    """Extract lightweight 16kHz mono WAV audio via FFmpeg."""
    v_in = Path(video_path).resolve()
    a_out = Path(output_audio_path).resolve()
    a_out.parent.mkdir(parents=True, exist_ok=True)

    if not v_in.is_file():
        raise FileNotFoundError(f"Source video not found: {v_in}")

    if a_out.is_file() and a_out.stat().st_size > 1024:
        logger.info("Found existing 16kHz audio: %s (%.2f MB). Using cached file.", a_out.name, a_out.stat().st_size / (1024 * 1024))
        return a_out

    bin_path = ffmpeg_bin or get_ffmpeg_binary()
    cmd = [
        bin_path,
        "-y",
        "-i", str(v_in),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(a_out),
    ]

    logger.info("Extracting 16kHz mono audio: %s -> %s...", v_in.name, a_out.name)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed (exit {proc.returncode}): {proc.stderr[:300]}")

    if not a_out.is_file() or a_out.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction produced zero-byte file at {a_out}")

    logger.info("Successfully extracted 16kHz audio: %s (%.2f MB)", a_out.name, a_out.stat().st_size / (1024 * 1024))
    return a_out


def transcribe_english_words(
    audio_path: Path,
    output_json_path: Path,
    model_size: str = "small.en",
) -> Dict[str, Any]:
    """Transcribe audio using faster-whisper on CPU with real word-level timestamps.

    VULNERABILITY PREVENTION:
      - Strictly prohibits mock or dummy subtitles.
      - If faster-whisper is not installed or returns 0 words, aborts immediately.
    """
    a_path = Path(audio_path).resolve()
    out_json = Path(output_json_path).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not a_path.is_file():
        raise FileNotFoundError(f"Audio file not found for transcription: {a_path}")

    # If transcript already exists and is non-empty with real words, reuse
    if out_json.is_file():
        try:
            cached_data = json.loads(out_json.read_text(encoding="utf-8"))
            words = [
                w for seg in cached_data.get("segments", [])
                for w in seg.get("words", [])
            ]
            if words:
                logger.info("Found existing verified transcript with %d words at %s", len(words), out_json.name)
                return cached_data
        except Exception:
            pass

    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Real word-level transcription requires faster-whisper. "
            "Mock or dummy subtitles are strictly prohibited."
        )

    logger.info("Initializing faster-whisper '%s' (device=cpu, compute_type=int8)...", model_size)
    try:
        model = WhisperModel(
            model_size_or_path=model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load faster-whisper model '{model_size}': {exc}") from exc

    logger.info("Transcribing '%s' with vad_filter=True, word_timestamps=True...", a_path.name)
    segments_iter, info = model.transcribe(
        str(a_path),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
        language="en",
    )

    collected_segments: List[Dict[str, Any]] = []
    total_words = 0

    for seg in segments_iter:
        words_data: List[Dict[str, Any]] = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                cleaned_word = w.word.strip()
                if cleaned_word:
                    words_data.append({
                        "word": cleaned_word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(getattr(w, "probability", 1.0), 3),
                    })
                    total_words += 1

        seg_text = seg.text.strip()
        collected_segments.append({
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg_text,
            "words": words_data,
        })

    # Zero words check (strictly prevents mock fallbacks)
    if total_words == 0:
        raise RuntimeError(
            f"Transcription yielded 0 words for '{a_path.name}'. "
            "Cannot generate viral Shorts without authentic speech."
        )

    transcript_data = {
        "duration": getattr(info, "duration", 0.0),
        "language": "en",
        "total_words_count": total_words,
        "segments": collected_segments,
    }

    out_json.write_text(json.dumps(transcript_data, indent=2), encoding="utf-8")
    logger.info("Transcription completed: %d segments, %d words saved to %s", len(collected_segments), total_words, out_json.name)
    return transcript_data


# ==============================================================================
# 5. Loudness & Excitement Peak Detection (librosa)
# ==============================================================================

def detect_loudness_spikes(
    audio_path: Union[str, Path],
    top_n: int = 10,
    hop_length: int = 512,
) -> List[Dict[str, Any]]:
    """Analyze audio with librosa (RMS energy + onset strength) for loud reaction spikes.

    Args:
        audio_path: Path to 16kHz mono WAV file.
        top_n: Maximum number of peak moments to extract.
        hop_length: Frame hop length.

    Returns:
        List[Dict[str, Any]]: High-energy reaction moments with timestamp, rms, and score.
    """
    path = Path(audio_path).resolve()
    if not path.is_file():
        logger.warning("Audio file not found for librosa analysis: %s", path)
        return []

    if librosa is None or np is None:
        logger.warning("librosa/numpy not available. Skipping audio spike detection.")
        return []

    try:
        logger.info("Analyzing loudness & excitement peaks with librosa: %s...", path.name)
        y, sr = librosa.load(str(path), sr=16000, mono=True)
        if len(y) == 0:
            return []

        # 1. Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # 2. Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Ensure matching length
        min_len = min(len(rms), len(onset_env))
        rms = rms[:min_len]
        onset_env = onset_env[:min_len]

        # 3. Combine normalized RMS + onset strength
        max_rms = float(np.max(rms)) if np.max(rms) > 0 else 1e-6
        max_onset = float(np.max(onset_env)) if np.max(onset_env) > 0 else 1e-6

        norm_rms = rms / max_rms
        norm_onset = onset_env / max_onset
        combined_energy = 0.60 * norm_rms + 0.40 * norm_onset

        # Frame timestamps
        times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)

        # Sliding window peak detection (min spacing: 4.0 seconds)
        min_dist_frames = int(4.0 * sr / hop_length)
        peaks: List[Tuple[float, float, float]] = []

        # Extract local peaks
        for i in range(1, min_len - 1):
            if combined_energy[i] > combined_energy[i - 1] and combined_energy[i] > combined_energy[i + 1]:
                if combined_energy[i] >= 0.45:  # High excitement threshold
                    peaks.append((times[i], float(rms[i]), float(combined_energy[i])))

        # Sort descending by combined energy score
        peaks.sort(key=lambda x: x[2], reverse=True)

        # Non-maximum suppression with 4.0s minimum distance
        selected_spikes: List[Dict[str, Any]] = []
        for t, r_val, c_score in peaks:
            if not any(abs(t - s["time_sec"]) < 4.0 for s in selected_spikes):
                reaction_label = "chaos_or_scream" if c_score >= 0.75 else "loud_reaction"
                selected_spikes.append({
                    "time_sec": round(float(t), 2),
                    "rms_energy": round(float(r_val), 3),
                    "intensity": round(float(c_score), 2),
                    "type": reaction_label,
                })
                if len(selected_spikes) >= top_n:
                    break

        selected_spikes.sort(key=lambda s: s["time_sec"])
        logger.info("Identified %d librosa excitement/loudness spikes.", len(selected_spikes))
        return selected_spikes

    except Exception as exc:
        logger.warning("Error during librosa peak detection: %s. Continuing without audio spikes.", exc)
        return []


# ==============================================================================
# 6. Word Boundary Snapping
# ==============================================================================

def snap_to_word_boundaries(
    proposed_start: float,
    proposed_end: float,
    words: List[Dict[str, Any]],
    total_duration: Optional[float] = None,
    min_sec: float = 18.0,
    max_sec: float = 48.0,
    lead_in: float = 0.25,
    tail_padding: float = 0.35,
) -> Tuple[float, float]:
    """Snap proposed clip timestamps to real word boundaries with natural lead-in/padding.

    Boundary Rules:
      - Begin 0.25s prior to first word start (never mid-sentence).
      - End 0.35s after final word end (never cut off punchline or laugh).
      - Clamp strictly to [min_sec, max_sec] and [0, total_duration].
    """
    if not words:
        dur = max(min_sec, min(max_sec, proposed_end - proposed_start))
        start = max(0.0, proposed_start)
        end = start + dur
        if total_duration and end > total_duration:
            end = total_duration
            start = max(0.0, end - dur)
        return round(start, 2), round(end, 2)

    # 1. Snap Start: find word closest to proposed_start
    best_start_word = min(words, key=lambda w: abs(w["start"] - proposed_start))
    snapped_start = max(0.0, best_start_word["start"] - lead_in)

    # 2. Target end time within [min_sec, max_sec]
    target_end = max(snapped_start + min_sec, proposed_end)
    target_end = min(snapped_start + max_sec, target_end)

    # 3. Snap End: find word whose end is closest to target_end
    eligible_end_words = [w for w in words if w["end"] >= snapped_start + min_sec - 1.0]
    if eligible_end_words:
        best_end_word = min(eligible_end_words, key=lambda w: abs(w["end"] - target_end))
        snapped_end = best_end_word["end"] + tail_padding
    else:
        snapped_end = target_end + tail_padding

    # 4. Enforce duration bounds (18s - 48s)
    dur = snapped_end - snapped_start
    if dur < min_sec:
        snapped_end = snapped_start + min_sec
    elif dur > max_sec:
        snapped_end = snapped_start + max_sec

    # 5. Strict clamping within total video duration
    if total_duration and total_duration > 0:
        if snapped_end > total_duration:
            snapped_end = total_duration
            snapped_start = max(0.0, snapped_end - min(max_sec, max(min_sec, dur)))

    return round(snapped_start, 2), round(snapped_end, 2)


# ==============================================================================
# 7. Gemini 2.5 Flash Editorial Director
# ==============================================================================

def load_system_instruction(skill_file_path: Optional[Union[str, Path]] = None) -> str:
    """Load system instructions from skills/MOOD_EDITOR_SKILL.md."""
    target_path = Path(skill_file_path) if skill_file_path else DEFAULT_SKILL_PATH
    target_path = target_path.resolve()

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Skill file '{target_path.name}' not found at {target_path}. "
            "Ensure 'skills/MOOD_EDITOR_SKILL.md' exists."
        )

    return target_path.read_text(encoding="utf-8")


def sanitize_gemini_json(raw_text: str) -> Any:
    """Extract and sanitize JSON from Gemini responses, eliminating markdown fences."""
    cleaned = raw_text.strip()

    # Strip code block wrappers
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Array extraction [ { ... } ]
    array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    # Single object extraction { ... }
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if "clips" in parsed and isinstance(parsed["clips"], list):
                return parsed["clips"]
            return [parsed]
        except Exception:
            pass

    raise json.JSONDecodeError(
        f"Could not parse valid JSON from Gemini output: {raw_text[:300]}...",
        doc=raw_text,
        pos=0,
    )


def format_director_prompt(
    transcript_segments: List[Dict[str, Any]],
    hotspots: List[Dict[str, Any]],
    audio_spikes: List[Dict[str, Any]],
    video_title: str,
    total_duration: float,
    max_clips: int = 3,
) -> str:
    """Format rich multimodal editorial prompt for Gemini 2.5 Flash."""
    sections = []

    # Section 1: Video Context
    sections.append(
        f"=== VIDEO CONTEXT ===\n"
        f"Title: {video_title}\n"
        f"Total Duration: {total_duration:.1f}s ({int(total_duration // 60)}m {int(total_duration % 60)}s)\n"
        f"Target Clips: Find up to {max_clips} independent viral Shorts (18s to 48s each)\n"
        f"Audience: Strictly English US Gaming / GTA mobile viewers"
    )

    # Section 2: YouTube Heatmap Hotspots
    if hotspots:
        hm_lines = ["=== YOUTUBE 'MOST REPLAYED' HEATMAP HOTSPOTS ==="]
        for h in hotspots:
            hm_lines.append(
                f"- [{h['start_sec']:.1f}s - {h['end_sec']:.1f}s] Replay Intensity: {h['intensity']:.2f}"
            )
        sections.append("\n".join(hm_lines))

    # Section 3: Librosa Audio Loudness Spikes
    if audio_spikes:
        spike_lines = ["=== LIBROSA AUDIO REACTION & LOUDNESS SPIKES ==="]
        for s in audio_spikes:
            spike_lines.append(
                f"- At {s['time_sec']:.1f}s: {s['type'].upper()} (Intensity: {s['intensity']:.2f})"
            )
        sections.append("\n".join(spike_lines))

    # Section 4: Word-Timestamped Speech Transcript
    tr_lines = ["=== TIMESTAMPED SPEECH TRANSCRIPT ==="]
    for seg in transcript_segments:
        tr_lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    sections.append("\n".join(tr_lines))

    # Section 5: Editorial Directives
    sections.append(
        "=== EDITORIAL INSTRUCTIONS ===\n"
        "Apply the virality principles from skills/MOOD_EDITOR_SKILL.md:\n"
        "1. Identify segments where high-retention speech intersects with YouTube heatmap peaks or audio spikes.\n"
        "2. Ensure each clip contains a complete Story Arc (Setup -> Escalation -> Payoff) between 18s and 48s.\n"
        "3. Generate a 2-4 word urgent 'hook_kicker' (e.g. 'HOW DID THIS HAPPEN?! 🔥') and punchy 'main_title'.\n"
        "4. Assign 'overall_mood' ('funny' | 'sigma_crazy' | 'hype').\n"
        "5. Plan 'mood_events' (laugh / sigma_crazy / shock_impact) with relative SFX timing (pop, boom, whoosh, hit).\n"
        "6. Select key 'highlight_words' with color tags ('red' for crazy/danger, 'yellow' for hype/money, 'green' for win).\n"
        "Return strictly valid JSON matching the requested schema."
    )

    return "\n\n".join(sections)


def find_hotspot_moments(
    transcript_path: Union[str, Path],
    metadata_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    max_clips: int = 3,
    api_key: Optional[str] = None,
    skill_doc_path: Optional[Union[str, Path]] = None,
) -> List[HotspotClip]:
    """Execute Gemini 2.5 Flash editorial analysis grounded in MOOD_EDITOR_SKILL.md."""
    t_file = Path(transcript_path).resolve()
    m_file = Path(metadata_path).resolve()

    if not t_file.is_file():
        raise FileNotFoundError(f"Transcript JSON not found: {t_file}")
    if not m_file.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {m_file}")

    transcript_data = json.loads(t_file.read_text(encoding="utf-8"))
    metadata = json.loads(m_file.read_text(encoding="utf-8"))

    segments = transcript_data.get("segments", [])
    flat_words = [w for seg in segments for w in seg.get("words", [])]

    # Vulnerability Prevention: Zero words check
    if not flat_words:
        raise RuntimeError(
            f"Transcript '{t_file.name}' has 0 spoken words. "
            "Cannot run viral hotspot analysis without authentic speech."
        )

    # 1. Detect audio spikes via librosa if audio available
    audio_spikes: List[Dict[str, Any]] = []
    if audio_path and Path(audio_path).is_file():
        audio_spikes = detect_loudness_spikes(audio_path, top_n=8)

    hotspots = metadata.get("hotspots", [])
    video_title = metadata.get("title", "GTA 6 Gameplay")
    total_duration = metadata.get("duration", transcript_data.get("duration", 0.0))

    # 2. Verify Gemini API Key
    active_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not active_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Real viral hook and hotspot analysis requires Gemini 2.5 Flash."
        )

    if genai is None:
        raise RuntimeError("google-genai SDK is not installed. Install via: pip install google-genai")

    # 3. Load System Instructions
    system_instruction = load_system_instruction(skill_doc_path)

    # 4. Build Prompt
    user_prompt = format_director_prompt(
        transcript_segments=segments,
        hotspots=hotspots,
        audio_spikes=audio_spikes,
        video_title=video_title,
        total_duration=total_duration,
        max_clips=max_clips,
    )

    logger.info("Calling Gemini 2.5 Flash Editorial Director...")
    client = genai.Client(api_key=active_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        raw_text = response.text or ""
        logger.info("Received Gemini 2.5 Flash response (%d characters)", len(raw_text))
    except Exception as exc:
        logger.error("Gemini 2.5 Flash request failed: %s", exc)
        raise RuntimeError(f"Gemini 2.5 Flash editorial analysis failed: {exc}") from exc

    # 5. Sanitize and Parse JSON
    parsed_json = sanitize_gemini_json(raw_text)
    if not isinstance(parsed_json, list):
        parsed_json = [parsed_json]

    validated_clips: List[HotspotClip] = []
    for item in parsed_json:
        if not isinstance(item, dict):
            continue

        raw_start = float(item.get("start_sec", 0.0))
        raw_end = float(item.get("end_sec", raw_start + 30.0))

        # Programmatic Word Boundary Snapping
        snapped_start, snapped_end = snap_to_word_boundaries(
            proposed_start=raw_start,
            proposed_end=raw_end,
            words=flat_words,
            total_duration=total_duration,
            min_sec=18.0,
            max_sec=48.0,
        )

        item["start_sec"] = snapped_start
        item["end_sec"] = snapped_end

        try:
            clip_obj = HotspotClip.model_validate(item)
            validated_clips.append(clip_obj)
        except Exception as err:
            logger.warning("Skipping invalid clip from Gemini: %s (error: %s)", item, err)

    if not validated_clips:
        raise RuntimeError("No valid viral hotspot clips were identified by Gemini 2.5 Flash.")

    logger.info("Successfully discovered %d viral hotspot clip(s).", len(validated_clips))
    return validated_clips[:max_clips]


# ==============================================================================
# 8. Master CLI Orchestrator
# ==============================================================================

def run_hotspot_pipeline(
    url: str = DEFAULT_TARGET_URL,
    work_dir: Path = DEFAULT_WORK_DIR,
    max_clips: int = 3,
    max_height: int = 2160,
    model_size: str = "small.en",
    cookies_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute complete end-to-end viral hotspot detection pipeline."""
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = get_ffmpeg_binary()

    source_video = work / "source.mp4"
    metadata_json = work / "metadata.json"
    audio_wav = work / "audio_16k.wav"
    transcript_json = work / "transcript.json"
    hotspots_out = work / "hotspot_clips.json"

    logger.info("=" * 80)
    logger.info("STARTING VIRAL HOTSPOT DETECTION PIPELINE")
    logger.info("URL:         %s", url)
    logger.info("Work Dir:    %s", work)
    logger.info("FFmpeg:      %s", ffmpeg_bin)
    logger.info("=" * 80)

    # 1. Download Video & YouTube Heatmap
    with secure_cookie_context(cookies_path, work_dir=work) as active_cookies:
        metadata = download_source_and_metadata(
            url=url,
            output_video_path=source_video,
            output_metadata_path=metadata_json,
            max_height=max_height,
            cookie_file=active_cookies,
            ffmpeg_bin=ffmpeg_bin,
        )

        # 2. Extract 16kHz Audio
        extract_16k_mono_audio(
            video_path=source_video,
            output_audio_path=audio_wav,
            ffmpeg_bin=ffmpeg_bin,
        )

        # 3. Transcribe English Speech
        transcribe_english_words(
            audio_path=audio_wav,
            output_json_path=transcript_json,
            model_size=model_size,
        )

    # 4. Run Gemini 2.5 Flash Editorial Director
    clips = find_hotspot_moments(
        transcript_path=transcript_json,
        metadata_path=metadata_json,
        audio_path=audio_wav,
        max_clips=max_clips,
        api_key=api_key,
    )

    clips_dicts = [c.model_dump() for c in clips]
    hotspots_out.write_text(json.dumps(clips_dicts, indent=2), encoding="utf-8")
    logger.info("Saved %d hotspot clips to %s", len(clips_dicts), hotspots_out.name)

    print("\n" + "=" * 80)
    print("VIRAL HOTSPOT ANALYSIS COMPLETE")
    print("=" * 80)
    for i, c in enumerate(clips, 1):
        dur = c.end_sec - c.start_sec
        print(f"[{i}] {c.hook_kicker} | {c.main_title}")
        print(f"    Timestamps: {c.start_sec:.2f}s -> {c.end_sec:.2f}s (Duration: {dur:.1f}s)")
        print(f"    Viral Score: {c.viral_score}/100 | Mood: {c.overall_mood}")
        print(f"    Reason: {c.reason}")
        print(f"    Mood Events: {len(c.mood_events)} cues | Highlight Words: {len(c.highlight_words)}")
        print("-" * 80)

    return clips_dicts


def main():
    parser = argparse.ArgumentParser(description="Viral Hotspot & Editorial Director Engine")
    parser.add_argument("--url", type=str, default=DEFAULT_TARGET_URL, help="Target YouTube URL")
    parser.add_argument("--work-dir", type=str, default="./work", help="Working directory for artifacts")
    parser.add_argument("--max-clips", type=int, default=3, help="Max clips to return (default: 3)")
    parser.add_argument("--max-height", type=int, default=2160, help="Max download height (default: 2160)")
    parser.add_argument("--model", type=str, default="small.en", help="faster-whisper model size (default: small.en)")
    parser.add_argument("--cookies", type=str, default=None, help="Path to cookies.txt")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key override")

    args = parser.parse_args()
    try:
        run_hotspot_pipeline(
            url=args.url,
            work_dir=Path(args.work_dir),
            max_clips=args.max_clips,
            max_height=args.max_height,
            model_size=args.model,
            cookies_path=args.cookies,
            api_key=args.api_key,
        )
    except Exception as exc:
        logger.error("[FATAL ERROR] Viral hotspot pipeline aborted: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
