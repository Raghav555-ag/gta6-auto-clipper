"""Dynamic Captioning and Hook Banner Burning Module (captioner.py).

Generates high-contrast kinetic karaoke subtitles and on-screen hook banners
for 1080x1920 vertical Shorts using Advanced SubStation Alpha (.ass) scripts
and FFmpeg burning filters.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("captioner")


# ==============================================================================
# 1. FFmpeg Binary Discovery
# ==============================================================================

def get_ffmpeg_binary() -> str:
    """Locate functional FFmpeg executable."""
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

    raise RuntimeError("FFmpeg executable not found. Please ensure FFmpeg is on PATH.")


# ==============================================================================
# 2. ASS Subtitle Formatting & Timestamps
# ==============================================================================

@dataclass
class CaptionWord:
    start: float
    end: float
    text: str


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cs."""
    sec = max(0.0, seconds)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def get_preferred_font() -> str:
    """Return high-readability cross-platform sans-serif font."""
    return "Arial, DejaVu Sans, Liberation Sans, Impact, sans-serif"


def escape_ffmpeg_filter_path(filepath: Union[str, Path]) -> str:
    """Escape file path for FFmpeg subtitles filter on Windows/Linux."""
    path_str = str(Path(filepath).resolve()).replace("\\", "/")
    # Escape colon after drive letter: C:/ -> C\\:/
    path_str = re.sub(r"^([a-zA-Z]):/", r"\1\\:/", path_str)
    # Escape single quotes and brackets
    path_str = path_str.replace("'", "\\'").replace("[", "\\[").replace("]", "\\]")
    return path_str


# ==============================================================================
# 3. Kinetic Karaoke ASS Generation
# ==============================================================================

def generate_ass_script(
    words: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_ass_path: Union[str, Path],
    hook_banner: Optional[str] = None,
    words_per_chunk: int = 3,
) -> Tuple[Path, int]:
    """Generate .ass script with 0-3s curiosity hook banner and karaoke subtitles.

    Args:
        words: List of word tokens with 'start', 'end', 'word'/'text'.
        clip_start: Start timestamp of the video slice in seconds.
        clip_end: End timestamp of the video slice in seconds.
        output_ass_path: Output .ass file destination.
        hook_banner: 4-7 word curiosity headline for first 3 seconds.
        words_per_chunk: Number of words per subtitle chunk (default: 3).

    Returns:
        Tuple[Path, int]: Path to .ass file and count of dialogue cues generated.
    """
    ass_path = Path(output_ass_path).resolve()
    ass_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter words strictly within this clip's time window
    clip_words: List[CaptionWord] = []
    for w in words:
        start = float(w.get("start", 0.0))
        end = float(w.get("end", 0.0))
        text = str(w.get("word") or w.get("text") or "").strip()

        if text and (start >= clip_start - 0.25) and (end <= clip_end + 0.35):
            rel_start = max(0.0, start - clip_start)
            rel_end = max(rel_start + 0.1, end - clip_start)
            clip_words.append(CaptionWord(start=rel_start, end=rel_end, text=text.upper()))

    font = get_preferred_font()

    # ASS Header & Styles
    # PlayRes: 1080x1920 (9:16 vertical canvas)
    # Default style: Bottom Center (Alignment 2), MarginV: 380, White text, Black outline, Yellow karaoke
    # Banner style: Top Center (Alignment 8), MarginV: 180, Bright Yellow text, Thick Black Outline
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Default Subtitles Style
        f"Style: Default,{font},74,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,"
        "-1,0,0,0,100,100,1,0,1,8,4,2,60,60,380,1",
        # Hook Banner Style (Top Display)
        f"Style: HookBanner,{font},82,&H0000FFFF,&H00FFFFFF,&H00000000,&HA0000000,"
        "-1,0,0,0,100,100,2,0,1,10,6,8,50,50,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    cues_count = 0

    # 1. Add Hook Banner Cue (First 3.0 Seconds)
    if hook_banner and hook_banner.strip():
        clean_banner = hook_banner.strip().upper()
        # Display from 0.0s to 3.2s
        b_start = format_ass_time(0.0)
        b_end = format_ass_time(min(3.2, clip_end - clip_start))
        # Add subtle pop-in scale effect
        banner_text = f"{{\\fad(150,200)}}{clean_banner}"
        lines.append(f"Dialogue: 1,{b_start},{b_end},HookBanner,,0,0,0,,{banner_text}")
        cues_count += 1

    # 2. Add Kinetic Karaoke Subtitle Chunks
    for i in range(0, len(clip_words), words_per_chunk):
        chunk = clip_words[i : i + words_per_chunk]
        if not chunk:
            continue

        c_start = chunk[0].start
        c_end = chunk[-1].end

        # Karaoke line generation: {\k<cs>}Word
        karaoke_parts = []
        for w in chunk:
            dur_cs = int(round((w.end - w.start) * 100))
            dur_cs = max(10, min(200, dur_cs))
            karaoke_parts.append(f"{{\\k{dur_cs}}}{w.text}")

        line_text = " ".join(karaoke_parts)
        t_start = format_ass_time(c_start)
        t_end = format_ass_time(c_end)

        lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{line_text}")
        cues_count += 1

    ass_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated kinetic ASS script: %s (%d dialogue cues)", ass_path.name, cues_count)
    return ass_path, cues_count


# ==============================================================================
# 4. Burn Captions onto Video via FFmpeg
# ==============================================================================

def burn_captions(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    transcript_path: Union[str, Path],
    clip_start: float,
    clip_end: float,
    hook_banner: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Burn kinetic subtitles and curiosity hook banner onto a 9:16 framed video.

    VULNERABILITY PREVENTION:
    - Verifies transcript words exist for the specified time window.
    - Uses robust cross-platform path escaping for FFmpeg subtitle filter.
    - Falls back cleanly to uncaptioned video if subtitle filter fails.

    Args:
        video_path: Source framed video file.
        output_path: Destination path for captioned video.
        transcript_path: Path to transcript.json containing word timestamps.
        clip_start: Segment start time in seconds.
        clip_end: Segment end time in seconds.
        hook_banner: Optional 4-7 word headline for the first 3 seconds.
        ffmpeg_bin: Optional FFmpeg executable override.
        crf: Encoding CRF (default: 18).
        preset: x264 preset.

    Returns:
        Path: Path to final captioned video.
    """
    v_in = Path(video_path).resolve()
    v_out = Path(output_path).resolve()
    t_path = Path(transcript_path).resolve()
    v_out.parent.mkdir(parents=True, exist_ok=True)

    if not v_in.is_file():
        raise FileNotFoundError(f"Input video not found: {v_in}")

    if not t_path.is_file():
        logger.warning("Transcript file not found: %s. Skipping captions.", t_path)
        if v_in != v_out:
            shutil.copy2(v_in, v_out)
        return v_out

    # 1. Parse Word Timestamps from transcript.json
    with open(t_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)

    flat_words: List[Dict[str, Any]] = []
    # Check if transcript has segments with words list
    segments = t_data.get("segments", []) if isinstance(t_data, dict) else t_data
    for seg in segments:
        if isinstance(seg, dict):
            if "words" in seg and isinstance(seg["words"], list) and seg["words"]:
                for w in seg["words"]:
                    flat_words.append(w)
            else:
                # Fallback to segment-level token
                flat_words.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "word": seg.get("text", ""),
                })

    # 2. Generate .ass file
    ass_file = v_out.parent / f"sub_{v_out.stem}.ass"
    ass_path, cues_count = generate_ass_script(
        words=flat_words,
        clip_start=clip_start,
        clip_end=clip_end,
        output_ass_path=ass_file,
        hook_banner=hook_banner,
    )

    if cues_count == 0:
        logger.warning("No spoken words found within [%.1fs - %.1fs]. Proceeding without subtitle burn.", clip_start, clip_end)
        if v_in != v_out:
            shutil.copy2(v_in, v_out)
        return v_out

    bin_path = ffmpeg_bin or get_ffmpeg_binary()
    escaped_ass = escape_ffmpeg_filter_path(ass_path)

    # 3. Burn Subtitles via FFmpeg
    cmd = [
        bin_path,
        "-y",
        "-i", str(v_in),
        "-vf", f"ass='{escaped_ass}'",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(v_out),
    ]

    logger.info("Burning kinetic subtitles & hook banner: %s -> %s...", v_in.name, v_out.name)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        logger.warning(
            "FFmpeg 'ass' filter failed (code %d): %s. Attempting fallback with 'subtitles' filter...",
            proc.returncode,
            proc.stderr[:200],
        )
        # Secondary fallback: try subtitles filter
        cmd_sub = [
            bin_path,
            "-y",
            "-i", str(v_in),
            "-vf", f"subtitles='{escaped_ass}'",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(v_out),
        ]
        proc_sub = subprocess.run(cmd_sub, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc_sub.returncode != 0:
            logger.warning("Subtitle burning failed. Falling back to uncaptioned video.")
            if v_in != v_out:
                shutil.copy2(v_in, v_out)
            return v_out

    logger.info("Successfully burned captions: %s (%.2f MB)", v_out.name, v_out.stat().st_size / (1024 * 1024))
    return v_out


# ==============================================================================
# 5. CLI Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Burn kinetic subtitles and hook banners onto 9:16 Shorts")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input 9:16 video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output captioned video path")
    parser.add_argument("--transcript", "-t", type=str, required=True, help="Path to transcript.json")
    parser.add_argument("--start", type=float, required=True, help="Clip start timestamp in seconds")
    parser.add_argument("--end", type=float, required=True, help="Clip end timestamp in seconds")
    parser.add_argument("--banner", type=str, default=None, help="Hook banner headline for first 3 seconds")
    parser.add_argument("--crf", type=int, default=18, help="CRF encoding quality (default: 18)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        burn_captions(
            video_path=Path(args.input),
            output_path=Path(args.output),
            transcript_path=Path(args.transcript),
            clip_start=args.start,
            clip_end=args.end,
            hook_banner=args.banner,
            crf=args.crf,
        )
    except Exception as exc:
        logger.error("[FATAL ERROR] Captioning failed: %s", exc)
        sys.exit(1)
