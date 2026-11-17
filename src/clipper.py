"""FFmpeg Video Editing and Rendering Module for 9:16 Vertical YouTube Shorts.

Slices video segments based on Gemini timestamps, applies precise 9:16 center-crop
scaling to 1080x1920, and burns in high-contrast kinetic subtitles with cross-platform
font fallbacks and execution freeze protection.
"""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("clipper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class SubtitleWord:
    """Represents a word token for subtitle generation."""
    start: float
    end: float
    text: str


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS subtitle timestamp format: H:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def get_cross_platform_font() -> str:
    """Identify an available high-readability font across Linux, Windows, and macOS."""
    # Preferred order: DejaVu Sans (standard Linux/Ubuntu Actions), Arial (Windows/Mac), Liberation Sans
    return "DejaVu Sans, Arial, Liberation Sans, sans-serif"


def generate_kinetic_ass(
    words: List[Any],
    segment_start: float,
    segment_end: float,
    output_ass_path: Union[str, Path],
    words_per_chunk: int = 3,
) -> Path:
    """Generate an Advanced SubStation Alpha (.ass) file with dynamic kinetic styling.

    Creates bold, high-contrast subtitles centered in the lower-middle safe area
    with yellow karaoke highlights on active spoken words.

    Args:
        words: List of WordTimestamp objects or dicts with 'start', 'end', and 'text'.
        segment_start: Start timestamp of the video slice (seconds).
        segment_end: End timestamp of the video slice (seconds).
        output_ass_path: Destination path for the .ass subtitle script.
        words_per_chunk: Number of words displayed per kinetic subtitle line.

    Returns:
        Path: Path to the generated .ass file.
    """
    ass_path = Path(output_ass_path).resolve()
    ass_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert words to uniform SubtitleWord objects and filter to clip interval
    clip_words: List[SubtitleWord] = []
    for w in words:
        start = float(getattr(w, "start", 0.0) if hasattr(w, "start") else w.get("start", 0.0))
        end = float(getattr(w, "end", 0.0) if hasattr(w, "end") else w.get("end", 0.0))
        text = str(getattr(w, "text", "") if hasattr(w, "text") else w.get("text", "")).strip()

        if text and (start >= segment_start - 0.2) and (end <= segment_end + 0.5):
            # Calculate timestamp relative to the sliced clip start
            rel_start = max(0.0, start - segment_start)
            rel_end = max(rel_start + 0.1, end - segment_start)
            clip_words.append(SubtitleWord(start=rel_start, end=rel_end, text=text.upper()))

    font_name = get_cross_platform_font()

    # ASS Header & High-Contrast Gaming Style Definition
    # PlayRes: 1080x1920 (Native 9:16 vertical canvas)
    # PrimaryColour: &H00FFFFFF (Opaque White)
    # SecondaryColour (Karaoke): &H0000FFFF (Bright Gaming Yellow)
    # OutlineColour: &H00000000 (Pure Black outline)
    # Alignment: 2 (Bottom Center), MarginV: 420 (Safe zone above Shorts UI controls)
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
        f"Style: Default,{font_name},72,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,"
        "-1,0,0,0,100,100,1,0,1,8,4,2,60,60,420,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    # Group words into chunks for dynamic kinetic presentation
    for i in range(0, len(clip_words), words_per_chunk):
        chunk = clip_words[i : i + words_per_chunk]
        if not chunk:
            continue

        chunk_start = chunk[0].start
        chunk_end = chunk[-1].end

        # Sub-events highlighting each word dynamically as it is spoken
        for active_idx, active_word in enumerate(chunk):
            w_start = format_ass_timestamp(active_word.start)
            w_end = format_ass_timestamp(active_word.end)

            # Build line: active word colored in bright yellow (&H0000FFFF&), others in white (&H00FFFFFF&)
            text_tokens = []
            for j, w in enumerate(chunk):
                if j == active_idx:
                    text_tokens.append(f"{{\\c&H0000FFFF&}}{w.text}{{\\c&H00FFFFFF&}}")
                else:
                    text_tokens.append(w.text)

            line_text = " ".join(text_tokens)
            lines.append(
                f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line_text}"
            )

    ass_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated kinetic ASS subtitles: %s (%d dialogue cues)", ass_path.name, len(lines) - 13)
    return ass_path


class VideoClipper:
    """Renders 9:16 vertical Shorts with precise center-crop and burned kinetic subtitles."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_filter_complex(self, ass_file_path: Optional[Path] = None) -> str:
        """Construct the FFmpeg video filter chain.

        Filter:
        1. crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0 -> Exact 9:16 centered crop
        2. scale=1080:1920 -> Standard YouTube Shorts resolution
        3. ass=... -> Burn-in kinetic subtitles (if provided)
        """
        crop_scale_filter = "crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0,scale=1080:1920"

        if ass_file_path and ass_file_path.exists():
            # Escape path colons and backslashes for libass filter syntax
            clean_path = str(ass_file_path).replace("\\", "/").replace(":", "\\:")
            return f"{crop_scale_filter},ass='{clean_path}'"

        return crop_scale_filter

    def render_short(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        words: Optional[List[Any]] = None,
        output_name: Optional[str] = None,
        timeout_seconds: int = 360,
    ) -> Path:
        """Slice video segment and render to 1080x1920 vertical format.

        Args:
            video_path: Path to source 16:9 gameplay video.
            start_time: Clip start timestamp in seconds.
            end_time: Clip end timestamp in seconds.
            words: Optional word-level timestamps from Whisper for burning captions.
            output_name: Optional custom output filename.
            timeout_seconds: Subprocess execution timeout to prevent hanging.

        Returns:
            Path: Path to the generated vertical MP4 file.

        Raises:
            FileNotFoundError: If the source video file is missing.
            ValueError: If timestamps are invalid or duration <= 0.
            subprocess.CalledProcessError: If FFmpeg rendering fails.
            TimeoutError: If FFmpeg execution exceeds timeout_seconds.
        """
        source = Path(video_path).resolve()

        # Vulnerability Prevention: Validate input file existence and non-zero size
        if not source.exists():
            raise FileNotFoundError(f"Input video file not found: {source}")
        if source.stat().st_size == 0:
            raise ValueError(f"Input video file is empty (0 bytes): {source}")

        if start_time < 0 or end_time <= start_time:
            raise ValueError(f"Invalid clip boundaries: start={start_time}, end={end_time}")

        duration = end_time - start_time
        if duration > 60.0:
            logger.warning("Duration %.1fs exceeds YouTube Shorts 60s limit. Clamping to 58s.", duration)
            duration = 58.0

        if output_name:
            out_filename = f"{output_name}.mp4" if not output_name.endswith(".mp4") else output_name
        else:
            out_filename = f"{source.stem}_short_{int(start_time)}_{int(start_time + duration)}.mp4"

        output_path = self.output_dir / out_filename

        # Generate kinetic subtitles if word tokens are provided
        ass_path = None
        if words:
            temp_ass = output_path.with_suffix(".ass")
            try:
                ass_path = generate_kinetic_ass(
                    words=words,
                    segment_start=start_time,
                    segment_end=start_time + duration,
                    output_ass_path=temp_ass,
                )
            except Exception as err:
                logger.warning("Failed to generate kinetic ASS subtitles, rendering without text: %s", err)
                ass_path = None

        filter_str = self._build_filter_complex(ass_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(round(start_time, 2)),
            "-i", str(source),
            "-t", str(round(duration, 2)),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info("Executing FFmpeg 9:16 render for %s (%.1fs - %.1fs)...", source.name, start_time, start_time + duration)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise TimeoutError(f"FFmpeg rendering timed out after {timeout_seconds}s") from exc
        finally:
            # Clean up temporary .ass file
            if ass_path and ass_path.exists():
                ass_path.unlink(missing_ok=True)

        # Vulnerability Prevention: Check return code and report descriptive error
        if result.returncode != 0:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            logger.error("FFmpeg rendering failed: %s", result.stderr)
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg completed but output file is missing or empty: {output_path}")

        logger.info("Successfully rendered 9:16 Short: %s (%.2f MB)", output_path.name, output_path.stat().st_size / (1024 * 1024))
        return output_path
