"""Smart Video Framing and Dynamic Pacing Module (framer.py).

Fixes blurry crops, awkward camera angles, and boring slow pacing for 9:16 Shorts.
Implements:
1. Tight Pacing via auto-editor (--margin 0.25s) to eliminate dead air without clipping speech.
2. Smart Framing Modes:
   - 'blur_wide': Crisp 1080x1920 canvas with softly blurred background and centered gameplay,
                  leaving clean space for hook banners and animated captions without HUD cutoffs.
   - 'track': Smooth pan-and-scan tracking using EMA smoothing, dead-zone damper, and max pan speed.
   - 'split_facecam': Stacks creator facecam on top and gameplay below.
3. High-Fidelity Sharp Encoding (CRF 18, libx264, preset medium, yuv420p).
"""

import argparse
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("framer")


# ==============================================================================
# 1. FFmpeg & auto-editor Binary Discovery
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


def get_auto_editor_binary() -> Optional[str]:
    """Locate auto-editor executable if installed."""
    ae_path = shutil.which("auto-editor")
    if ae_path:
        return ae_path

    # Check python Scripts directory on Windows
    py_dir = Path(sys.executable).parent
    candidates = [
        py_dir / "Scripts" / "auto-editor.exe",
        py_dir / "auto-editor.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    return None


# ==============================================================================
# 2. Tight Pacing Optimization (auto-editor)
# ==============================================================================

def tighten_pacing(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    margin_sec: float = 0.25,
    threshold: str = "4%",
    auto_editor_bin: Optional[str] = None,
) -> Path:
    """Tighten video pacing using auto-editor to eliminate silent walking dead-air.

    Uses conservative margin (0.25s) and audio loudness threshold so jokes,
    speech, and action are never clipped.

    Args:
        input_path: Path to source video segment.
        output_path: Path to output tightened video.
        margin_sec: Margin in seconds around voiced audio (default: 0.25s).
        threshold: Silence detection threshold (default: '4%').
        auto_editor_bin: Path to auto-editor binary.

    Returns:
        Path: Path to tightened video (or original path if auto-editor is unavailable).
    """
    in_file = Path(input_path).resolve()
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not in_file.is_file():
        raise FileNotFoundError(f"Source video for pacing tightening not found: {in_file}")

    ae_bin = auto_editor_bin or get_auto_editor_binary()
    if not ae_bin:
        logger.warning("auto-editor binary not found. Skipping pace tightening and preserving original timing.")
        if in_file != out_file:
            shutil.copy2(in_file, out_file)
        return out_file

    logger.info(
        "Applying tight pacing to '%s' (margin=%.2fs, threshold=%s)...",
        in_file.name,
        margin_sec,
        threshold,
    )

    cmd = [
        ae_bin,
        str(in_file),
        "--margin", f"{margin_sec}s",
        "--edit", f"audio:threshold={threshold}",
        "-o", str(out_file),
        "--no-open",
        "--quiet",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and out_file.is_file() and out_file.stat().st_size > 1024:
            in_size = in_file.stat().st_size / (1024 * 1024)
            out_size = out_file.stat().st_size / (1024 * 1024)
            logger.info("Tightened pacing successfully: %s (%.2f MB -> %.2f MB)", out_file.name, in_size, out_size)
            return out_file
        else:
            logger.warning("auto-editor returned non-zero (%d): %s. Preserving original timing.", proc.returncode, proc.stderr[:200])
    except Exception as exc:
        logger.warning("auto-editor execution failed: %s. Continuing with original file.", exc)

    if in_file != out_file:
        shutil.copy2(in_file, out_file)
    return out_file


# ==============================================================================
# 3. Dynamic Smoothing Tracker (EMA + Dead-Zone + Max Pan Speed)
# ==============================================================================

@dataclass
class FrameBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0


def compute_smooth_pan_offsets(
    video_path: Path,
    crop_width: int,
    source_width: int,
    source_height: int,
    sample_every_n: int = 4,
    ema_alpha: float = 0.10,
    dead_zone_px: int = 40,
    max_pan_speed_px: float = 18.0,
    yolo_model_path: Optional[str] = None,
) -> List[int]:
    """Compute frame-by-frame horizontal crop offsets with cinematic glide damping.

    VULNERABILITY PREVENTION:
    - Strictly clamps crop coordinates within [0, source_width - crop_width].
    - Clamps delta per frame to max_pan_speed_px to prevent sudden whiplash pans.

    Args:
        video_path: Path to video.
        crop_width: Width of the 9:16 vertical crop window.
        source_width: Width of the source frame.
        source_height: Height of the source frame.
        sample_every_n: Sample detection every N frames for high performance.
        ema_alpha: Exponential smoothing factor (0.05 to 0.20).
        dead_zone_px: Tolerance threshold before camera begins moving.
        max_pan_speed_px: Maximum allowed horizontal pixel shift per frame.
        yolo_model_path: Optional path to YOLO weights.

    Returns:
        List[int]: Left x-coordinate crop offset for every frame.
    """
    max_x = max(0, source_width - crop_width)
    default_center_x = max_x // 2

    if cv2 is None:
        logger.warning("OpenCV not installed. Using static center crop.")
        return [default_center_x]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video '%s' for tracking analysis. Using center crop.", video_path.name)
        return [default_center_x]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Initialize YOLO if available, otherwise fallback to Haar Cascade / motion detector
    yolo_model = None
    if YOLO is not None:
        try:
            model_name = yolo_model_path or "yolov8n.pt"
            yolo_model = YOLO(model_name)
            logger.info("Loaded YOLOv8 model for player/action tracking: %s", model_name)
        except Exception as e:
            logger.info("YOLOv8 initialization skipped (%s). Using OpenCV motion tracking.", e)

    haar_cascade = None
    if yolo_model is None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
            if Path(cascade_path).is_file():
                haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

    current_crop_x = float(default_center_x)
    target_crop_x = float(default_center_x)
    offsets: List[int] = []

    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run heavy object/motion detection every N frames to maximize speed
        if frame_idx % sample_every_n == 0:
            detected_x: Optional[float] = None

            # 1. Try YOLO detection (Persons or vehicles)
            if yolo_model is not None:
                try:
                    results = yolo_model(frame, verbose=False, classes=[0, 2, 5, 7])
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        # Find box with highest confidence or largest area
                        best_box = None
                        best_area = 0
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                            if area > best_area:
                                best_area = area
                                best_box = xyxy
                        if best_box is not None:
                            center_subj = (best_box[0] + best_box[2]) / 2.0
                            detected_x = center_subj - (crop_width / 2.0)
                except Exception:
                    pass

            # 2. Try Haar Cascade detection
            if detected_x is None and haar_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bodies = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
                    if len(bodies) > 0:
                        # Pick largest body
                        largest = max(bodies, key=lambda b: b[2] * b[3])
                        center_subj = largest[0] + (largest[2] / 2.0)
                        detected_x = center_subj - (crop_width / 2.0)
                except Exception:
                    pass

            # 3. Fallback: Motion Centroid via Frame Differencing
            if detected_x is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    moments = cv2.moments(thresh)
                    if moments["m00"] > 1000:
                        motion_cx = moments["m10"] / moments["m00"]
                        detected_x = motion_cx - (crop_width / 2.0)
                prev_gray = gray

            if detected_x is not None:
                # Clamp proposed target strictly within [0, max_x]
                target_crop_x = max(0.0, min(float(max_x), detected_x))

        # Dead-zone Damper: only move camera if deviation exceeds threshold
        diff = target_crop_x - current_crop_x
        if abs(diff) > dead_zone_px:
            # Apply Exponential Moving Average (EMA)
            delta = diff * ema_alpha

            # Enforce max pan speed per frame (anti-whiplash safeguard)
            if abs(delta) > max_pan_speed_px:
                delta = math.copysign(max_pan_speed_px, delta)

            current_crop_x += delta

        # Strict clamping on final position
        clamped_x = int(round(max(0.0, min(float(max_x), current_crop_x))))
        offsets.append(clamped_x)
        frame_idx += 1

    cap.release()
    return offsets if offsets else [default_center_x]


# ==============================================================================
# 4. Smart Framing Rendering Layouts
# ==============================================================================

class SmartFramer:
    """Renders vertical 9:16 Shorts with professional layouts and sharp encoding."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or get_ffmpeg_binary()

    def render_blur_wide(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        crf: int = 18,
        preset: str = "medium",
        target_fps: Optional[float] = None,
    ) -> Path:
        """Render blur_wide layout: 1080x1920 canvas with softly blurred background and centered gameplay.

        Ensures gameplay, combat, mini-maps, and HUD are NEVER cropped, leaving ample
        top space for hook banners and bottom space for animated subtitles.

        Args:
            input_path: Source clip path.
            output_path: Destination 9:16 video path.
            crf: Constant Rate Factor (18 for sharp mastering).
            preset: x264 preset ('medium' or 'fast').
            target_fps: Output framerate.

        Returns:
            Path: Rendered video path.
        """
        in_file = Path(input_path).resolve()
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Rendering 'blur_wide' 9:16 Short: %s -> %s...", in_file.name, out_file.name)

        # Filtergraph:
        # 1. Scale background to fill 1080x1920, apply boxblur and slight darkening.
        # 2. Scale foreground gameplay to 1080px wide (maintaining aspect ratio).
        # 3. Center foreground over background.
        filter_complex = (
            "[0:v]split=2[bg_src][fg_src];"
            "[bg_src]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=25:5,eq=brightness=-0.08:contrast=0.95[bg];"
            "[fg_src]scale=1080:-2[fg];"
            "[bg][fg]overlay=0:(H-h)/2[outv]"
        )

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(in_file),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
        ]

        if target_fps:
            cmd.extend(["-r", str(target_fps)])

        cmd.append(str(out_file))

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("FFmpeg blur_wide rendering failed:\n%s", proc.stderr)
            raise RuntimeError(f"FFmpeg render failed (code {proc.returncode}): {proc.stderr[:400]}")

        logger.info("Successfully rendered blur_wide Short: %s (%.2f MB)", out_file.name, out_file.stat().st_size / (1024 * 1024))
        return out_file

    def render_track(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        crf: int = 18,
        preset: str = "medium",
        sample_every_n: int = 4,
        max_pan_speed_px: float = 18.0,
    ) -> Path:
        """Render track layout: Smooth dynamic pan-and-scan following main action/player.

        Args:
            input_path: Source clip path.
            output_path: Destination 9:16 video path.
            crf: Quality factor.
            preset: x264 preset.
            sample_every_n: Frame detection skip interval.
            max_pan_speed_px: Max pan speed damper.

        Returns:
            Path: Rendered video path.
        """
        in_file = Path(input_path).resolve()
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if cv2 is None or np is None:
            logger.warning("OpenCV/Numpy missing. Falling back to blur_wide mode.")
            return self.render_blur_wide(in_file, out_file, crf=crf, preset=preset)

        cap = cv2.VideoCapture(str(in_file))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for tracking: {in_file}")

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Vertical 9:16 crop width based on source height
        crop_h = src_h
        crop_w = int(crop_h * 9 / 16)
        crop_w = (crop_w // 2) * 2  # Enforce even dimension

        if crop_w >= src_w:
            # Source is already vertical or square: fall back to blur_wide
            return self.render_blur_wide(in_file, out_file, crf=crf, preset=preset)

        # 1. Compute smooth camera tracking path
        offsets = compute_smooth_pan_offsets(
            video_path=in_file,
            crop_width=crop_w,
            source_width=src_w,
            source_height=src_h,
            sample_every_n=sample_every_n,
            max_pan_speed_px=max_pan_speed_px,
        )

        # 2. Render frame-by-frame via OpenCV piped to FFmpeg for lossless audio preservation
        temp_raw_video = out_file.parent / f"temp_track_{out_file.name}"

        cap = cv2.VideoCapture(str(in_file))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_raw_video), fourcc, fps, (1080, 1920))

        frame_idx = 0
        total_offsets = len(offsets)

        logger.info("Executing smooth tracking crop on %s (%d frames)...", in_file.name, total_offsets)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Safe offset lookup
            off_x = offsets[frame_idx] if frame_idx < total_offsets else offsets[-1]
            # Strict clamping safeguard
            off_x = max(0, min(src_w - crop_w, off_x))

            # Crop 9:16 window
            cropped = frame[0:crop_h, off_x:off_x + crop_w]
            # Scale to 1080x1920
            scaled = cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
            writer.write(scaled)
            frame_idx += 1

        cap.release()
        writer.release()

        # 3. Mux original audio and re-encode sharply via FFmpeg
        mux_cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(temp_raw_video),
            "-i", str(in_file),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_file),
        ]

        proc = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if temp_raw_video.exists():
            temp_raw_video.unlink()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg muxing failed: {proc.stderr[:400]}")

        logger.info("Successfully rendered tracked Short: %s (%.2f MB)", out_file.name, out_file.stat().st_size / (1024 * 1024))
        return out_file

    def render_split_facecam(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        facecam_box: Tuple[int, int, int, int] = (0, 0, 480, 360),
        crf: int = 18,
        preset: str = "medium",
    ) -> Path:
        """Render split_facecam layout: Facecam on top, gameplay stacked below.

        Args:
            input_path: Source clip path.
            output_path: Destination 9:16 video path.
            facecam_box: Tuple of (x, y, width, height) of facecam in source video.
            crf: Quality factor.
            preset: x264 preset.

        Returns:
            Path: Rendered video path.
        """
        in_file = Path(input_path).resolve()
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fx, fy, fw, fh = facecam_box

        logger.info(
            "Rendering 'split_facecam' 9:16 Short: %s (facecam at x=%d, y=%d, w=%d, h=%d)...",
            in_file.name, fx, fy, fw, fh
        )

        # Top section: Facecam cropped and scaled to 1080x720 (3:2)
        # Bottom section: Gameplay scaled/cropped to 1080x1200
        filter_complex = (
            f"[0:v]split=2[cam_src][game_src];"
            f"[cam_src]crop={fw}:{fh}:{fx}:{fy},scale=1080:720:force_original_aspect_ratio=increase,crop=1080:720[cam];"
            f"[game_src]scale=1080:-2,pad=1080:1200:0:(1200-h)/2:black[game];"
            f"[cam][game]vstack=inputs=2[outv]"
        )

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(in_file),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_file),
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("FFmpeg split_facecam rendering failed:\n%s", proc.stderr)
            raise RuntimeError(f"FFmpeg split_facecam failed: {proc.stderr[:400]}")

        logger.info("Successfully rendered split_facecam Short: %s (%.2f MB)", out_file.name, out_file.stat().st_size / (1024 * 1024))
        return out_file

    def process_clip(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        layout: str = "blur_wide",
        tighten: bool = False,
        margin_sec: float = 0.25,
        threshold: str = "4%",
        crf: int = 18,
        preset: str = "medium",
        facecam_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Path:
        """Process candidate clip through pacing tightening and smart framing layout.

        Args:
            input_path: Source clip file path.
            output_path: Final rendered 1080x1920 Short path.
            layout: Framing mode ('blur_wide', 'track', or 'split_facecam').
            tighten: Whether to run auto-editor pacing tightening.
            margin_sec: auto-editor margin in seconds.
            threshold: auto-editor loudness threshold.
            crf: Encoding CRF (default: 18).
            preset: Encoding preset.
            facecam_box: Facecam coordinates for split_facecam.

        Returns:
            Path: Path to final rendered video.
        """
        in_file = Path(input_path).resolve()
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        working_input = in_file
        temp_tightened: Optional[Path] = None

        if tighten:
            temp_tightened = out_file.parent / f"tightened_{in_file.name}"
            working_input = tighten_pacing(
                input_path=in_file,
                output_path=temp_tightened,
                margin_sec=margin_sec,
                threshold=threshold,
            )

        try:
            layout_mode = layout.lower().strip()
            if layout_mode == "track":
                final_path = self.render_track(working_input, out_file, crf=crf, preset=preset)
            elif layout_mode == "split_facecam":
                box = facecam_box or (0, 0, 480, 360)
                final_path = self.render_split_facecam(working_input, out_file, facecam_box=box, crf=crf, preset=preset)
            else:
                # Default: blur_wide (crisp centered canvas with blurred background)
                final_path = self.render_blur_wide(working_input, out_file, crf=crf, preset=preset)

            return final_path

        finally:
            if temp_tightened and temp_tightened.exists() and temp_tightened != out_file:
                try:
                    temp_tightened.unlink()
                except Exception:
                    pass


# ==============================================================================
# 5. CLI Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart video framing and dynamic pacing for high-retention 9:16 Shorts"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video clip path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output 9:16 video path")
    parser.add_argument(
        "--layout",
        type=str,
        default="blur_wide",
        choices=["blur_wide", "track", "split_facecam"],
        help="Smart framing layout (default: blur_wide)",
    )
    parser.add_argument("--tighten", action="store_true", help="Run auto-editor tight pacing optimization")
    parser.add_argument("--margin", type=float, default=0.25, help="auto-editor margin in seconds (default: 0.25s)")
    parser.add_argument("--threshold", type=str, default="4%", help="auto-editor silence threshold (default: 4%)")
    parser.add_argument("--crf", type=int, default=18, help="Constant Rate Factor (default: 18 for sharp mastering)")
    parser.add_argument("--preset", type=str, default="medium", help="x264 encoding preset (default: medium)")
    parser.add_argument(
        "--facecam-box",
        type=str,
        default="0,0,480,360",
        help="Facecam box as 'x,y,w,h' for split_facecam mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    facecam_coords = None
    if args.facecam_box:
        try:
            parts = [int(p.strip()) for p in args.facecam_box.split(",")]
            if len(parts) == 4:
                facecam_coords = tuple(parts)
        except Exception:
            logger.warning("Could not parse --facecam-box '%s'. Using default (0,0,480,360).", args.facecam_box)

    framer = SmartFramer()
    try:
        framer.process_clip(
            input_path=Path(args.input),
            output_path=Path(args.output),
            layout=args.layout,
            tighten=args.tighten,
            margin_sec=args.margin,
            threshold=args.threshold,
            crf=args.crf,
            preset=args.preset,
            facecam_box=facecam_coords,
        )
    except Exception as exc:
        logger.error("[FATAL ERROR] Framer execution failed: %s", exc)
        sys.exit(1)
