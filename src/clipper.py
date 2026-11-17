"""Video clipping and vertical formatting module."""
import subprocess
from pathlib import Path
from .detector import HighlightSegment


class VideoClipper:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_short(self, video_path: str, segment: HighlightSegment, filename_prefix: str = "clip") -> Path:
        """Slice video segment and convert to 1080x1920 (9:16) vertical format."""
        out_path = self.output_dir / f"{filename_prefix}_{int(segment.start_time)}_{int(segment.end_time)}.mp4"
        duration = segment.end_time - segment.start_time

        # FFmpeg filter:
        # Scale to fit height 1920, center-crop to 1080 width (9:16 ratio)
        # or stacked blurred background + centered gameplay
        filter_str = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(segment.start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_path)
        ]

        print(f"Rendering vertical clip: {out_path} ...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg clipping failed: {res.stderr}")

        return out_path
