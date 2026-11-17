"""Downloader module using yt-dlp to retrieve videos or stream VODs."""
import os
import subprocess
from pathlib import Path


class VideoDownloader:
    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, output_name: str = None) -> Path:
        """Download video at URL in highest 1080p MP4 quality."""
        out_template = (
            str(self.download_dir / f"{output_name}.%(ext)s")
            if output_name
            else str(self.download_dir / "%(id)s.%(ext)s")
        )
        
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", out_template,
            url
        ]
        
        print(f"Downloading: {url} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        # Find the resulting file
        expected_files = list(self.download_dir.glob(f"{output_name or '*'}.mp4"))
        if not expected_files:
            raise FileNotFoundError("Downloaded MP4 not found.")
        
        return expected_files[0]
