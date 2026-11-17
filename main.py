#!/usr/bin/env python3
"""GTA 6 Auto-Clipper Pipeline Entrypoint."""
import argparse
import sys
from pathlib import Path
from src.downloader import VideoDownloader
from src.detector import HighlightDetector
from src.clipper import VideoClipper
from src.uploader import YouTubeUploader


def main():
    parser = argparse.ArgumentParser(description="GTA 6 Auto-Clipper & Uploader Pipeline")
    parser.add_argument("--url", type=str, help="YouTube video or stream URL to process")
    parser.add_argument("--video-file", type=str, help="Local video file to process directly")
    parser.add_argument("--duration", type=int, default=45, help="Clip duration in seconds")
    parser.add_argument("--upload", action="store_true", help="Automatically upload generated clips to YouTube")
    args = parser.parse_args()

    if not args.url and not args.video_file:
        print("Usage error: Please provide either --url <YOUTUBE_URL> or --video-file <PATH>")
        sys.exit(1)

    print("========================================")
    print("GTA 6 AUTO-CLIPPER PIPELINE")
    print("========================================")

    # 1. Download
    downloader = VideoDownloader(download_dir="downloads")
    if args.url:
        video_path = downloader.download(args.url)
    else:
        video_path = Path(args.video_file)

    # 2. Detect Highlights
    detector = HighlightDetector(target_duration=args.duration)
    highlights = detector.detect_highlights(str(video_path))
    print(f"Found {len(highlights)} highlight segments.")

    # 3. Clip and Format into 9:16 Shorts
    clipper = VideoClipper(output_dir="output")
    uploader = YouTubeUploader()

    for idx, segment in enumerate(highlights, start=1):
        print(f"\nProcessing highlight {idx}: {segment.title} ({segment.start_time}s - {segment.end_time}s)")
        clip_path = clipper.create_short(str(video_path), segment, filename_prefix=f"gta6_clip_{idx}")
        print(f"Generated Short at: {clip_path}")

        # 4. Upload if requested
        if args.upload:
            title = f"{segment.title} #GTA6 #Shorts"
            description = f"Insane GTA 6 moment! Subscribe for more clips.\n\n#GTA6 #GTAOnline #Shorts #Gaming"
            uploader.upload_short(str(clip_path), title, description)

    print("\n[Done] Pipeline finished successfully.")


if __name__ == "__main__":
    main()
