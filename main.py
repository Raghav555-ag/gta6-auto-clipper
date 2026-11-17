#!/usr/bin/env python3
"""GTA 6 Auto-Clipper Autonomous Pipeline Orchestrator.

Coordinates secure ingestion, faster-whisper transcription, Gemini 2.5 Flash
viral hook detection, FFmpeg 9:16 kinetic rendering, and YouTube Shorts publishing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.ai_analyzer import GeminiSegmentDetector
from src.clipper import VideoClipper
from src.downloader import VideoDownloader, secure_cookie_file
from src.transcriber import AudioTranscriber
from src.uploader import StateTracker, YouTubePublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def run_pipeline(
    custom_url: str = None,
    config_path: str = "config.json",
    ledger_path: str = "processed_videos.json",
    privacy_status: str = "public",
    max_clips_per_video: int = 2,
    keep_artifacts: bool = False,
):
    """Run the end-to-end auto-clipper pipeline."""
    logger.info("========================================")
    logger.info("STARTING GTA 6 AUTO-CLIPPER CLOUD RUN")
    logger.info("========================================")

    state_tracker = StateTracker(ledger_path=ledger_path)
    downloader = VideoDownloader(config_path=config_path)
    transcriber = AudioTranscriber(model_size="base.en", device="cpu", compute_type="int8")
    ai_detector = GeminiSegmentDetector()
    clipper = VideoClipper(output_dir="output")
    publisher = YouTubePublisher(ledger_path=ledger_path)

    # 1. Download target footage
    downloaded_videos = []
    if custom_url:
        logger.info("Manual URL override provided: %s", custom_url)
        with secure_cookie_file() as cookie_path:
            video_file = downloader._download_with_fallback(custom_url, cookie_path)
            downloaded_videos.append(video_file)
    else:
        logger.info("Polling creator sources from %s...", config_path)
        downloaded_videos = downloader.download_latest()

    if not downloaded_videos:
        logger.info("No new videos to process. Exiting cleanly.")
        return

    logger.info("Ingested %d video(s) for processing.", len(downloaded_videos))

    for video_path in downloaded_videos:
        video_file = Path(video_path).resolve()
        source_id = video_file.stem

        # Duplicate Prevention Check
        if state_tracker.is_processed(source_id):
            logger.info("Source video '%s' has already been processed. Skipping.", source_id)
            if not keep_artifacts and video_file.exists():
                video_file.unlink(missing_ok=True)
            continue

        logger.info("\n--- Processing Video: %s ---", video_file.name)

        try:
            # 2. Extract Audio & Transcribe
            logger.info("Step 1: Extracting audio and running faster-whisper transcription...")
            transcription = transcriber.process_video(str(video_file), keep_audio_file=False)

            if not transcription.words and not transcription.full_text:
                logger.warning("No speech detected in %s. Using default highlight windows.", video_file.name)

            # 3. AI Viral Hook Analysis with Gemini 2.5 Flash
            logger.info("Step 2: Analyzing transcript for US gaming viral moments with Gemini 2.5 Flash...")
            viral_segments = ai_detector.detect_viral_segments(
                transcript_text=transcription.full_text,
                total_duration=transcription.duration,
                video_title=video_file.stem,
                max_clips=max_clips_per_video,
            )

            logger.info("Identified %d viral candidate segment(s).", len(viral_segments))

            # 4. Vertical 9:16 Rendering & YouTube Publishing
            for idx, segment in enumerate(viral_segments, start=1):
                logger.info(
                    "Rendering Short #%d: '%s' (%.1fs - %.1fs)...",
                    idx,
                    segment.hook_title,
                    segment.start_time,
                    segment.end_time,
                )

                clip_name = f"{source_id}_clip_{idx}"
                rendered_clip = clipper.render_short(
                    video_path=str(video_file),
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    words=transcription.words,
                    output_name=clip_name,
                )

                # Step 5: Upload to YouTube Shorts
                try:
                    logger.info("Publishing Short #%d to YouTube...", idx)
                    short_id = publisher.upload_short(
                        video_path=str(rendered_clip),
                        source_video_id=f"{source_id}_clip_{idx}",
                        title=segment.hook_title,
                        privacy_status=privacy_status,
                    )
                    logger.info("Published Short #%d: https://youtube.com/shorts/%s", idx, short_id)
                finally:
                    if not keep_artifacts and rendered_clip.exists():
                        rendered_clip.unlink(missing_ok=True)

            # Record overall source video as processed in the ledger
            state_tracker.data["processed_source_ids"].append(source_id)
            state_tracker.save()
            logger.info("Finished processing video '%s'. State updated.", source_id)

        except Exception as err:
            logger.error("Failed processing video '%s': %s", video_file.name, err, exc_info=True)

        finally:
            # Clean up raw source video to save disk on ephemeral runner
            if not keep_artifacts and video_file.exists():
                video_file.unlink(missing_ok=True)

    logger.info("\n========================================")
    logger.info("PIPELINE EXECUTION COMPLETED")
    logger.info("========================================")


def main():
    parser = argparse.ArgumentParser(description="GTA 6 Auto-Clipper Autonomous Pipeline")
    parser.add_argument("--url", type=str, default=None, help="Specific YouTube video/stream URL to clip")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--ledger", type=str, default="processed_videos.json", help="Path to processed_videos.json")
    parser.add_argument(
        "--privacy",
        type=str,
        default=os.environ.get("PUBLISH_PRIVACY", "public"),
        choices=["public", "unlisted", "private"],
        help="YouTube Shorts privacy status",
    )
    parser.add_argument("--max-clips", type=int, default=2, help="Max clips per source video")
    parser.add_argument("--keep-artifacts", action="store_true", help="Retain local raw videos and clips")
    args = parser.parse_args()

    run_pipeline(
        custom_url=args.url,
        config_path=args.config,
        ledger_path=args.ledger,
        privacy_status=args.privacy,
        max_clips_per_video=args.max_clips,
        keep_artifacts=args.keep_artifacts,
    )


if __name__ == "__main__":
    main()
