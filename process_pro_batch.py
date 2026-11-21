"""Batch Pro Clipper Script (process_pro_batch.py).

Renders Clip 2 and Clip 3 directly from the target video using the 5-stage pro pipeline:
- Section download via yt-dlp
- Smart 1080x1920 blur_wide framing
- Kinetic karaoke subtitles & 0-3s hook banner
- Studio audio mixing (ducked BGM + SFX + loudnorm)
- Final Quality Gate verification across all 3 clips
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import audio_mixer
import captioner
import framer
import ingest
import run_pro_clipper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pro_batch")

TARGET_URL = "https://youtu.be/_nSjfe3k6Ss"
WORK_DIR = Path("./work").resolve()
OUT_DIR = Path("./pro_clips").resolve()
ASSETS_DIR = Path("./assets").resolve()
TRANSCRIPT_PATH = WORK_DIR / "transcript.json"

CLIPS_CONFIG = [
    {
        "index": 2,
        "title": "Insane Highway Stunt Jump Escape",
        "hook_banner": "INSANE HIGHWAY STUNT JUMP!",
        "time_range": "*00:00:55-00:01:28",
        "clip_start": 55.0,
        "clip_end": 88.0,
        "filename": "clip_02_insane_highway_stunt_jump.mp4",
        "sfx_cues": [
            {"timestamp_sec": 12.0, "sfx_name": "pop"},
            {"timestamp_sec": 22.0, "sfx_name": "boom"},
        ],
        "viral_score": 97,
    },
    {
        "index": 3,
        "title": "Surviving The Worst Crash in GTA 6",
        "hook_banner": "SURVIVING THE WORST CRASH?!",
        "time_range": "*00:01:40-00:02:15",
        "clip_start": 100.0,
        "clip_end": 135.0,
        "filename": "clip_03_surviving_the_worst_crash.mp4",
        "sfx_cues": [
            {"timestamp_sec": 14.0, "sfx_name": "boom"},
            {"timestamp_sec": 24.0, "sfx_name": "pop"},
        ],
        "viral_score": 94,
    },
]


def download_section(url: str, time_range: str, output_path: Path, ffmpeg_bin: str) -> Path:
    """Download a targeted section using yt-dlp."""
    logger.info("Downloading section %s from %s...", time_range, url)
    cmd = [
        "yt-dlp",
        "--ffmpeg-location", ffmpeg_bin,
        "--download-sections", time_range,
        "-f", "bv*[height<=720]+ba/b",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def main():
    ffmpeg_bin = ingest.get_ffmpeg_binary()
    framer_engine = framer.SmartFramer(ffmpeg_bin=ffmpeg_bin)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in CLIPS_CONFIG:
        idx = cfg["index"]
        out_final = OUT_DIR / cfg["filename"]
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING PRO SHORT #%d: %s", idx, cfg["hook_banner"])
        logger.info("=" * 60)

        raw_section = WORK_DIR / f"section_{idx}.mp4"
        framed_video = WORK_DIR / f"framed_{idx}.mp4"
        captioned_video = WORK_DIR / f"captioned_{idx}.mp4"

        try:
            # 1. Download targeted 16:9 60fps section
            if not raw_section.is_file() or raw_section.stat().st_size < 1024 * 1024:
                download_section(TARGET_URL, cfg["time_range"], raw_section, ffmpeg_bin)

            # 2. Smart Framing (blur_wide 1080x1920)
            logger.info(">>> Applying Smart Framing (blur_wide)...")
            framer_engine.render_blur_wide(
                input_path=raw_section,
                output_path=framed_video,
                crf=18,
                preset="medium",
            )

            # 3. Kinetic Subtitles & 0-3s Hook Banner
            logger.info(">>> Burning Kinetic Karaoke Subtitles & 0-3s Hook Banner...")
            captioner.burn_captions(
                video_path=framed_video,
                output_path=captioned_video,
                transcript_path=TRANSCRIPT_PATH,
                clip_start=cfg["clip_start"],
                clip_end=cfg["clip_end"],
                hook_banner=cfg["hook_banner"],
                ffmpeg_bin=ffmpeg_bin,
                crf=18,
                preset="medium",
            )

            # 4. Studio Audio Mixing
            logger.info(">>> Mixing BGM Ducking, Timed SFX, and EBU R128 Loudnorm...")
            audio_mixer.mix_clip_audio(
                video_path=captioned_video,
                output_path=out_final,
                sfx_cues=cfg["sfx_cues"],
                assets_dir=ASSETS_DIR,
                bgm_volume_db=-22.0,
                sfx_volume_db=-16.0,
                add_opening_whoosh=True,
                ffmpeg_bin=ffmpeg_bin,
            )

            logger.info("Successfully finished Short #%d: %s", idx, out_final.name)

        finally:
            # Clean up intermediate heavy work files to preserve disk space
            for temp_f in [raw_section, framed_video, captioned_video]:
                if temp_f.is_file():
                    try:
                        temp_f.unlink()
                    except Exception:
                        pass

    # --------------------------------------------------------------------------
    # QUALITY GATE VERIFICATION ACROSS ALL 3 CLIPS
    # --------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING QUALITY GATE ON ALL 3 PRO SHORTS")
    logger.info("=" * 80)

    all_clips = [
        {
            "file_path": str(OUT_DIR / "clip_01_gta6_graphics_in_india.mp4"),
            "hook_banner": "GTA 6 GRAPHICS IN INDIA?!",
            "viral_score": 96,
            "start_sec": 15.0,
            "end_sec": 45.0,
            "layout": "blur_wide",
            "word_count": 28,
        },
        {
            "file_path": str(OUT_DIR / "clip_02_insane_highway_stunt_jump.mp4"),
            "hook_banner": "INSANE HIGHWAY STUNT JUMP!",
            "viral_score": 97,
            "start_sec": 55.0,
            "end_sec": 88.0,
            "layout": "blur_wide",
            "word_count": 31,
        },
        {
            "file_path": str(OUT_DIR / "clip_03_surviving_the_worst_crash.mp4"),
            "hook_banner": "SURVIVING THE WORST CRASH?!",
            "viral_score": 94,
            "start_sec": 100.0,
            "end_sec": 135.0,
            "layout": "blur_wide",
            "word_count": 30,
        },
    ]

    passed = run_pro_clipper.run_quality_gate(all_clips, ffmpeg_bin=ffmpeg_bin)
    if not passed:
        raise RuntimeError("Quality Gate failed for one or more pro clips!")
    logger.info("ALL 3 PRO CLIPS VERIFIED AND READY IN %s", OUT_DIR)


if __name__ == "__main__":
    main()
