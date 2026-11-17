# GTA 6 Auto-Clipper Agent Guidelines

## Overview
This repository automates clipping GTA 6 gameplay and streams from target creators, processing highlights into vertical (9:16) YouTube Shorts/TikTok/Reels format, adding subtitles, and uploading them to YouTube.

## Tech Stack & Architecture
- **Language**: Python 3.12+
- **Video Downloader**: `yt-dlp`
- **Video Processing & Editing**: `ffmpeg`, `moviepy`, `opencv-python`
- **Audio & Transcription**: Whisper / Gemini Multimodal API for speech-to-text and highlight detection
- **Uploader**: YouTube Data API v3 (`google-api-python-client`)

## Project Layout
- `main.py`: CLI entry point for monitoring channels, clipping, and uploading
- `src/downloader.py`: Stream and VOD downloading using `yt-dlp`
- `src/detector.py`: Audio energy, laughter, chat spike, and AI scene detection for highlight extraction
- `src/clipper.py`: Video slicing, vertical 9:16 dynamic reframing, and caption overlay
- `src/uploader.py`: YouTube Shorts uploader via OAuth2
- `config/`: Channel sources and clipping configurations

## Rules for Agents
- Never commit raw media files (`.mp4`, `.mkv`, etc.) or API credentials (`.env`, `token.json`).
- Ensure all video operations handle temporary file cleanup.
- Keep dependencies lean and provide clear CLI options in `main.py`.
