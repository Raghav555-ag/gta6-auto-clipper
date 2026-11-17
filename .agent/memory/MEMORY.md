# Project Memory & Context Store

Last Updated: 2026-11-17

## 1. Project Conventions
- **Language / Runtime**: Python 3.11+ on GitHub Actions (`ubuntu-latest`).
- **Core Libraries**: `yt-dlp` (ingestion), `faster-whisper` (local transcription), `google-genai` (Gemini 2.5 Flash), `ffmpeg` (9:16 rendering + ASS subtitles), `google-api-python-client` (YouTube Data API v3).
- **Execution Target**: Headless 24/7 cloud workflow on GitHub Actions (ephemeral Linux runner).
- **Format**: YouTube Shorts (1080x1920, 9:16, 30–55s duration, kinetic gaming subtitles).

## 2. Architecture & Decisions (ADRs)
- **ADR-001 (Cloud-First Automation)**: Use GitHub Actions as a free, 24/7 ephemeral Linux runner with scheduled cron jobs and manual workflow dispatch.
- **ADR-002 (Bot & Age-Gate Ingestion)**: Ingest YouTube videos using Base64-decoded cookies (`YOUTUBE_COOKIES_BASE64`) passed via GitHub Secrets.
- **ADR-003 (Local Speech-to-Text)**: Use `faster-whisper` with `int8` quantization running directly on the runner CPU for zero-cost word-level timestamps.
- **ADR-004 (AI Hook Detection)**: Use `gemini-2.5-flash` to evaluate transcripts specifically for high-retention US gaming hooks and viral titles/tags.
- **ADR-005 (State Persistence)**: Store processed video IDs and upload metadata in `processed_videos.json`, automatically committed back to the repository with `[skip ci]`.
- **ADR-006 (Headless Publishing)**: Use OAuth2 Refresh Token credentials for unattended YouTube Data API uploads.

## 3. User Preferences
- **Architecture**: Autonomous cloud workflow; local machine does not need to stay on.
- **Content**: GTA 6 creator gameplay and stream clips optimized for viral retention.

## 4. Current State & Active Milestones
- **Active Plan**: Implementation plan created for the Autonomous Cloud Workflow Architecture on GitHub Actions.
- **Awaiting**: User review and confirmation on target creator channels, video layout format, and initial publishing privacy.
