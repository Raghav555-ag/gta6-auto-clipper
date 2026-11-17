# GTA 6 Auto-Clipper

An automated pipeline built for Google Antigravity to clip famous creators' GTA 6 gameplay/streams, transform highlights into vertical (9:16) YouTube Shorts/TikToks, and automatically upload them to your channel.

---

## Architecture & Features

- **Automated Download**: Fetches high-resolution videos/streams using `yt-dlp`.
- **Highlight Detection**: Analyzes gameplay segments and audio excitement for key moments.
- **Vertical 9:16 Reframing**: Crops and centers horizontal video into 1080x1920 mobile Shorts using `ffmpeg`.
- **YouTube Shorts Uploader**: Auto-uploads generated clips with tags and descriptions via the YouTube Data API.

---

## Project Structure

```text
gta6-auto-clipper/
├── AGENTS.md             # Guidelines for Antigravity AI agents
├── main.py               # Main CLI orchestrator
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables & secrets template
├── .gitignore            # Git exclusion rules
└── src/
    ├── downloader.py     # Video stream/VOD downloader
    ├── detector.py       # Highlight & audio spike detection
    ├── clipper.py        # 9:16 vertical video conversion & cutting
    └── uploader.py       # YouTube Data API uploader
```

---

## Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg** (if not already installed):
   ```bash
   winget install Gyan.FFmpeg
   ```

3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in any required credentials:
   ```bash
   cp .env.example .env
   ```

---

## Usage

- **Clip directly from a YouTube video URL**:
  ```bash
  python main.py --url "https://www.youtube.com/watch?v=EXAMPLE_ID" --duration 45
  ```

- **Process a local video file**:
  ```bash
  python main.py --video-file "downloads/sample.mp4" --duration 30
  ```

- **Auto-upload generated clips**:
  ```bash
  python main.py --url "https://www.youtube.com/watch?v=EXAMPLE_ID" --upload
  ```
