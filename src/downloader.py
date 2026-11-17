"""Secure Cookie Decoding and Ingestion Module using yt-dlp.

Handles Base64 cookie decoding, Netscape format verification,
creator channel/URL downloading, proxy rotation, and credential cleanup.
"""

import base64
import binascii
import contextlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("downloader")


def decode_and_validate_cookies(raw_b64: Optional[str]) -> str:
    """Validate and decode a Base64-encoded Netscape cookie string.

    Args:
        raw_b64: Base64 string from environment variable.

    Returns:
        Decoded UTF-8 string containing Netscape cookies.

    Raises:
        ValueError: If environment variable is missing, empty, invalid Base64,
                    or lacks valid Netscape YouTube domain entries.
    """
    if not raw_b64 or not raw_b64.strip():
        raise ValueError(
            "Environment variable 'YT_COOKIES_B64' is missing or empty. "
            "Please provide a base64-encoded Netscape cookie export."
        )

    # Clean whitespace and quotes
    cleaned_b64 = raw_b64.strip().strip('"\'')

    # Gracefully fix missing Base64 padding
    padding_needed = len(cleaned_b64) % 4
    if padding_needed:
        cleaned_b64 += "=" * (4 - padding_needed)

    try:
        decoded_bytes = base64.b64decode(cleaned_b64, validate=False)
        decoded_text = decoded_bytes.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to decode 'YT_COOKIES_B64' (invalid Base64 or UTF-8): {exc}") from exc

    if not decoded_text.strip():
        raise ValueError("Decoded cookies file is empty.")

    # Netscape Cookie Format & Bot-Detection Validation:
    # Must contain Netscape header or valid tab-delimited domain entries for YouTube/Google
    lines = [line.strip() for line in decoded_text.splitlines() if line.strip()]
    has_header = any("# Netscape HTTP Cookie File" in line for line in lines)

    valid_domain_entries = [
        line for line in lines
        if (
            not line.startswith("#")
            and len(line.split("\t")) >= 6
            and any(dom in line.split("\t")[0] for dom in ["youtube.com", "google.com"])
        )
        or (
            line.startswith("#HttpOnly_")
            and any(dom in line for dom in ["youtube.com", "google.com"])
        )
    ]

    if not (has_header or valid_domain_entries):
        raise ValueError(
            "Bot-detection safeguard triggered: Decoded cookies do not appear to contain "
            "valid Netscape domain lines for YouTube (.youtube.com / google.com). "
            "Ensure cookies are exported in valid Netscape format."
        )

    return decoded_text


@contextlib.contextmanager
def secure_cookie_file(cookie_filename: str = "cookies.txt"):
    """Context manager to securely write decoded cookies to disk and guarantee deletion.

    Yields:
        Path: Absolute path to the created cookies.txt.
    """
    raw_b64 = os.environ.get("YT_COOKIES_B64")
    cookies_content = decode_and_validate_cookies(raw_b64)
    cookie_path = Path(cookie_filename).resolve()

    try:
        # Write securely with UTF-8 encoding
        with open(cookie_path, "w", encoding="utf-8") as f:
            f.write(cookies_content)

        # Restrict permissions on POSIX (read/write by owner only)
        try:
            os.chmod(cookie_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

        logger.info("Decoded Netscape cookies securely written to %s", cookie_path.name)
        yield cookie_path

    finally:
        # Overwrite with zeros before deletion to prevent credential leakage on disk/logs
        if cookie_path.exists():
            try:
                file_size = cookie_path.stat().st_size
                with open(cookie_path, "wb") as f:
                    f.write(b"\x00" * file_size)
                cookie_path.unlink()
                logger.info("Securely shredded and removed temporary cookie file: %s", cookie_path.name)
            except Exception as cleanup_err:
                # Fallback to direct unlink if overwriting fails
                cookie_path.unlink(missing_ok=True)
                logger.warning("Unlinked cookie file with fallback: %s", cleanup_err)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load downloader configuration from JSON file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class VideoDownloader:
    """Downloader coordinating yt-dlp execution, cookie injection, and proxy rotation."""

    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.download_dir = Path(self.config.get("download_dir", "downloads")).resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.proxies: List[str] = self.config.get("proxies", [])

    def _build_ydl_opts(self, cookie_path: Path, proxy: Optional[str] = None) -> Dict[str, Any]:
        """Construct robust yt-dlp options for best quality and reliability."""
        opts: Dict[str, Any] = {
            "cookiefile": str(cookie_path),
            # Best 1080p MP4 video + best AAC/M4A audio, merged into MP4 container
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(self.download_dir / "%(uploader)s_%(id)s_%(title).50s.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            # YouTube bot-detection evasion headers
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        }

        if proxy:
            opts["proxy"] = proxy

        return opts

    def _download_with_fallback(self, url: str, cookie_path: Path) -> Path:
        """Download a single video URL with proxy rotation fallback."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

        # Candidate proxies: configured proxies list + direct connection (None)
        candidate_proxies: List[Optional[str]] = list(self.proxies) + [None]
        last_error = None

        for proxy in candidate_proxies:
            proxy_label = proxy if proxy else "Direct Connection"
            logger.info("Attempting download of %s using %s...", url, proxy_label)
            ydl_opts = self._build_ydl_opts(cookie_path, proxy=proxy)

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                    
                    # Ensure filename reflects merged mp4
                    mp4_path = Path(filename).with_suffix(".mp4")
                    if mp4_path.exists():
                        logger.info("Successfully downloaded: %s", mp4_path.name)
                        return mp4_path
                    if Path(filename).exists():
                        return Path(filename)
                    
                    raise FileNotFoundError(f"Expected download file not found for {url}")

            except Exception as exc:
                logger.warning("Download failed using %s: %s", proxy_label, exc)
                last_error = exc
                continue

        raise RuntimeError(f"All download attempts failed for {url}. Last error: {last_error}")

    def download_latest(self) -> List[Path]:
        """Download latest videos for all creator channels/URLs defined in config.json.

        Returns:
            List[Path]: List of downloaded video file paths.
        """
        sources = self.config.get("sources", [])
        if not sources:
            logger.warning("No sources specified in config.json")
            return []

        max_videos = self.config.get("max_videos_per_source", 1)
        downloaded_files: List[Path] = []

        with secure_cookie_file() as cookie_path:
            for source in sources:
                logger.info("Checking source: %s", source)
                urls_to_download: List[str] = []

                # If channel or playlist, extract latest video URLs without full download
                if any(x in source for x in ["/@", "/channel/", "/c/", "/playlist", "/videos"]):
                    flat_opts = {
                        "cookiefile": str(cookie_path),
                        "extract_flat": True,
                        "playlist_items": f"1-{max_videos}",
                    }
                    if yt_dlp:
                        with yt_dlp.YoutubeDL(flat_opts) as ydl:
                            res = ydl.extract_info(source, download=False)
                            entries = res.get("entries", []) if res else []
                            for entry in entries[:max_videos]:
                                video_url = entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}"
                                urls_to_download.append(video_url)
                else:
                    urls_to_download.append(source)

                # Execute downloads with proxy fallbacks
                for url in urls_to_download:
                    try:
                        filepath = self._download_with_fallback(url, cookie_path)
                        downloaded_files.append(filepath)
                    except Exception as err:
                        logger.error("Failed to process %s: %s", url, err)

        return downloaded_files


if __name__ == "__main__":
    import sys

    try:
        downloader = VideoDownloader(config_path="config.json")
        results = downloader.download_latest()
        print(f"Downloaded {len(results)} videos successfully.")
    except Exception as e:
        logger.error("Downloader execution failed: %s", e)
        sys.exit(1)
