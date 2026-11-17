"""YouTube Data API v3 Publishing and State-Tracking Module.

Handles headless OAuth2 authentication via refresh tokens, duplicate prevention
using an atomic JSON ledger, resumable video uploading, and exponential backoff
retry logic for HTTP 429/5xx rate limits.
"""

import datetime
import json
import logging
import os
import random
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
except ImportError:
    Credentials = None
    build = None
    HttpError = Exception
    MediaFileUpload = None

logger = logging.getLogger("uploader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"


class StateTracker:
    """Manages the local processed_videos.json ledger with atomic saves to prevent duplicates."""

    def __init__(self, ledger_path: str = "processed_videos.json"):
        self.ledger_path = Path(ledger_path).resolve()
        self.data: Dict[str, Any] = {"processed_source_ids": [], "uploaded_shorts": []}
        self.load()

    def load(self) -> None:
        """Load state ledger from disk or initialize empty ledger."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    if not isinstance(self.data.get("processed_source_ids"), list):
                        self.data["processed_source_ids"] = []
                    if not isinstance(self.data.get("uploaded_shorts"), list):
                        self.data["uploaded_shorts"] = []
            except Exception as exc:
                logger.warning("Corrupted or unreadable ledger at %s, initializing empty: %s", self.ledger_path, exc)
                self.data = {"processed_source_ids": [], "uploaded_shorts": []}
        else:
            self.save()

    def is_processed(self, source_video_id: str) -> bool:
        """Vulnerability Prevention: Check if source video ID has already been clipped/uploaded."""
        clean_id = str(source_video_id).strip()
        return clean_id in self.data["processed_source_ids"]

    def record_upload(
        self,
        source_video_id: str,
        short_id: str,
        title: str,
        privacy_status: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a successful Short upload in the ledger and persist atomically."""
        clean_id = str(source_video_id).strip()
        if clean_id not in self.data["processed_source_ids"]:
            self.data["processed_source_ids"].append(clean_id)

        entry = {
            "source_video_id": clean_id,
            "youtube_short_id": short_id,
            "short_url": f"https://youtube.com/shorts/{short_id}",
            "title": title,
            "privacy_status": privacy_status,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self.data["uploaded_shorts"].append(entry)
        self.save()
        logger.info("Recorded upload for source '%s' -> Short ID: %s", clean_id, short_id)

    def save(self) -> None:
        """Atomically persist state to prevent half-written corruption during unexpected exits."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = self.ledger_path.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            json.dump(self.data, tf, indent=2)
            temp_name = tf.name

        # Atomic file replacement
        os.replace(temp_name, self.ledger_path)


def format_shorts_metadata(
    title: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Tuple[str, str, List[str]]:
    """Format and optimize title, description, and tags for US gaming YouTube Shorts."""
    # Ensure #GTA6 and #Shorts hashtags are present in title
    formatted_title = title.strip()
    if "#Shorts" not in formatted_title:
        formatted_title += " #Shorts"
    if "#GTA6" not in formatted_title:
        formatted_title += " #GTA6"

    # YouTube titles are capped strictly at 100 characters
    if len(formatted_title) > 100:
        base_title = title.strip()[:80].rstrip()
        formatted_title = f"{base_title}... #Shorts #GTA6"[:100]

    # Build description
    default_desc = (
        f"{title}\n\n"
        "Insane moments from GTA 6! Drop a like and subscribe for daily GTA 6 clips.\n\n"
        "#GTA6 #Shorts #Gaming #GTAOnline #GrandTheftAuto6"
    )
    formatted_desc = (description.strip() + "\n\n#GTA6 #Shorts") if description else default_desc

    # Standardized gaming tags
    base_tags = ["GTA6", "GTA 6", "Grand Theft Auto 6", "Shorts", "Gaming", "GTA Online"]
    if tags:
        for t in tags:
            clean_tag = t.strip("# ").strip()
            if clean_tag and clean_tag not in base_tags:
                base_tags.append(clean_tag)

    return formatted_title, formatted_desc, base_tags[:15]


class YouTubePublisher:
    """Publishes videos to YouTube using OAuth2 credentials and resumable uploads."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        ledger_path: str = "processed_videos.json",
    ):
        self.client_id = client_id or os.environ.get("YT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("YT_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.environ.get("YT_REFRESH_TOKEN")
        self.state_tracker = StateTracker(ledger_path=ledger_path)
        self._service = None

    def _get_service(self):
        """Build authorized YouTube service client via OAuth2 refresh token."""
        if self._service is None:
            if not (self.client_id and self.client_secret and self.refresh_token):
                missing = []
                if not self.client_id:
                    missing.append("YT_CLIENT_ID")
                if not self.client_secret:
                    missing.append("YT_CLIENT_SECRET")
                if not self.refresh_token:
                    missing.append("YT_REFRESH_TOKEN")
                raise ValueError(
                    f"Missing required YouTube OAuth credentials: {', '.join(missing)}. "
                    "Ensure these secrets are set in GitHub Actions or environment variables."
                )

            if Credentials is None or build is None:
                raise RuntimeError(
                    "Google API packages are not installed. Install via: "
                    "pip install google-api-python-client google-auth google-auth-oauthlib"
                )

            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=[YOUTUBE_UPLOAD_SCOPE],
            )

            logger.info("Initializing YouTube Data API client with OAuth refresh token...")
            self._service = build("youtube", "v3", credentials=creds)

        return self._service

    def upload_short(
        self,
        video_path: str,
        source_video_id: str,
        title: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        privacy_status: str = "public",
        max_retries: int = 5,
    ) -> str:
        """Upload a rendered 9:16 vertical video as a YouTube Short with exponential backoff.

        Args:
            video_path: Local path to the rendered 9:16 vertical MP4.
            source_video_id: Unique ID of original source video (for duplicate prevention).
            title: Title for the Short.
            description: Optional custom description.
            tags: Optional tags list.
            privacy_status: 'public', 'unlisted', or 'private'.
            max_retries: Max retry attempts on 429/5xx rate limits.

        Returns:
            str: Uploaded YouTube video ID.

        Raises:
            FileExistsError: If source_video_id was already uploaded.
            FileNotFoundError: If video_path does not exist.
            RuntimeError: If upload fails after all retries.
        """
        # Vulnerability Prevention: Prevent duplicate uploads
        if self.state_tracker.is_processed(source_video_id):
            raise FileExistsError(
                f"Duplicate prevention: Source video '{source_video_id}' has already been processed and uploaded."
            )

        video_file = Path(video_path).resolve()
        if not video_file.exists():
            raise FileNotFoundError(f"Video file to upload not found: {video_file}")
        if video_file.stat().st_size == 0:
            raise ValueError(f"Video file is empty: {video_file}")

        final_title, final_desc, final_tags = format_shorts_metadata(title, description, tags)
        youtube = self._get_service()

        body = {
            "snippet": {
                "title": final_title,
                "description": final_desc,
                "tags": final_tags,
                "categoryId": "20",  # Gaming
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en",
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

        # 5MB chunk size for resumable streaming
        chunk_size = 5 * 1024 * 1024
        media = MediaFileUpload(
            str(video_file),
            mimetype="video/mp4",
            chunksize=chunk_size,
            resumable=True,
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        logger.info("Initiating resumable upload for '%s' to YouTube Shorts...", final_title)

        response = None
        attempt = 0

        # Vulnerability Prevention: Exponential backoff retry loop for 429/5xx and network drops
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    logger.info("Upload progress: %d%%", int(status.progress() * 100))

            except HttpError as err:
                status_code = err.resp.status if hasattr(err, "resp") else None
                if status_code in RETRIABLE_STATUS_CODES and attempt < max_retries:
                    attempt += 1
                    sleep_time = min(60.0, (2 ** attempt) + random.uniform(0.5, 2.0))
                    logger.warning(
                        "YouTube API rate limit or server error (HTTP %s). Retrying in %.2fs (attempt %d/%d)...",
                        status_code,
                        sleep_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Non-retriable YouTube API error (HTTP %s): %s", status_code, err)
                    raise

            except (socket.error, OSError) as net_err:
                if attempt < max_retries:
                    attempt += 1
                    sleep_time = min(60.0, (2 ** attempt) + random.uniform(0.5, 2.0))
                    logger.warning(
                        "Network connection error: %s. Retrying in %.2fs (attempt %d/%d)...",
                        net_err,
                        sleep_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Exceeded max retries after network error: %s", net_err)
                    raise

        uploaded_id = response.get("id")
        if not uploaded_id:
            raise RuntimeError(f"YouTube API upload completed without returning video ID: {response}")

        logger.info("YouTube Short published successfully! Video ID: %s", uploaded_id)
        logger.info("Watch Short at: https://youtube.com/shorts/%s", uploaded_id)

        # Record in local ledger
        self.state_tracker.record_upload(
            source_video_id=source_video_id,
            short_id=uploaded_id,
            title=final_title,
            privacy_status=privacy_status,
            metadata={"video_size_mb": round(video_file.stat().st_size / (1024 * 1024), 2)},
        )

        return uploaded_id
