"""Unit tests for YouTube Data API v3 publishing and state-tracking module."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.uploader import (
    StateTracker,
    YouTubePublisher,
    format_shorts_metadata,
)


class TestYouTubeUploader(unittest.TestCase):

    def setUp(self):
        self.test_ledger = Path("test_processed_videos.json")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self.test_ledger.exists():
            self.test_ledger.unlink(missing_ok=True)

    def test_state_tracker_lifecycle_and_duplicate_prevention(self):
        """Requirement 2 & Vulnerability Prevention: Prevent duplicate uploads with JSON ledger."""
        tracker = StateTracker(str(self.test_ledger))
        self.assertFalse(tracker.is_processed("source_vid_123"))

        # Record upload
        tracker.record_upload(
            source_video_id="source_vid_123",
            short_id="short_abc",
            title="Insane GTA 6 Stunt #Shorts #GTA6",
        )

        # Must now report as processed
        self.assertTrue(tracker.is_processed("source_vid_123"))

        # Reload from disk to verify atomic persistence
        tracker_reloaded = StateTracker(str(self.test_ledger))
        self.assertTrue(tracker_reloaded.is_processed("source_vid_123"))
        self.assertEqual(len(tracker_reloaded.data["uploaded_shorts"]), 1)
        self.assertEqual(tracker_reloaded.data["uploaded_shorts"][0]["youtube_short_id"], "short_abc")

    def test_state_tracker_handles_corrupt_file_gracefully(self):
        """Vulnerability Prevention: Corrupted JSON ledger handled without crashing."""
        self.test_ledger.write_text("CORRUPTED_NON_JSON_DATA{{{", encoding="utf-8")
        tracker = StateTracker(str(self.test_ledger))
        self.assertFalse(tracker.is_processed("any_id"))
        self.assertEqual(tracker.data["processed_source_ids"], [])

    def test_format_shorts_metadata(self):
        """Requirement 3: Ensure #GTA6 and #Shorts hashtags and 100 char limit."""
        title = "Incredible Secret GTA 6 Location Found by Creator"
        fmt_title, fmt_desc, tags = format_shorts_metadata(title)

        self.assertIn("#Shorts", fmt_title)
        self.assertIn("#GTA6", fmt_title)
        self.assertLessEqual(len(fmt_title), 100)
        self.assertIn("#GTA6 #Shorts", fmt_desc)
        self.assertIn("GTA6", tags)
        self.assertIn("Shorts", tags)

        # Test over-length title truncation
        huge_title = "A" * 120
        trunc_title, _, _ = format_shorts_metadata(huge_title)
        self.assertLessEqual(len(trunc_title), 100)
        self.assertIn("#Shorts", trunc_title)

    def test_publisher_missing_credentials_raises_value_error(self):
        """Requirement 1: Missing OAuth secrets must raise a clear ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            publisher = YouTubePublisher(ledger_path=str(self.test_ledger))
            with self.assertRaises(ValueError) as ctx:
                publisher._get_service()
            self.assertIn("Missing required YouTube OAuth credentials", str(ctx.exception))
            self.assertIn("YT_CLIENT_ID", str(ctx.exception))

    def test_duplicate_source_rejection_before_upload(self):
        """Vulnerability Prevention: Reject upload if source video ID is already marked processed."""
        tracker = StateTracker(str(self.test_ledger))
        tracker.record_upload("duplicate_source_id", "prev_short", "Test")

        publisher = YouTubePublisher(
            client_id="dummy",
            client_secret="dummy",
            refresh_token="dummy",
            ledger_path=str(self.test_ledger),
        )

        dummy_video = Path("dummy_video.mp4")
        dummy_video.touch()
        self.addCleanup(lambda: dummy_video.unlink(missing_ok=True))

        with self.assertRaises(FileExistsError) as ctx:
            publisher.upload_short(
                video_path=str(dummy_video),
                source_video_id="duplicate_source_id",
                title="Duplicate Clip",
            )
        self.assertIn("Duplicate prevention", str(ctx.exception))

    @patch("src.uploader.MediaFileUpload")
    def test_upload_short_resumable_success(self, mock_media):
        """Requirement 3: Test successful resumable video upload."""
        publisher = YouTubePublisher(
            client_id="dummy_client",
            client_secret="dummy_secret",
            refresh_token="dummy_refresh",
            ledger_path=str(self.test_ledger),
        )

        dummy_video = Path("sample_short.mp4")
        dummy_video.write_bytes(b"dummy_short_data")
        self.addCleanup(lambda: dummy_video.unlink(missing_ok=True))

        # Mock YouTube API service
        mock_service = MagicMock()
        mock_request = MagicMock()
        # Simulate next_chunk() returning (status=None, response={"id": "uploaded_12345"})
        mock_request.next_chunk.return_value = (None, {"id": "uploaded_12345"})
        mock_service.videos().insert.return_value = mock_request
        publisher._service = mock_service

        short_id = publisher.upload_short(
            video_path=str(dummy_video),
            source_video_id="new_raw_video_001",
            title="Crazy GTA 6 Police Escape",
        )

        self.assertEqual(short_id, "uploaded_12345")
        self.assertTrue(publisher.state_tracker.is_processed("new_raw_video_001"))

    @patch("src.uploader.MediaFileUpload")
    @patch("time.sleep", return_value=None)
    def test_exponential_backoff_on_rate_limit(self, mock_sleep, mock_media):
        """Vulnerability Prevention: Exponential backoff on HTTP 429/5xx errors."""
        publisher = YouTubePublisher(
            client_id="dummy_client",
            client_secret="dummy_secret",
            refresh_token="dummy_refresh",
            ledger_path=str(self.test_ledger),
        )

        dummy_video = Path("backoff_test.mp4")
        dummy_video.write_bytes(b"dummy_bytes")
        self.addCleanup(lambda: dummy_video.unlink(missing_ok=True))

        # Create mock HttpError with 429
        class MockHttpResp:
            status = 429

        mock_http_error = Exception("Rate limit exceeded")
        mock_http_error.resp = MockHttpResp()

        mock_service = MagicMock()
        mock_request = MagicMock()
        # Fail once with 429, then succeed
        mock_request.next_chunk.side_effect = [
            mock_http_error,
            (None, {"id": "recovered_id"}),
        ]
        mock_service.videos().insert.return_value = mock_request
        publisher._service = mock_service

        short_id = publisher.upload_short(
            video_path=str(dummy_video),
            source_video_id="rate_limited_vid",
            title="Rate Limit Test Clip",
            max_retries=2,
        )

        self.assertEqual(short_id, "recovered_id")
        self.assertTrue(mock_sleep.called)
        self.assertTrue(publisher.state_tracker.is_processed("rate_limited_vid"))


if __name__ == "__main__":
    unittest.main()
