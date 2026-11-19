"""Unit tests for local-first clipping prototype (local_clipper.py)."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from local_clipper import (
    DEFAULT_TARGET_URL,
    create_clip_srt,
    discover_cookies,
    format_srt_timestamp,
    generate_fallback_segments,
    slugify,
)


class TestLocalClipper(unittest.TestCase):

    def test_default_target_url(self):
        """Requirement 1: Verify default URL matches requested target."""
        self.assertEqual(DEFAULT_TARGET_URL, "https://youtu.be/_nSjfe3k6Ss")

    def test_slugify(self):
        """Verify safe filename slugification."""
        self.assertEqual(slugify("CRAZY GTA 6 STUNT! 🚨 (Wait)"), "crazy_gta_6_stunt_wait")
        self.assertEqual(slugify(""), "clip")

    def test_format_srt_timestamp(self):
        """Verify SRT timestamp format HH:MM:SS,mmm."""
        self.assertEqual(format_srt_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_srt_timestamp(12.345), "00:00:12,345")
        self.assertEqual(format_srt_timestamp(65.5), "00:01:05,500")
        self.assertEqual(format_srt_timestamp(3661.05), "01:01:01,050")

    def test_create_clip_srt_relative_timestamps(self):
        """Requirement 5: Verify SRT generation relative to clip start time."""
        segments = [
            {"start": 10.0, "end": 15.0, "text": "Welcome to GTA 6!"},
            {"start": 15.5, "end": 20.0, "text": "Look at that car!"},
            {"start": 35.0, "end": 40.0, "text": "Out of bounds text."},
        ]
        out_srt = Path("test_clip.srt")
        self.addCleanup(lambda: out_srt.unlink(missing_ok=True))

        has_cues = create_clip_srt(
            transcript_segments=segments,
            clip_start=10.0,
            clip_end=25.0,
            output_srt=out_srt,
        )

        self.assertTrue(has_cues)
        self.assertTrue(out_srt.exists())

        content = out_srt.read_text(encoding="utf-8")
        # 10.0s should become 00:00:00,000
        self.assertIn("00:00:00,000 --> 00:00:05,000", content)
        self.assertIn("Welcome to GTA 6!", content)
        # 35s segment should not appear
        self.assertNotIn("Out of bounds text", content)

    def test_discover_cookies_absent_does_not_crash(self):
        """Requirement 2: Ensure missing cookies does not crash and returns None."""
        with patch.dict("os.environ", {}, clear=True):
            cookie = discover_cookies(explicit_path="non_existent_cookies.txt")
            self.assertIsNone(cookie)

    def test_fallback_segments_generator(self):
        """Vulnerability Prevention: Fallback segments within duration and min/max limits."""
        fallbacks = generate_fallback_segments(
            total_duration=180.0,
            max_clips=3,
            min_sec=20,
            max_sec=55,
        )
        self.assertEqual(len(fallbacks), 3)
        for fb in fallbacks:
            self.assertGreaterEqual(fb["start_sec"], 0.0)
            self.assertLessEqual(fb["end_sec"], 180.0)
            dur = fb["end_sec"] - fb["start_sec"]
            self.assertTrue(20 <= dur <= 55)
            self.assertIn("hook_title", fb)
            self.assertIn("reason", fb)


if __name__ == "__main__":
    unittest.main()
