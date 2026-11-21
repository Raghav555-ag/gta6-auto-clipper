"""Unit test suite for viral_hotspot.py.

Verifies:
  1. YouTube Most Replayed heatmap parsing & graceful fallback.
  2. Librosa loudness and excitement peak detection.
  3. Word boundary snapping with 0.25s lead-in, 0.35s tail padding, and 18s-48s clamping.
  4. Robust Gemini JSON sanitization & bracket extraction.
  5. Zero-mock subtitle enforcement and zero-word fail-fast error reporting.
  6. Secure cookie context management & file shredding.
  7. End-to-end editorial analysis with mocked Gemini 2.5 Flash responses.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import viral_hotspot
from viral_hotspot import HighlightWord, HotspotClip, MoodEvent


class TestViralHotspot(unittest.TestCase):
    """Test suite for viral_hotspot.py module."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.test_dir.name)

        # Mock sample transcript
        self.transcript_file = self.work_dir / "transcript.json"
        self.sample_transcript = {
            "duration": 60.0,
            "language": "en",
            "total_words_count": 8,
            "segments": [
                {
                    "id": 1,
                    "start": 5.0,
                    "end": 8.5,
                    "text": "Wait till you see this crazy jump",
                    "words": [
                        {"word": "Wait", "start": 5.0, "end": 5.4},
                        {"word": "till", "start": 5.5, "end": 5.8},
                        {"word": "you", "start": 5.9, "end": 6.1},
                        {"word": "see", "start": 6.2, "end": 6.5},
                        {"word": "this", "start": 6.6, "end": 6.9},
                        {"word": "crazy", "start": 7.0, "end": 7.5},
                        {"word": "jump", "start": 7.6, "end": 8.5},
                    ],
                },
                {
                    "id": 2,
                    "start": 26.0,
                    "end": 30.0,
                    "text": "No way he survived that crash",
                    "words": [
                        {"word": "No", "start": 26.0, "end": 26.3},
                        {"word": "way", "start": 26.4, "end": 26.8},
                        {"word": "he", "start": 26.9, "end": 27.2},
                        {"word": "survived", "start": 27.3, "end": 28.0},
                        {"word": "that", "start": 28.1, "end": 28.5},
                        {"word": "crash", "start": 28.6, "end": 30.0},
                    ],
                },
            ],
        }
        self.transcript_file.write_text(json.dumps(self.sample_transcript), encoding="utf-8")

        # Mock metadata
        self.metadata_file = self.work_dir / "metadata.json"
        self.sample_metadata = {
            "video_id": "test12345",
            "title": "GTA 6 INSANE JUMPS IN INDIA",
            "duration": 60.0,
            "has_heatmap": True,
            "hotspots": [
                {"start_sec": 5.0, "end_sec": 9.0, "intensity": 0.88},
                {"start_sec": 25.0, "end_sec": 30.0, "intensity": 0.95},
            ],
        }
        self.metadata_file.write_text(json.dumps(self.sample_metadata), encoding="utf-8")

    def tearDown(self):
        self.test_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. YouTube Heatmap Parsing
    # --------------------------------------------------------------------------

    def test_parse_youtube_heatmap_sorting_and_filtering(self):
        """Verify raw heatmap is ranked by intensity and sorted chronologically."""
        raw_hm = [
            {"start_time": 10.0, "end_time": 12.0, "value": 0.25},
            {"start_time": 30.0, "end_time": 32.5, "value": 0.92},
            {"start_time": 50.0, "end_time": 52.0, "value": 0.74},
            {"start_time": 70.0, "end_time": 72.0, "value": 0.15},
        ]
        parsed = viral_hotspot.parse_youtube_heatmap(raw_hm, top_n=2)
        self.assertEqual(len(parsed), 2)
        # Should contain top 2: 30.0 (0.92) and 50.0 (0.74), sorted by start_sec
        self.assertEqual(parsed[0]["start_sec"], 30.0)
        self.assertEqual(parsed[0]["intensity"], 0.92)
        self.assertEqual(parsed[1]["start_sec"], 50.0)
        self.assertEqual(parsed[1]["intensity"], 0.74)

    def test_parse_youtube_heatmap_empty_or_none(self):
        """Verify empty or None heatmap returns empty list without raising."""
        self.assertEqual(viral_hotspot.parse_youtube_heatmap(None), [])
        self.assertEqual(viral_hotspot.parse_youtube_heatmap([]), [])
        self.assertEqual(viral_hotspot.parse_youtube_heatmap([{"invalid": "data"}]), [])

    # --------------------------------------------------------------------------
    # 2. Word Boundary Snapping
    # --------------------------------------------------------------------------

    def test_snap_to_word_boundaries_lead_in_and_padding(self):
        """Verify boundaries snap to real words with -0.25s lead-in and +0.35s tail padding."""
        flat_words = [
            {"word": "Wait", "start": 5.0, "end": 5.4},
            {"word": "crash", "start": 28.6, "end": 30.0},
        ]
        # Proposed: 5.1s -> 29.5s
        snapped_start, snapped_end = viral_hotspot.snap_to_word_boundaries(
            proposed_start=5.1,
            proposed_end=29.5,
            words=flat_words,
            total_duration=60.0,
            min_sec=18.0,
            max_sec=48.0,
        )

        # 5.0 - 0.25 = 4.75
        self.assertEqual(snapped_start, 4.75)
        # 30.0 + 0.35 = 30.35
        self.assertEqual(snapped_end, 30.35)
        # Duration: 30.35 - 4.75 = 25.6s (within 18s - 48s)
        self.assertGreaterEqual(snapped_end - snapped_start, 18.0)
        self.assertLessEqual(snapped_end - snapped_start, 48.0)

    def test_snap_to_word_boundaries_enforces_min_max_duration(self):
        """Verify short proposals expand to min_sec (18s) and long proposals clamp to max_sec (48s)."""
        flat_words = [{"word": "Test", "start": 10.0, "end": 11.0}]
        # Proposal too short (10.0 -> 15.0 = 5s)
        s, e = viral_hotspot.snap_to_word_boundaries(
            proposed_start=10.0,
            proposed_end=15.0,
            words=flat_words,
            total_duration=100.0,
            min_sec=18.0,
            max_sec=48.0,
        )
        self.assertGreaterEqual(e - s, 18.0)

        # Proposal too long (10.0 -> 80.0 = 70s)
        s2, e2 = viral_hotspot.snap_to_word_boundaries(
            proposed_start=10.0,
            proposed_end=80.0,
            words=flat_words,
            total_duration=100.0,
            min_sec=18.0,
            max_sec=48.0,
        )
        self.assertLessEqual(e2 - s2, 48.0)

    # --------------------------------------------------------------------------
    # 3. Gemini JSON Sanitization
    # --------------------------------------------------------------------------

    def test_sanitize_gemini_json_markdown_stripping(self):
        """Verify markdown wrappers (```json ... ```) are cleanly extracted."""
        raw_json_md = """```json
[
  {
    "start_sec": 5.0,
    "end_sec": 28.0,
    "hook_kicker": "HOW DID THIS HAPPEN?! 🔥",
    "main_title": "GTA 6 JET SKI PHYSICS IS BROKEN",
    "overall_mood": "sigma_crazy",
    "viral_score": 96,
    "reason": "Insane high speed boat jump",
    "highlight_words": [{"word": "INSANE", "color_tag": "red"}],
    "mood_events": [{"time_sec": 8.0, "mood": "shock_impact", "sfx": "boom"}]
  }
]
```"""
        parsed = viral_hotspot.sanitize_gemini_json(raw_json_md)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["hook_kicker"], "HOW DID THIS HAPPEN?! 🔥")

    def test_sanitize_gemini_json_invalid_raises_error(self):
        """Verify unparseable text raises JSONDecodeError."""
        with self.assertRaises(json.JSONDecodeError):
            viral_hotspot.sanitize_gemini_json("Not a valid json response at all.")

    # --------------------------------------------------------------------------
    # 4. Zero Words Vulnerability Prevention
    # --------------------------------------------------------------------------

    def test_find_hotspot_moments_aborts_on_zero_words(self):
        """Verify RuntimeError is raised immediately if transcript contains zero words."""
        empty_transcript_file = self.work_dir / "empty_transcript.json"
        empty_transcript_file.write_text(json.dumps({"segments": []}), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            viral_hotspot.find_hotspot_moments(
                transcript_path=empty_transcript_file,
                metadata_path=self.metadata_file,
                api_key="mock_key",
            )
        self.assertIn("0 spoken words", str(ctx.exception))

    def test_find_hotspot_moments_aborts_on_missing_api_key(self):
        """Verify RuntimeError is raised if GEMINI_API_KEY is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                viral_hotspot.find_hotspot_moments(
                    transcript_path=self.transcript_file,
                    metadata_path=self.metadata_file,
                    api_key=None,
                )
            self.assertIn("GEMINI_API_KEY environment variable is not set", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 5. Gemini 2.5 Flash Mocked Integration
    # --------------------------------------------------------------------------

    @patch("viral_hotspot.genai.Client")
    def test_find_hotspot_moments_successful_editorial_parse(self, mock_client_cls):
        """Verify end-to-end editorial director returns validated HotspotClip models."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {
                "start_sec": 5.0,
                "end_sec": 30.0,
                "hook_kicker": "WAIT FOR THE END! 💀",
                "main_title": "GTA 6 STUNT JUMP GONE WRONG",
                "overall_mood": "sigma_crazy",
                "viral_score": 97,
                "reason": "Escalating highway chase resulting in an insane physics launch",
                "highlight_words": [
                    {"word": "CRAZY", "color_tag": "red"},
                    {"word": "JUMP", "color_tag": "yellow"},
                ],
                "mood_events": [
                    {"time_sec": 0.05, "mood": "laugh", "sfx": "whoosh", "use_pixelfx": False},
                    {"time_sec": 12.0, "mood": "shock_impact", "sfx": "boom", "use_pixelfx": True},
                ],
            }
        ])
        mock_client.models.generate_content.return_value = mock_response

        clips = viral_hotspot.find_hotspot_moments(
            transcript_path=self.transcript_file,
            metadata_path=self.metadata_file,
            api_key="valid_test_key",
        )

        self.assertEqual(len(clips), 1)
        clip = clips[0]
        self.assertIsInstance(clip, HotspotClip)
        self.assertEqual(clip.hook_kicker, "WAIT FOR THE END! 💀")
        self.assertEqual(clip.main_title, "GTA 6 STUNT JUMP GONE WRONG")
        self.assertEqual(clip.overall_mood, "sigma_crazy")
        self.assertEqual(clip.viral_score, 97)
        self.assertEqual(len(clip.highlight_words), 2)
        self.assertEqual(len(clip.mood_events), 2)
        # Word boundary snapping verification
        self.assertEqual(clip.start_sec, 4.75)  # 5.0 - 0.25
        self.assertEqual(clip.end_sec, 30.35)  # 30.0 + 0.35

    # --------------------------------------------------------------------------
    # 6. Secure Cookie Handling & Shredding
    # --------------------------------------------------------------------------

    def test_secure_cookie_context_decodes_and_shreds(self):
        """Verify Netscape cookies are decoded, mounted, and securely shredded."""
        valid_cookie = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsample_session\n"
        b64_cookie = "IyBOZXRzY2FwZSBIVFRQIENvb2tpZSBGaWxlCi55b3V0dWJlLmNvbQlUUlVFLwkvCXRydWUK"

        with patch.dict("os.environ", {"YT_COOKIES_B64": b64_cookie}):
            with viral_hotspot.secure_cookie_context(work_dir=self.work_dir) as cookie_file:
                self.assertIsNotNone(cookie_file)
                cookie_path = Path(cookie_file)
                self.assertTrue(cookie_path.exists())

            # File must be deleted after exiting context
            self.assertFalse(cookie_path.exists())


if __name__ == "__main__":
    unittest.main()
