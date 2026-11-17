"""Unit tests for Gemini 2.5 Flash viral segment detection module."""

import unittest
from unittest.mock import MagicMock, patch

from src.ai_analyzer import (
    SYSTEM_PROMPT,
    GeminiSegmentDetector,
    ViralSegment,
    extract_json_array_from_text,
    generate_fallback_segments,
)


class TestAiAnalyzer(unittest.TestCase):

    def test_system_prompt_matches_requirement(self):
        """Requirement 2: Ensure system prompt contains the exact required wording."""
        self.assertIn("Analyze this transcript for high-retention", SYSTEM_PROMPT)
        self.assertIn("US YouTube Shorts viewers", SYSTEM_PROMPT)
        self.assertIn("`start_time`, `end_time`, `hook_title`, and `viral_score`", SYSTEM_PROMPT)

    def test_extract_json_array_clean(self):
        """Requirement 3: Strict JSON array extraction."""
        raw = '[{"start_time": 12.0, "end_time": 45.0, "hook_title": "Police Chase", "viral_score": 9.5}]'
        parsed = extract_json_array_from_text(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["hook_title"], "Police Chase")

    def test_extract_json_array_with_markdown_and_preamble(self):
        """Vulnerability Prevention: Strip markdown code blocks, backticks, preamble & postscript."""
        raw = (
            "Certainly! Here are the viral clips detected:\n\n"
            "```json\n"
            "[\n"
            '  {"start_time": 10.5, "end_time": 48.0, "hook_title": "INSANE GLITCH 💀", "viral_score": 9.8},\n'
            '  {"start_time": 60.0, "end_time": 95.0, "hook_title": "Crazy Stunt", "viral_score": 8.7}\n'
            "]\n"
            "```\n"
            "Let me know if you need any adjustments!"
        )
        parsed = extract_json_array_from_text(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["hook_title"], "INSANE GLITCH 💀")
        self.assertEqual(parsed[1]["viral_score"], 8.7)

    def test_extract_json_array_malformed_raises_value_error(self):
        """Vulnerability Prevention: Malformed content without brackets raises ValueError."""
        with self.assertRaises(ValueError):
            extract_json_array_from_text("Sorry, I could not find any clips.")

        with self.assertRaises(ValueError):
            extract_json_array_from_text("")

    def test_generate_fallback_segments(self):
        """Vulnerability Prevention: Generate reliable default segment windows."""
        fallbacks = generate_fallback_segments(total_duration=180.0, segment_duration=45.0, max_segments=2)
        self.assertEqual(len(fallbacks), 2)
        self.assertIsInstance(fallbacks[0], ViralSegment)
        self.assertTrue(30.0 <= (fallbacks[0].end_time - fallbacks[0].start_time) <= 55.0)
        self.assertIn("GTA 6", fallbacks[0].hook_title)
        self.assertGreater(fallbacks[0].viral_score, 0)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key"})
    def test_detect_viral_segments_with_mocked_gemini(self):
        """End-to-end detection with mocked Gemini 2.5 Flash response."""
        detector = GeminiSegmentDetector()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "```json\n"
            "[\n"
            '  {"start_time": 15.0, "end_time": 55.0, "hook_title": "Best GTA 6 Heist", "viral_score": 9.9}\n'
            "]\n"
            "```"
        )
        mock_client.models.generate_content.return_value = mock_response
        detector._client = mock_client

        segments = detector.detect_viral_segments(
            transcript_text="[15.0s] Look at that vault explosion!",
            total_duration=120.0,
            video_title="GTA 6 Gameplay Episode 1",
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].hook_title, "Best GTA 6 Heist")
        self.assertEqual(segments[0].viral_score, 9.9)
        self.assertEqual(segments[0].start_time, 15.0)
        self.assertEqual(segments[0].end_time, 55.0)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test_dummy_key"})
    def test_detect_viral_segments_falls_back_on_llm_failure(self):
        """Vulnerability Prevention: Fallback triggers seamlessly if LLM call throws."""
        detector = GeminiSegmentDetector()

        mock_client = MagicMock()
        # Simulate network or quota exception
        mock_client.models.generate_content.side_effect = RuntimeError("Resource exhausted (429)")
        detector._client = mock_client

        # Must not crash, but return safe fallback segment
        segments = detector.detect_viral_segments(
            transcript_text="Some gameplay audio text",
            total_duration=100.0,
        )

        self.assertGreater(len(segments), 0)
        self.assertIn("GTA 6", segments[0].hook_title)


if __name__ == "__main__":
    unittest.main()
