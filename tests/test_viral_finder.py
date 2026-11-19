"""Unit tests for viral moment detection module (viral_finder.py)."""

import os
import struct
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from viral_finder import (
    DEFAULT_SKILL_PATH,
    SfxEvent,
    ViralShortCandidate,
    detect_audio_energy_peaks,
    find_viral_moments,
    load_system_instruction,
    snap_to_word_boundaries,
)


class TestViralFinder(unittest.TestCase):

    def test_viral_editor_skill_documentation(self):
        """Requirement 1: Verify retention principles in skills/viral_editor_skill.md."""
        self.assertTrue(DEFAULT_SKILL_PATH.is_file(), "skills/viral_editor_skill.md must exist")
        content = load_system_instruction()

        self.assertIn("0-3s Hook Rule", content)
        self.assertIn("Story Arc Rule", content)
        self.assertIn("Hook Banner Rule", content)
        self.assertIn("Audio Cue Planning", content)
        self.assertIn("whoosh", content)
        self.assertIn("pop", content)
        self.assertIn("boom", content)

    def test_snap_to_word_boundaries(self):
        """Vulnerability Prevention: Snap start/end to word boundaries with 0.25s padding."""
        transcript = [
            {"start": 10.5, "end": 12.0, "text": "Look at that cop car"},
            {"start": 12.2, "end": 15.0, "text": "approaching from behind!"},
            {"start": 15.2, "end": 20.0, "text": "He just hit the barrier!"},
            {"start": 20.2, "end": 35.0, "text": "Insane crash explosion!"},
            {"start": 35.2, "end": 40.0, "text": "No way we survived that!"},
        ]

        # Proposed cut: 10.8s to 34.5s
        snapped_start, snapped_end = snap_to_word_boundaries(
            start_sec=10.8,
            end_sec=34.5,
            transcript_segments=transcript,
            padding_sec=0.25,
            min_duration=18.0,
            max_duration=55.0,
            total_duration=60.0,
        )

        # 10.8 should snap to 10.5 - 0.25 = 10.25
        self.assertEqual(snapped_start, 10.25)
        # 34.5 should snap to 35.0 + 0.25 = 35.25
        self.assertEqual(snapped_end, 35.25)
        self.assertTrue(18.0 <= (snapped_end - snapped_start) <= 55.0)

    def test_audio_energy_peaks_detection(self):
        """Requirement 2: Extract loudness spikes from audio WAV."""
        test_wav = Path("test_synthetic_loudness.wav")
        self.addCleanup(lambda: test_wav.unlink(missing_ok=True))

        framerate = 16000
        duration_sec = 4
        total_samples = framerate * duration_sec

        # Create audio: silence for 2 seconds, huge loudness burst at 2.5s - 3.5s
        with wave.open(str(test_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)

            samples = []
            for i in range(total_samples):
                t = i / framerate
                if 2.2 <= t <= 3.2:
                    # Loud shout/explosion wave
                    val = int(25000 * (1 if (i % 20) < 10 else -1))
                else:
                    # Low whisper/background
                    val = int(300 * (1 if (i % 50) < 25 else -1))
                samples.append(val)

            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        peaks = detect_audio_energy_peaks(test_wav, window_sec=0.5, top_n=5)
        self.assertGreater(len(peaks), 0)
        self.assertGreaterEqual(peaks[0]["intensity"], 0.8)
        self.assertIn(peaks[0]["reaction_type"], ["high_shout_or_chaos", "laughter_or_reaction"])

    def test_missing_api_key_raises_error_strictly(self):
        """Vulnerability Prevention: MUST fail cleanly if GEMINI_API_KEY is missing (NO DUMMY TEXT)."""
        with patch.dict(os.environ, {}, clear=True):
            dummy_transcript = [{"start": 0.0, "end": 20.0, "text": "Testing speech"}]
            with self.assertRaises(ValueError) as ctx:
                find_viral_moments(transcript_data=dummy_transcript, api_key=None)
            self.assertIn("GEMINI_API_KEY is not set or empty", str(ctx.exception))

    @patch("viral_finder.genai")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "valid_mock_gemini_key"})
    def test_find_viral_moments_structured_output(self, mock_genai):
        """Requirement 2: Return structured JSON with hook banner, SFX, and score."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """[
            {
                "start_sec": 14.5,
                "end_sec": 48.0,
                "hook_banner": "GTA 6 NPCs React Like Real People?!",
                "title": "Insane GTA 6 NPC Reaction #Shorts #GTA6",
                "viral_score": 96,
                "reason": "Immediate shout hook followed by an intense car pursuit and hilarious crash.",
                "sfx_events": [
                    {"time_sec": 0.2, "type": "whoosh"},
                    {"time_sec": 12.0, "type": "pop"},
                    {"time_sec": 30.5, "type": "boom"}
                ]
            }
        ]"""
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        transcript = [
            {"start": 14.2, "end": 20.0, "text": "Did you see that?!"},
            {"start": 20.5, "end": 48.2, "text": "He just flew over the building!"},
        ]

        candidates = find_viral_moments(
            transcript_data=transcript,
            total_duration=120.0,
            video_title="GTA 6 Fun Moments",
            max_clips=1,
            api_key="valid_mock_gemini_key",
        )

        self.assertEqual(len(candidates), 1)
        clip = candidates[0]
        self.assertIsInstance(clip, ViralShortCandidate)
        self.assertEqual(clip.viral_score, 96)
        self.assertEqual(clip.hook_banner, "GTA 6 NPCs React Like Real People?!")
        self.assertEqual(len(clip.sfx_events), 3)
        self.assertEqual(clip.sfx_events[0].type, "whoosh")
        self.assertEqual(clip.sfx_events[2].type, "boom")
        self.assertTrue(18.0 <= (clip.end_sec - clip.start_sec) <= 55.0)


if __name__ == "__main__":
    unittest.main()
