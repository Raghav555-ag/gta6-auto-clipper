"""Unit tests for run_pro_clipper.py module.

Verifies end-to-end 5-stage orchestration, Quality Gate verification
(1080x1920, non-zero file size, audio presence, transcript coverage,
varied clip duration), and slugification.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import run_pro_clipper
from viral_finder import ViralShortCandidate, SfxEvent


class TestRunProClipper(unittest.TestCase):
    """Test suite for run_pro_clipper.py."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.test_dir.name)
        self.sample_clip = self.work_dir / "sample_short.mp4"
        self.sample_clip.write_bytes(b"dummy video data" * 10000)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_slugify(self):
        """Verify title slugification."""
        self.assertEqual(run_pro_clipper.slugify("GTA 6 NPCs React Like Real People?!"), "gta_6_npcs_react_like_real_people")
        self.assertEqual(run_pro_clipper.slugify("HE DID NOT EXPECT THIS ENDING 💀"), "he_did_not_expect_this_ending")

    # --------------------------------------------------------------------------
    # Quality Gate Verification Tests
    # --------------------------------------------------------------------------

    @patch("run_pro_clipper.probe_clip")
    def test_quality_gate_passes_healthy_clips(self, mock_probe):
        """Verify quality gate passes when all criteria are satisfied."""
        mock_probe.return_value = {
            "width": 1080,
            "height": 1920,
            "resolution": "1080x1920",
            "duration": 32.5,
            "has_audio": True,
            "size_bytes": 1024 * 500,
            "size_mb": 0.5,
        }

        clips = [
            {
                "file_path": str(self.sample_clip),
                "hook_banner": "GTA 6 INSANE MOMENT",
                "viral_score": 95,
                "start_sec": 10.0,
                "end_sec": 42.5,
                "layout": "blur_wide",
                "word_count": 25,
            }
        ]

        passed = run_pro_clipper.run_quality_gate(clips, ffmpeg_bin="ffmpeg.exe")
        self.assertTrue(passed)

    @patch("run_pro_clipper.probe_clip")
    def test_quality_gate_fails_bad_resolution(self, mock_probe):
        """Verify quality gate fails if resolution is not 1080x1920."""
        mock_probe.return_value = {
            "width": 1920,
            "height": 1080,  # 16:9 instead of 9:16
            "resolution": "1920x1080",
            "duration": 30.0,
            "has_audio": True,
            "size_bytes": 1024 * 500,
            "size_mb": 0.5,
        }

        clips = [
            {
                "file_path": str(self.sample_clip),
                "hook_banner": "TEST BANNER",
                "viral_score": 88,
                "start_sec": 5.0,
                "end_sec": 35.0,
                "layout": "blur_wide",
                "word_count": 20,
            }
        ]

        passed = run_pro_clipper.run_quality_gate(clips, ffmpeg_bin="ffmpeg.exe")
        self.assertFalse(passed)

    @patch("run_pro_clipper.probe_clip")
    def test_quality_gate_fails_zero_words(self, mock_probe):
        """Verify quality gate fails if transcript has 0 words in clip window."""
        mock_probe.return_value = {
            "width": 1080,
            "height": 1920,
            "resolution": "1080x1920",
            "duration": 30.0,
            "has_audio": True,
            "size_bytes": 1024 * 500,
            "size_mb": 0.5,
        }

        clips = [
            {
                "file_path": str(self.sample_clip),
                "hook_banner": "SILENT VIDEO",
                "viral_score": 75,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "layout": "blur_wide",
                "word_count": 0,  # Zero words!
            }
        ]

        passed = run_pro_clipper.run_quality_gate(clips, ffmpeg_bin="ffmpeg.exe")
        self.assertFalse(passed)

    @patch("run_pro_clipper.probe_clip")
    def test_quality_gate_fails_missing_audio(self, mock_probe):
        """Verify quality gate fails if audio track is missing."""
        mock_probe.return_value = {
            "width": 1080,
            "height": 1920,
            "resolution": "1080x1920",
            "duration": 30.0,
            "has_audio": False,  # Missing audio!
            "size_bytes": 1024 * 500,
            "size_mb": 0.5,
        }

        clips = [
            {
                "file_path": str(self.sample_clip),
                "hook_banner": "NO AUDIO CLIP",
                "viral_score": 80,
                "start_sec": 10.0,
                "end_sec": 40.0,
                "layout": "blur_wide",
                "word_count": 15,
            }
        ]

        passed = run_pro_clipper.run_quality_gate(clips, ffmpeg_bin="ffmpeg.exe")
        self.assertFalse(passed)

    # --------------------------------------------------------------------------
    # 5-Stage Orchestration Mock Test
    # --------------------------------------------------------------------------

    @patch("ingest.download_high_res_video")
    @patch("ingest.extract_audio_16k_mono")
    @patch("ingest.transcribe_real_audio")
    @patch("viral_finder.find_viral_moments")
    @patch("subprocess.run")
    @patch("framer.SmartFramer.process_clip")
    @patch("captioner.burn_captions")
    @patch("audio_mixer.mix_clip_audio")
    @patch("run_pro_clipper.run_quality_gate")
    def test_run_pro_clipper_pipeline(
        self,
        mock_qgate,
        mock_mix,
        mock_caption,
        mock_frame,
        mock_subproc,
        mock_viral,
        mock_transcribe,
        mock_extract,
        mock_dl,
    ):
        """Verify sequential 5-stage pipeline execution."""
        mock_dl.return_value = {
            "resolution": "3840x2160",
            "fps": 60.0,
            "duration": 120.0,
            "title": "GTA 6 Test Gameplay",
        }
        mock_transcribe.return_value = {
            "total_words_count": 50,
            "segments": [
                {
                    "start": 10.0,
                    "end": 35.0,
                    "words": [{"word": "GTA", "start": 10.0, "end": 10.5}],
                }
            ],
        }

        # Mock viral candidate
        mock_cand = ViralShortCandidate(
            start_sec=10.0,
            end_sec=35.0,
            hook_banner="GTA 6 REAL LIFE GRAPHICS",
            title="Crazy GTA 6 Graphics #Shorts #GTA6",
            viral_score=96,
            reason="High retention combat reveal with police payoff",
            sfx_events=[SfxEvent(time_sec=1.5, type="pop")],
        )
        mock_viral.return_value = [mock_cand]
        mock_qgate.return_value = True

        out_dir = self.work_dir / "pro_clips"
        results = run_pro_clipper.run_pro_clipper(
            url="https://youtu.be/testvideo",
            work_dir=self.work_dir / "work",
            output_dir=out_dir,
            layout="blur_wide",
            max_clips=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["viral_score"], 96)
        self.assertEqual(results[0]["hook_banner"], "GTA 6 REAL LIFE GRAPHICS")

        # Verify each stage was called
        mock_dl.assert_called_once()
        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_viral.assert_called_once()
        mock_frame.assert_called_once()
        mock_caption.assert_called_once()
        mock_mix.assert_called_once()
        mock_qgate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
