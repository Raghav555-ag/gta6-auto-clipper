"""Unit tests for audio_mixer.py module.

Verifies synthetic SFX/BGM generation, FFmpeg filtergraph construction
(sidechaincompress, adelay, amix normalize=0, loudnorm), cue sanitization,
and zero-clipping safeguards.
"""

import json
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import audio_mixer


class TestAudioMixer(unittest.TestCase):
    """Test suite for audio_mixer.py."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.test_dir.name)
        self.sample_video = self.work_dir / "sample.mp4"
        self.sample_video.write_bytes(b"dummy video data")

    def tearDown(self):
        self.test_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. Asset Initialization & Synthetic Audio Generation
    # --------------------------------------------------------------------------

    def test_ensure_assets_creates_clean_synthetic_audio(self):
        """Verify fallback SFX and BGM are synthesized as valid 16-bit PCM WAV files."""
        assets_dir = self.work_dir / "assets"
        info = audio_mixer.ensure_assets(assets_dir)

        sfx_dir = Path(info["sfx_dir"])
        bgm_dir = Path(info["bgm_dir"])

        # Check SFX files exist
        for sfx_name in ["whoosh.wav", "pop.wav", "boom.wav"]:
            f = sfx_dir / sfx_name
            self.assertTrue(f.is_file())
            self.assertGreater(f.stat().st_size, 100)
            # Verify valid WAV header
            with wave.open(str(f), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 44100)
                self.assertGreater(wf.getnframes(), 0)

        # Check BGM loop exists
        bgm_loop = bgm_dir / "ambient_loop.wav"
        self.assertTrue(bgm_loop.is_file())
        with wave.open(str(bgm_loop), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertGreater(wf.getnframes(), 44100 * 5)

        # Check README instructions
        self.assertTrue((sfx_dir / "README.md").is_file())
        self.assertTrue((bgm_dir / "README.md").is_file())

    # --------------------------------------------------------------------------
    # 2. SFX Cue Sanitization & Opening Whoosh
    # --------------------------------------------------------------------------

    def test_sanitize_sfx_cues(self):
        """Verify automatic opening whoosh and duration clamping."""
        raw_cues = [
            {"time_sec": -2.0, "type": "pop"},       # Invalid negative
            {"time_sec": 12.5, "type": "pop"},       # Valid
            {"time_sec": 25.0, "type": "boom"},      # Valid
            {"time_sec": 95.0, "type": "whoosh"},    # Out of bounds (>30s)
            {"time_sec": 14.0, "type": "unknown"},   # Unsupported type -> pop
        ]

        sanitized = audio_mixer.sanitize_sfx_cues(raw_cues, clip_duration=30.0, add_opening_whoosh=True)

        # Must include opening whoosh at 0.05s
        self.assertEqual(sanitized[0].type, "whoosh")
        self.assertEqual(sanitized[0].time_sec, 0.05)

        # Out of bounds and negative should be removed
        times = [c.time_sec for c in sanitized]
        self.assertNotIn(-2.0, times)
        self.assertNotIn(95.0, times)

        # Unknown type should fallback to pop
        types = [c.type for c in sanitized]
        self.assertIn("boom", types)
        self.assertNotIn("unknown", types)

    # --------------------------------------------------------------------------
    # 3. FFmpeg Filtergraph Construction (Sidechain Ducking + Loudnorm)
    # --------------------------------------------------------------------------

    def test_build_audio_filtergraph_with_bgm_and_sfx(self):
        """Verify sidechaincompress, adelay, amix normalize=0, and loudnorm in filtergraph."""
        dummy_whoosh = self.work_dir / "whoosh.wav"
        dummy_pop = self.work_dir / "pop.wav"
        sfx_list = [(dummy_whoosh, 0.05), (dummy_pop, 10.5)]

        filter_str = audio_mixer.build_audio_filtergraph(
            has_bgm=True,
            sfx_files=sfx_list,
            bgm_volume_db=-22.0,
            sfx_volume_db=-16.0,
        )

        # Check sidechain ducking chain
        self.assertIn("asplit=2[voice_main][voice_sidechain]", filter_str)
        self.assertIn("sidechaincompress=", filter_str)
        self.assertIn("[bgm_ducked]", filter_str)

        # Check SFX adelay (0.05s -> 50ms, 10.5s -> 10500ms)
        self.assertIn("adelay=50|50", filter_str)
        self.assertIn("adelay=10500|10500", filter_str)

        # Check anti-clipping amix normalize=0
        self.assertIn("amix=inputs=4:duration=first:dropout_transition=2:normalize=0[mixed]", filter_str)

        # Check EBU loudnorm and limiter
        self.assertIn("alimiter=limit=0.95", filter_str)
        self.assertIn("loudnorm=I=-14:TP=-1.5:LRA=11[aout]", filter_str)

    def test_build_audio_filtergraph_without_bgm(self):
        """Verify filtergraph constructs cleanly when BGM is disabled or absent."""
        dummy_pop = self.work_dir / "pop.wav"
        sfx_list = [(dummy_pop, 5.0)]

        filter_str = audio_mixer.build_audio_filtergraph(
            has_bgm=False,
            sfx_files=sfx_list,
        )

        self.assertNotIn("sidechaincompress", filter_str)
        self.assertIn("[voice_main]", filter_str)
        self.assertIn("amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mixed]", filter_str)
        self.assertIn("loudnorm=I=-14:TP=-1.5:LRA=11[aout]", filter_str)

    # --------------------------------------------------------------------------
    # 4. Master Mixer Execution & Remuxing
    # --------------------------------------------------------------------------

    @patch("subprocess.run")
    def test_mix_clip_audio_invokes_ffmpeg(self, mock_subproc):
        """Verify mix_clip_audio constructs command with -c:v copy, -c:a aac, and loudness flags."""
        out_mixed = self.work_dir / "mixed.mp4"

        # Mock probe duration and render
        def fake_run(cmd, **kwargs):
            if "-hide_banner" in cmd:
                # Probe call
                return MagicMock(returncode=0, stderr="Duration: 00:00:25.00, bitrate: 2000 kb/s")
            else:
                # Mixing call: create output file
                out_mixed.write_bytes(b"dummy mixed video")
                return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = fake_run

        assets_dir = self.work_dir / "assets"
        res = audio_mixer.mix_clip_audio(
            video_path=self.sample_video,
            output_path=out_mixed,
            sfx_cues=[{"time_sec": 5.0, "type": "pop"}],
            assets_dir=assets_dir,
            ffmpeg_bin="ffmpeg.exe",
        )

        self.assertEqual(res, out_mixed.resolve())
        # Check mixing CLI call
        mix_call_args = mock_subproc.call_args[0][0]
        self.assertIn("-c:v", mix_call_args)
        self.assertEqual(mix_call_args[mix_call_args.index("-c:v") + 1], "copy")
        self.assertIn("-c:a", mix_call_args)
        self.assertEqual(mix_call_args[mix_call_args.index("-c:a") + 1], "aac")
        self.assertIn("-b:a", mix_call_args)
        self.assertEqual(mix_call_args[mix_call_args.index("-b:a") + 1], "192k")
        self.assertIn("-shortest", mix_call_args)

    @patch("subprocess.run")
    def test_mix_clip_audio_fallback_on_error(self, mock_subproc):
        """Verify that an FFmpeg error gracefully preserves input video without crashing."""
        out_mixed = self.work_dir / "mixed_err.mp4"

        def fake_run(cmd, **kwargs):
            if "-hide_banner" in cmd:
                return MagicMock(returncode=0, stderr="Duration: 00:00:10.00")
            else:
                # Simulate failure
                return MagicMock(returncode=1, stderr="Filter graph initialization failed")

        mock_subproc.side_effect = fake_run

        assets_dir = self.work_dir / "assets"
        res = audio_mixer.mix_clip_audio(
            video_path=self.sample_video,
            output_path=out_mixed,
            assets_dir=assets_dir,
            ffmpeg_bin="ffmpeg.exe",
        )

        # Should fall back to copying original video to destination
        self.assertTrue(out_mixed.exists())
        self.assertEqual(out_mixed.read_bytes(), b"dummy video data")


if __name__ == "__main__":
    unittest.main()
