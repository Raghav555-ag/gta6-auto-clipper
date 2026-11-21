"""Unit tests for captioner.py module.

Verifies ASS timestamp formatting, filter path escaping, kinetic karaoke styling,
0-3s hook banner injection, and FFmpeg subtitle burning commands.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import captioner


class TestCaptioner(unittest.TestCase):
    """Test suite for captioner.py."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.test_dir.name)
        self.sample_video = self.work_dir / "sample.mp4"
        self.sample_video.write_bytes(b"dummy video")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_format_ass_time(self):
        """Verify seconds to ASS timestamp formatting."""
        self.assertEqual(captioner.format_ass_time(0.0), "0:00:00.00")
        self.assertEqual(captioner.format_ass_time(1.25), "0:00:01.25")
        self.assertEqual(captioner.format_ass_time(75.50), "0:01:15.50")
        self.assertEqual(captioner.format_ass_time(3661.05), "1:01:01.05")

    def test_escape_ffmpeg_filter_path(self):
        """Verify Windows drive colon and backslash escaping."""
        escaped = captioner.escape_ffmpeg_filter_path("C:/videos/test.ass")
        self.assertIn("C\\:/videos/test.ass", escaped)

    def test_generate_ass_script_with_banner_and_karaoke(self):
        """Verify .ass file contains HookBanner and Default karaoke cues."""
        words = [
            {"start": 10.0, "end": 10.5, "word": "Look"},
            {"start": 10.6, "end": 11.2, "word": "at"},
            {"start": 11.3, "end": 11.9, "word": "this"},
            {"start": 12.0, "end": 12.8, "word": "police"},
            {"start": 12.9, "end": 13.5, "word": "chase"},
        ]
        out_ass = self.work_dir / "sub.ass"

        ass_path, count = captioner.generate_ass_script(
            words=words,
            clip_start=10.0,
            clip_end=15.0,
            output_ass_path=out_ass,
            hook_banner="CRAZY POLICE CHASE IN GTA 6",
            words_per_chunk=3,
        )

        self.assertTrue(ass_path.is_file())
        content = ass_path.read_text(encoding="utf-8")

        # Check styles
        self.assertIn("Style: Default,", content)
        self.assertIn("Style: HookBanner,", content)

        # Check Hook Banner cue
        self.assertIn("HookBanner,,0,0,0,,{\\fad(150,200)}CRAZY POLICE CHASE IN GTA 6", content)

        # Check Karaoke chunks
        self.assertIn("{\\k", content)
        self.assertIn("LOOK", content)
        self.assertIn("AT", content)
        self.assertIn("THIS", content)
        self.assertGreaterEqual(count, 2)

    @patch("subprocess.run")
    def test_burn_captions_invokes_ffmpeg(self, mock_subproc):
        """Verify burn_captions builds and executes FFmpeg command."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        out_video = self.work_dir / "captioned.mp4"

        # Create dummy transcript.json
        t_json = self.work_dir / "transcript.json"
        t_json.write_text(json.dumps({
            "segments": [
                {
                    "start": 5.0,
                    "end": 8.0,
                    "text": "Hello world",
                    "words": [
                        {"start": 5.0, "end": 6.0, "word": "Hello"},
                        {"start": 6.5, "end": 7.5, "word": "world"},
                    ]
                }
            ]
        }), encoding="utf-8")

        def fake_ffmpeg(cmd, **kwargs):
            out_video.write_bytes(b"dummy captioned video")
            return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = fake_ffmpeg

        res = captioner.burn_captions(
            video_path=self.sample_video,
            output_path=out_video,
            transcript_path=t_json,
            clip_start=5.0,
            clip_end=8.0,
            hook_banner="HELLO WORLD TEST",
            ffmpeg_bin="ffmpeg.exe",
        )

        self.assertEqual(res, out_video.resolve())
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("-vf", cmd)
        vf_str = cmd[cmd.index("-vf") + 1]
        self.assertIn("ass=", vf_str)
        self.assertIn("-crf", cmd)
        self.assertIn("18", cmd)


if __name__ == "__main__":
    unittest.main()
