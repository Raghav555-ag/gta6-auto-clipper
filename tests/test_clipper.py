"""Unit tests for FFmpeg video clipping, 9:16 vertical crop, and kinetic subtitles."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.clipper import (
    VideoClipper,
    format_ass_timestamp,
    generate_kinetic_ass,
    get_cross_platform_font,
)


class TestVideoClipper(unittest.TestCase):

    def setUp(self):
        self.clipper = VideoClipper(output_dir="test_output")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        out_dir = Path("test_output")
        if out_dir.exists():
            for f in out_dir.glob("*"):
                f.unlink(missing_ok=True)
            out_dir.rmdir()

    def test_format_ass_timestamp(self):
        """Requirement 3: Verify H:MM:SS.cs timestamp formatting."""
        self.assertEqual(format_ass_timestamp(0.0), "0:00:00.00")
        self.assertEqual(format_ass_timestamp(12.345), "0:00:12.35")
        self.assertEqual(format_ass_timestamp(65.5), "0:01:05.50")
        self.assertEqual(format_ass_timestamp(3661.05), "1:01:01.05")

    def test_get_cross_platform_font(self):
        """Vulnerability Prevention: Cross-platform font fallback availability."""
        fonts = get_cross_platform_font()
        self.assertIn("DejaVu Sans", fonts)
        self.assertIn("Arial", fonts)

    def test_generate_kinetic_ass(self):
        """Requirement 3: Test kinetic .ass subtitle script generation."""
        words = [
            {"start": 10.0, "end": 10.5, "text": "Insane"},
            {"start": 10.6, "end": 11.2, "text": "GTA"},
            {"start": 11.3, "end": 12.0, "text": "stunt!"},
        ]
        target_ass = Path("test_output/test_sub.ass")

        ass_file = generate_kinetic_ass(
            words=words,
            segment_start=10.0,
            segment_end=20.0,
            output_ass_path=target_ass,
            words_per_chunk=3,
        )

        self.assertTrue(ass_file.exists())
        content = ass_file.read_text(encoding="utf-8")

        # Verify 9:16 vertical resolution
        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)

        # Verify active word yellow highlight and text
        self.assertIn("&H0000FFFF&", content)
        self.assertIn("INSANE", content)
        self.assertIn("GTA", content)
        self.assertIn("STUNT!", content)

        # Verify relative start time (10.0s in video -> 0.0s in clip)
        self.assertIn("0:00:00.00", content)

    def test_render_short_missing_input_raises_filenotfound(self):
        """Vulnerability Prevention: Validate input video existence."""
        with self.assertRaises(FileNotFoundError):
            self.clipper.render_short("non_existent_file.mp4", start_time=10.0, end_time=30.0)

    def test_render_short_invalid_boundaries_raises_value_error(self):
        """Validate start/end boundaries."""
        fake_video = Path("test_output/fake_in.mp4")
        fake_video.write_bytes(b"dummy")
        with self.assertRaises(ValueError):
            self.clipper.render_short(str(fake_video), start_time=30.0, end_time=10.0)

    @patch("subprocess.run")
    def test_render_short_ffmpeg_command_and_crop_filter(self, mock_subprocess_run):
        """Requirement 2: Verify 9:16 center-crop filter and parameters."""
        fake_video = Path("test_output/sample_gameplay.mp4")
        fake_video.write_bytes(b"dummy_mp4_bytes")

        fake_out = Path("test_output/sample_short.mp4")

        def fake_run(cmd, **kwargs):
            fake_out.write_bytes(b"dummy_rendered_bytes")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        mock_subprocess_run.side_effect = fake_run

        res_path = self.clipper.render_short(
            video_path=str(fake_video),
            start_time=15.0,
            end_time=45.0,
            output_name="sample_short",
        )

        self.assertTrue(res_path.exists())
        cmd = mock_subprocess_run.call_args[0][0]

        # Verify FFmpeg flags
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-ss", cmd)
        self.assertIn("15.0", cmd)
        self.assertIn("-t", cmd)
        self.assertIn("30.0", cmd)

        # Verify center crop 9:16 and scale to 1080:1920
        vf_idx = cmd.index("-vf")
        filter_str = cmd[vf_idx + 1]
        self.assertIn("crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0", filter_str)
        self.assertIn("scale=1080:1920", filter_str)

    @patch("subprocess.run")
    def test_render_short_raises_called_process_error_on_failure(self, mock_subprocess_run):
        """Vulnerability Prevention: Report CalledProcessError on non-zero exit."""
        fake_video = Path("test_output/corrupt.mp4")
        fake_video.write_bytes(b"corrupt_data")

        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="Error while filtering: Invalid argument",
        )

        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self.clipper.render_short(str(fake_video), start_time=5.0, end_time=25.0)
        self.assertEqual(ctx.exception.returncode, 1)

    @patch("subprocess.run")
    def test_render_short_timeout_handling(self, mock_subprocess_run):
        """Vulnerability Prevention: Guard against indefinite hangs with TimeoutError."""
        fake_video = Path("test_output/hang.mp4")
        fake_video.write_bytes(b"dummy")

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=360)

        with self.assertRaises(TimeoutError):
            self.clipper.render_short(str(fake_video), start_time=0.0, end_time=30.0, timeout_seconds=360)


if __name__ == "__main__":
    unittest.main()
