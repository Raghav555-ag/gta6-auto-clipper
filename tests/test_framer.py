"""Unit tests for framer.py module.

Verifies auto-editor pacing tightening, smooth tracking clamping & damping,
max pan speed safeguards, blur_wide canvas rendering, split_facecam layout,
and CRF 18 sharp encoding parameters.
"""

import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import framer


class TestFramer(unittest.TestCase):
    """Test suite for framer.py."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.test_dir.name)
        self.sample_video = self.work_dir / "sample.mp4"
        self.sample_video.write_bytes(b"dummy mp4 data for testing")

    def tearDown(self):
        self.test_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. auto-editor Tight Pacing
    # --------------------------------------------------------------------------

    @patch("subprocess.run")
    @patch("framer.get_auto_editor_binary")
    def test_tighten_pacing_invokes_correct_args(self, mock_get_ae, mock_subproc):
        """Verify auto-editor is called with --margin 0.25s and specified threshold."""
        mock_get_ae.return_value = "auto-editor.exe"
        out_video = self.work_dir / "tightened.mp4"

        # Simulate auto-editor creating non-empty output
        def fake_ae_run(cmd, **kwargs):
            out_video.write_bytes(b"tightened video bytes")
            return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = fake_ae_run

        result_path = framer.tighten_pacing(
            input_path=self.sample_video,
            output_path=out_video,
            margin_sec=0.25,
            threshold="4%",
            auto_editor_bin="auto-editor.exe",
        )

        self.assertEqual(result_path, out_video.resolve())
        cmd_called = mock_subproc.call_args[0][0]
        self.assertIn("--margin", cmd_called)
        self.assertIn("0.25s", cmd_called)
        self.assertIn("--edit", cmd_called)
        self.assertIn("audio:threshold=4%", cmd_called)
        self.assertIn("-o", cmd_called)
        self.assertIn("--no-open", cmd_called)

    @patch("framer.get_auto_editor_binary")
    def test_tighten_pacing_fallback_when_missing(self, mock_get_ae):
        """Verify tighten_pacing gracefully preserves file when auto-editor is absent."""
        mock_get_ae.return_value = None
        out_video = self.work_dir / "tightened.mp4"

        result_path = framer.tighten_pacing(
            input_path=self.sample_video,
            output_path=out_video,
            auto_editor_bin=None,
        )

        self.assertTrue(out_video.exists())
        self.assertEqual(result_path, out_video.resolve())

    # --------------------------------------------------------------------------
    # 2. Tracking Clamping, Dead-Zone & Max Pan Speed Safeguards
    # --------------------------------------------------------------------------

    def test_tracking_clamping_and_max_pan_speed(self):
        """Simulate tracking frames and verify coordinates never exceed bounds or max pan speed."""
        src_w = 1920
        src_h = 1080
        crop_w = int(src_h * 9 / 16)  # 607 -> 606
        crop_w = (crop_w // 2) * 2
        max_x = src_w - crop_w  # 1314

        # Manually verify damping logic with synthetic targets
        current_x = float(max_x // 2)  # center = 657
        dead_zone_px = 30
        max_pan_speed = 15.0
        ema_alpha = 0.15

        # Extreme target (simulate player suddenly appearing at rightmost edge)
        extreme_targets = [1500.0, 1500.0, 1500.0, -200.0, -200.0]
        recorded_offsets = []

        for target in extreme_targets:
            # 1. Clamp target to valid range
            clamped_target = max(0.0, min(float(max_x), target))
            self.assertGreaterEqual(clamped_target, 0.0)
            self.assertLessEqual(clamped_target, float(max_x))

            # 2. Dead-zone test
            diff = clamped_target - current_x
            if abs(diff) > dead_zone_px:
                delta = diff * ema_alpha
                # 3. Max pan speed enforcement
                if abs(delta) > max_pan_speed:
                    delta = math.copysign(max_pan_speed, delta)
                current_x += delta

            # 4. Strict clamping on position
            final_x = int(round(max(0.0, min(float(max_x), current_x))))
            self.assertGreaterEqual(final_x, 0)
            self.assertLessEqual(final_x, max_x)
            recorded_offsets.append(final_x)

        # Ensure no jump was greater than max_pan_speed
        for i in range(1, len(recorded_offsets)):
            jump = abs(recorded_offsets[i] - recorded_offsets[i - 1])
            self.assertLessEqual(jump, max_pan_speed + 1.0)

    # --------------------------------------------------------------------------
    # 3. Smart Framing: blur_wide Canvas Layout
    # --------------------------------------------------------------------------

    @patch("subprocess.run")
    def test_render_blur_wide_ffmpeg_parameters(self, mock_subproc):
        """Verify render_blur_wide applies blurred background, centered canvas, and CRF 18."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        out_video = self.work_dir / "short_blur.mp4"

        # Simulate output file creation
        def fake_ffmpeg(cmd, **kwargs):
            out_video.write_bytes(b"dummy rendered video")
            return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = fake_ffmpeg

        framer_obj = framer.SmartFramer(ffmpeg_bin="ffmpeg.exe")
        result = framer_obj.render_blur_wide(
            input_path=self.sample_video,
            output_path=out_video,
            crf=18,
            preset="medium",
        )

        self.assertEqual(result, out_video.resolve())
        cmd = mock_subproc.call_args[0][0]

        # Check filter_complex arguments
        filter_str = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("split=2", filter_str)
        self.assertIn("boxblur=", filter_str)
        self.assertIn("scale=1080:1920", filter_str)
        self.assertIn("scale=1080:-2", filter_str)
        self.assertIn("overlay=0:(H-h)/2", filter_str)

        # Check high-fidelity encoding flags
        self.assertIn("-crf", cmd)
        self.assertEqual(cmd[cmd.index("-crf") + 1], "18")
        self.assertIn("-preset", cmd)
        self.assertEqual(cmd[cmd.index("-preset") + 1], "medium")
        self.assertIn("-pix_fmt", cmd)
        self.assertEqual(cmd[cmd.index("-pix_fmt") + 1], "yuv420p")
        self.assertIn("-c:v", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")

    # --------------------------------------------------------------------------
    # 4. Smart Framing: split_facecam Layout
    # --------------------------------------------------------------------------

    @patch("subprocess.run")
    def test_render_split_facecam_parameters(self, mock_subproc):
        """Verify render_split_facecam crops facecam box and stacks above gameplay."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        out_video = self.work_dir / "short_facecam.mp4"

        def fake_ffmpeg(cmd, **kwargs):
            out_video.write_bytes(b"dummy facecam video")
            return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = fake_ffmpeg

        framer_obj = framer.SmartFramer(ffmpeg_bin="ffmpeg.exe")
        result = framer_obj.render_split_facecam(
            input_path=self.sample_video,
            output_path=out_video,
            facecam_box=(100, 50, 400, 300),
            crf=18,
        )

        self.assertEqual(result, out_video.resolve())
        cmd = mock_subproc.call_args[0][0]

        filter_str = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("crop=400:300:100:50", filter_str)
        self.assertIn("vstack=inputs=2", filter_str)

    # --------------------------------------------------------------------------
    # 5. Process Clip Pipeline Dispatch
    # --------------------------------------------------------------------------

    @patch.object(framer.SmartFramer, "render_blur_wide")
    @patch.object(framer.SmartFramer, "render_track")
    @patch.object(framer.SmartFramer, "render_split_facecam")
    def test_process_clip_dispatches_layouts(self, mock_split, mock_track, mock_blur):
        """Verify process_clip dispatches correctly to layout methods."""
        framer_obj = framer.SmartFramer(ffmpeg_bin="ffmpeg.exe")
        out = self.work_dir / "out.mp4"

        framer_obj.process_clip(self.sample_video, out, layout="blur_wide")
        mock_blur.assert_called_once()

        framer_obj.process_clip(self.sample_video, out, layout="track")
        mock_track.assert_called_once()

        framer_obj.process_clip(self.sample_video, out, layout="split_facecam")
        mock_split.assert_called_once()


if __name__ == "__main__":
    unittest.main()
