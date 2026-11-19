"""Unit tests for ingest.py module.

Verifies high-resolution downloading, FFmpeg 16kHz audio extraction,
multilingual Whisper model enforcement, real word-level transcription
schema (pycaps compatibility), zero-word abort guardrails, and cookie lifecycle.
"""

import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import ingest


class TestIngest(unittest.TestCase):
    """Test suite for ingest.py."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.test_dir.name)
        self.sample_wav = self.work_dir / "audio_16k.wav"
        # Create a small valid WAV file for testing
        self.sample_wav.write_bytes(b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 2000)

    def tearDown(self):
        self.test_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. Multilingual Model Enforcement & Zero Fake Subtitles Guardrail
    # --------------------------------------------------------------------------

    def test_english_only_model_raises_value_error(self):
        """Verify that passing '.en' models raises ValueError requiring multilingual models."""
        out_json = self.work_dir / "transcript.json"
        for en_model in ["small.en", "base.en", "tiny.en", "medium.en"]:
            with self.assertRaises(ValueError) as ctx:
                ingest.transcribe_real_audio(
                    audio_path=self.sample_wav,
                    output_json_path=out_json,
                    model_size=en_model,
                )
            self.assertIn("English-only model", str(ctx.exception))
            self.assertIn("multilingual model", str(ctx.exception))

    def test_whisper_missing_raises_runtime_error_without_mock(self):
        """Verify that missing faster-whisper raises RuntimeError and NEVER falls back to fake subtitles."""
        out_json = self.work_dir / "transcript.json"
        with patch.object(ingest, "WhisperModel", None):
            with self.assertRaises(RuntimeError) as ctx:
                ingest.transcribe_real_audio(
                    audio_path=self.sample_wav,
                    output_json_path=out_json,
                    model_size="small",
                )
            self.assertIn("faster-whisper is not installed", str(ctx.exception))
            self.assertIn("mock subtitles are strictly prohibited", str(ctx.exception))
        # Verify no file was created
        self.assertFalse(out_json.exists())

    def test_zero_transcribed_words_aborts_immediately(self):
        """Verify that transcription returning 0 words aborts with a descriptive error."""
        out_json = self.work_dir / "transcript.json"

        # Mock WhisperModel returning an empty iterator or segment with no words
        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance

        # Segment with 0 words (silent audio)
        empty_seg = MagicMock()
        empty_seg.id = 0
        empty_seg.start = 0.0
        empty_seg.end = 5.0
        empty_seg.text = ""
        empty_seg.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 5.0

        mock_instance.transcribe.return_value = ([empty_seg], mock_info)

        with patch.object(ingest, "WhisperModel", mock_model_cls):
            with self.assertRaises(RuntimeError) as ctx:
                ingest.transcribe_real_audio(
                    audio_path=self.sample_wav,
                    output_json_path=out_json,
                    model_size="small",
                )
            self.assertIn("produced 0 transcribed words", str(ctx.exception))
            self.assertIn("mock or fake subtitles is strictly prohibited", str(ctx.exception))

        self.assertFalse(out_json.exists())

    # --------------------------------------------------------------------------
    # 2. Real Word-Level Transcription & pycaps JSON Schema Compatibility
    # --------------------------------------------------------------------------

    def test_successful_transcription_pycaps_schema(self):
        """Verify output JSON structure matches Whisper / pycaps specifications."""
        out_json = self.work_dir / "transcript.json"

        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance

        # Create mock words
        word1 = MagicMock()
        word1.word = "Welcome"
        word1.start = 0.5
        word1.end = 1.0
        word1.probability = 0.95

        word2 = MagicMock()
        word2.word = "everyone"
        word2.start = 1.05
        word2.end = 1.6
        word2.probability = 0.98

        seg = MagicMock()
        seg.id = 1
        seg.seek = 0
        seg.start = 0.5
        seg.end = 1.6
        seg.text = "Welcome everyone"
        seg.tokens = [123, 456]
        seg.temperature = 0.0
        seg.avg_logprob = -0.15
        seg.compression_ratio = 1.1
        seg.no_speech_prob = 0.02
        seg.words = [word1, word2]

        mock_info = MagicMock()
        mock_info.language = "hi"
        mock_info.language_probability = 0.92
        mock_info.duration = 2.0

        mock_instance.transcribe.return_value = ([seg], mock_info)

        with patch.object(ingest, "WhisperModel", mock_model_cls):
            result = ingest.transcribe_real_audio(
                audio_path=self.sample_wav,
                output_json_path=out_json,
                model_size="small",
                device="cpu",
                compute_type="int8",
            )

        # Check call arguments
        mock_instance.transcribe.assert_called_once()
        _, kwargs = mock_instance.transcribe.call_args
        self.assertTrue(kwargs.get("vad_filter"))
        self.assertTrue(kwargs.get("word_timestamps"))

        # Verify on-disk file
        self.assertTrue(out_json.is_file())
        with open(out_json, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["language"], "hi")
        self.assertEqual(saved_data["total_words_count"], 2)
        self.assertEqual(saved_data["segments_count"], 1)
        self.assertEqual(len(saved_data["segments"]), 1)

        # Check pycaps readable fields
        first_seg = saved_data["segments"][0]
        self.assertEqual(first_seg["text"], "Welcome everyone")
        self.assertEqual(len(first_seg["words"]), 2)
        self.assertEqual(first_seg["words"][0]["word"], "Welcome")
        self.assertEqual(first_seg["words"][0]["start"], 0.5)
        self.assertEqual(first_seg["words"][0]["end"], 1.0)
        self.assertEqual(first_seg["words"][1]["word"], "everyone")

    # --------------------------------------------------------------------------
    # 3. Cookie Decoding & Lifecycle Management
    # --------------------------------------------------------------------------

    def test_cookie_decoding_and_validation(self):
        """Verify Netscape cookie decoding, padding recovery, and invalid domain rejection."""
        netscape_content = (
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t1799999999\tLOGIN_INFO\taf93j20fj3\n"
        )
        b64_unpadded = base64.b64encode(netscape_content.encode("utf-8")).decode("utf-8").rstrip("=")

        decoded = ingest.decode_and_validate_cookies(b64_unpadded)
        self.assertIn(".youtube.com", decoded)

        # Non-youtube domain should fail
        bad_content = "# Netscape\n.example.com\tTRUE\t/\tTRUE\t1799999999\tNAME\tVAL\n"
        bad_b64 = base64.b64encode(bad_content.encode("utf-8")).decode("utf-8")
        with self.assertRaises(ValueError):
            ingest.decode_and_validate_cookies(bad_b64)

    def test_secure_cookie_context_cleanup(self):
        """Verify temporary cookie file created from YT_COOKIES_B64 is shredded on exit."""
        netscape_content = (
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t1799999999\tLOGIN_INFO\txyz123\n"
        )
        b64 = base64.b64encode(netscape_content.encode("utf-8")).decode("utf-8")

        temp_path = None
        with patch.dict(os.environ, {"YT_COOKIES_B64": b64}):
            with ingest.secure_cookie_context() as cookie_file:
                self.assertIsNotNone(cookie_file)
                temp_path = Path(cookie_file)
                self.assertTrue(temp_path.exists())

        # After context exit, file must be destroyed
        self.assertFalse(temp_path.exists())

    # --------------------------------------------------------------------------
    # 4. High-Resolution Video Download Configuration
    # --------------------------------------------------------------------------

    @patch("yt_dlp.YoutubeDL")
    def test_download_high_res_video_format_opts(self, mock_ydl_cls):
        """Verify format selector string specifies up to 2160p (4K) and merges with FFmpeg."""
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        fake_out = self.work_dir / "target_source.mp4"
        fake_downloaded = self.work_dir / "download_target_source.mp4"
        fake_downloaded.write_bytes(b"fake video data")

        mock_ydl.extract_info.return_value = {"id": "123", "ext": "mp4"}
        mock_ydl.prepare_filename.return_value = str(fake_downloaded)

        with patch("ingest.probe_video_metadata") as mock_probe:
            mock_probe.return_value = {
                "resolution": "3840x2160",
                "fps": 60.0,
                "vcodec": "vp9",
                "acodec": "opus",
                "duration": 120.0,
                "file_size_mb": 150.0,
                "file_path": str(fake_out),
            }

            result = ingest.download_high_res_video(
                url="https://youtu.be/testvideo",
                output_path=fake_out,
                max_height=2160,
                ffmpeg_bin="ffmpeg.exe",
            )

            # Check format selector passed to yt-dlp
            ydl_call_opts = mock_ydl_cls.call_args[0][0]
            self.assertIn("bv*[height<=2160]+ba/b[height<=2160]", ydl_call_opts["format"])
            self.assertEqual(ydl_call_opts["merge_output_format"], "mp4")
            self.assertEqual(ydl_call_opts["ffmpeg_location"], "ffmpeg.exe")
            self.assertEqual(result["resolution"], "3840x2160")

    # --------------------------------------------------------------------------
    # 5. FFmpeg Audio Extraction & Metadata Probing
    # --------------------------------------------------------------------------

    @patch("subprocess.run")
    def test_extract_audio_16k_mono_command(self, mock_subproc):
        """Verify FFmpeg extracts 16kHz mono pcm_s16le audio."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")

        fake_vid = self.work_dir / "source.mp4"
        fake_vid.write_bytes(b"fake video")
        fake_aud = self.work_dir / "audio_out.wav"

        # Simulate FFmpeg producing the file
        def fake_ffmpeg_run(cmd, **kwargs):
            fake_aud.write_bytes(b"fake wav data")
            return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = fake_ffmpeg_run

        out_path = ingest.extract_audio_16k_mono(
            video_path=fake_vid,
            output_audio_path=fake_aud,
            ffmpeg_bin="ffmpeg.exe",
        )

        self.assertEqual(out_path, fake_aud.resolve())
        cmd_called = mock_subproc.call_args[0][0]
        self.assertIn("-acodec", cmd_called)
        self.assertIn("pcm_s16le", cmd_called)
        self.assertIn("-ar", cmd_called)
        self.assertIn("16000", cmd_called)
        self.assertIn("-ac", cmd_called)
        self.assertIn("1", cmd_called)

    @patch("subprocess.run")
    def test_probe_video_metadata_parsing(self, mock_subproc):
        """Verify FFmpeg probe correctly extracts resolution, fps, codecs, and duration."""
        sample_stderr = (
            "Input #0, mov,mp4,m4a,3gp,from 'source.mp4':\n"
            "  Duration: 00:01:15.50, start: 0.000000, bitrate: 5400 kb/s\n"
            "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv), 3840x2160 [SAR 1:1 DAR 16:9], 60 fps, 60 tbr\n"
            "  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 192 kb/s\n"
        )
        mock_subproc.return_value = MagicMock(returncode=0, stderr=sample_stderr)

        fake_vid = self.work_dir / "source.mp4"
        fake_vid.write_bytes(b"dummy")

        meta = ingest.probe_video_metadata(fake_vid, ffmpeg_bin="ffmpeg.exe")
        self.assertEqual(meta["width"], 3840)
        self.assertEqual(meta["height"], 2160)
        self.assertEqual(meta["resolution"], "3840x2160")
        self.assertEqual(meta["fps"], 60.0)
        self.assertEqual(meta["vcodec"], "h264")
        self.assertEqual(meta["acodec"], "aac")
        self.assertEqual(meta["duration"], 75.5)


if __name__ == "__main__":
    unittest.main()
