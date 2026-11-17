"""Unit tests for audio extraction and faster-whisper transcription module."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.transcriber import AudioTranscriber, TranscriptionResult, WordTimestamp


class TestAudioTranscriber(unittest.TestCase):

    def setUp(self):
        self.transcriber = AudioTranscriber(model_size="base.en", device="cpu", compute_type="int8")

    def test_word_timestamp_and_result_dataclasses(self):
        """Verify structured output objects serialize correctly."""
        word = WordTimestamp(start=1.234, end=1.567, text="GTA", probability=0.98)
        self.assertEqual(word.to_dict()["text"], "GTA")
        self.assertEqual(word.to_dict()["start"], 1.234)

        result = TranscriptionResult(
            language="en",
            duration=60.0,
            words=[word],
            full_text="GTA",
            segments=[{"start": 1.0, "end": 2.0, "text": "GTA"}],
        )
        res_dict = result.to_dict()
        self.assertEqual(res_dict["language"], "en")
        self.assertEqual(len(res_dict["words"]), 1)
        self.assertEqual(res_dict["words"][0]["text"], "GTA")

    def test_extract_audio_missing_video_raises_filenotfound(self):
        """Extraction must fail if source video does not exist."""
        with self.assertRaises(FileNotFoundError):
            self.transcriber.extract_audio("non_existent_video_file.mp4")

    @patch("subprocess.run")
    def test_extract_audio_ffmpeg_command_flags(self, mock_subprocess_run):
        """Requirement 1: Verify FFmpeg command uses low-bitrate mono 16kHz flags."""
        fake_video = Path("fake_gameplay.mp4")
        fake_video.touch()
        self.addCleanup(lambda: fake_video.unlink(missing_ok=True))

        fake_audio = fake_video.with_name("fake_gameplay_audio.mp3")
        # Simulate successful FFmpeg run creating output file
        def fake_run(cmd, **kwargs):
            fake_audio.write_bytes(b"dummy_mp3_content")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        mock_subprocess_run.side_effect = fake_run

        try:
            out_path = self.transcriber.extract_audio(str(fake_video))
            self.assertTrue(out_path.exists())

            # Inspect FFmpeg args passed to subprocess.run
            call_args = mock_subprocess_run.call_args[0][0]
            self.assertEqual(call_args[0], "ffmpeg")
            self.assertIn("-vn", call_args)       # Strip video
            self.assertIn("-ac", call_args)       # Mono
            self.assertIn("1", call_args)
            self.assertIn("-ar", call_args)       # 16kHz
            self.assertIn("16000", call_args)
            self.assertIn("-b:a", call_args)      # 64k bitrate
            self.assertIn("64k", call_args)
        finally:
            fake_audio.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_audio_timeout_exception(self, mock_subprocess_run):
        """Vulnerability Prevention: TimeoutExpired must be caught and raised gracefully."""
        fake_video = Path("test_timeout.mp4")
        fake_video.touch()
        self.addCleanup(lambda: fake_video.unlink(missing_ok=True))

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=180)

        with self.assertRaises(TimeoutError) as ctx:
            self.transcriber.extract_audio(str(fake_video), timeout_seconds=180)
        self.assertIn("timed out after 180s", str(ctx.exception))

    @patch("subprocess.run")
    def test_extract_audio_corrupted_codec_exception(self, mock_subprocess_run):
        """Vulnerability Prevention: Detect corrupted video/audio stream errors."""
        fake_video = Path("corrupted.mp4")
        fake_video.touch()
        self.addCleanup(lambda: fake_video.unlink(missing_ok=True))

        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="Invalid data found when processing input",
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.transcriber.extract_audio(str(fake_video))
        self.assertIn("Corrupted or invalid video/audio container", str(ctx.exception))

    def test_transcribe_rejects_raw_video_to_prevent_oom(self):
        """Vulnerability Prevention: Enforce that only extracted audio can be passed to transcribe()."""
        fake_video = Path("heavy_raw_footage.mp4")
        fake_video.touch()
        self.addCleanup(lambda: fake_video.unlink(missing_ok=True))

        with self.assertRaises(ValueError) as ctx:
            self.transcriber.transcribe(str(fake_video))
        self.assertIn("Only lightweight extracted audio files", str(ctx.exception))

    def test_transcribe_with_mocked_whisper_model(self):
        """Requirement 2 & 3: Transcribe audio track returning word-level timestamps."""
        fake_audio = Path("test_audio.mp3")
        fake_audio.write_bytes(b"dummy_audio_bytes")
        self.addCleanup(lambda: fake_audio.unlink(missing_ok=True))

        # Mock WhisperModel and its output generator
        mock_model = MagicMock()

        # Mock Word objects
        word1 = MagicMock(start=0.5, end=0.9, word=" Insane", probability=0.95)
        word2 = MagicMock(start=1.0, end=1.5, word=" stunt!", probability=0.99)

        # Mock Segment object
        segment = MagicMock(start=0.5, end=1.5, text="Insane stunt!", words=[word1, word2])
        info = MagicMock(language="en", duration=2.5)

        mock_model.transcribe.return_value = ([segment], info)
        self.transcriber._model = mock_model

        result = self.transcriber.transcribe(str(fake_audio))

        self.assertEqual(result.language, "en")
        self.assertEqual(result.duration, 2.5)
        self.assertEqual(result.full_text, "Insane stunt!")
        self.assertEqual(len(result.words), 2)
        self.assertEqual(result.words[0].text, "Insane")
        self.assertEqual(result.words[0].start, 0.5)
        self.assertEqual(result.words[1].text, "stunt!")
        self.assertEqual(result.words[1].end, 1.5)


if __name__ == "__main__":
    unittest.main()
