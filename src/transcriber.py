"""Audio Extraction and Transcription Module using faster-whisper and FFmpeg.

Extracts lightweight 16kHz mono audio from high-resolution video and runs
faster-whisper locally with int8 CPU quantization to prevent runner OOM.
"""

import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger("transcriber")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class WordTimestamp:
    """Represents a single word with start/end timestamps."""
    start: float
    end: float
    text: str
    probability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Complete transcription output containing word timestamps and full text."""
    language: str
    duration: float
    words: List[WordTimestamp]
    full_text: str
    segments: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "duration": self.duration,
            "full_text": self.full_text,
            "words": [w.to_dict() for w in self.words],
            "segments": self.segments,
        }


class AudioTranscriber:
    """Extracts low-bitrate mono audio and transcribes locally using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
    ):
        """Initialize the transcriber.

        Args:
            model_size: Model variant ('base.en', 'small.en', etc.)
            device: 'cpu' for standard GitHub Actions runners.
            compute_type: 'int8' for 8-bit quantized low-memory inference.
            cpu_threads: Number of CPU threads to allocate.
        """
        self.model_size = os.environ.get("WHISPER_MODEL_SIZE", model_size)
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model = None

    def _get_model(self):
        """Lazy-load the Whisper model into memory on demand."""
        if self._model is None:
            if WhisperModel is None:
                raise RuntimeError(
                    "faster-whisper is not installed. Install via: pip install faster-whisper"
                )
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def extract_audio(
        self,
        video_path: str,
        output_audio_path: Optional[str] = None,
        timeout_seconds: int = 180,
    ) -> Path:
        """Extract a lightweight mono 16kHz MP3 track from a video file using FFmpeg.

        Vulnerability Prevention:
            Extracting 64kbps mono audio guarantees that high-resolution video
            frames are never passed into Whisper, eliminating RAM exhaustion (OOM)
            on ephemeral GitHub Actions runners.

        Args:
            video_path: Path to source video file.
            output_audio_path: Destination path for extracted audio.
            timeout_seconds: Subprocess timeout limit to avoid hanging.

        Returns:
            Path: Path to the generated audio file.

        Raises:
            FileNotFoundError: If video file is not found.
            RuntimeError: If FFmpeg fails, audio stream is missing, or codec is corrupted.
            TimeoutError: If FFmpeg hangs during extraction.
        """
        source_path = Path(video_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source video file not found: {source_path}")

        if output_audio_path:
            dest_audio = Path(output_audio_path).resolve()
        else:
            dest_audio = source_path.with_name(f"{source_path.stem}_audio.mp3")

        dest_audio.parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command:
        # -vn: strip video stream
        # -ac 1: downmix to mono (50% memory reduction)
        # -ar 16000: resample to 16kHz (native Whisper sample rate)
        # -b:a 64k: low-bitrate compression
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(source_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            "-f", "mp3",
            str(dest_audio),
        ]

        logger.info("Extracting low-bitrate audio from %s to %s...", source_path.name, dest_audio.name)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            if dest_audio.exists():
                dest_audio.unlink(missing_ok=True)
            raise TimeoutError(
                f"FFmpeg audio extraction timed out after {timeout_seconds}s for {source_path.name}"
            ) from exc

        if result.returncode != 0:
            stderr_msg = result.stderr.strip()
            if dest_audio.exists():
                dest_audio.unlink(missing_ok=True)

            # Detect common audio codec/stream corruption errors
            if "Invalid data found" in stderr_msg or "does not contain any stream" in stderr_msg:
                raise RuntimeError(
                    f"Corrupted or invalid video/audio container: {stderr_msg}"
                )
            if "Output file is empty" in stderr_msg or "matches no streams" in stderr_msg:
                raise RuntimeError(
                    f"No valid audio stream detected in video file: {stderr_msg}"
                )

            raise RuntimeError(f"FFmpeg audio extraction failed (code {result.returncode}): {stderr_msg}")

        if not dest_audio.exists() or dest_audio.stat().st_size == 0:
            if dest_audio.exists():
                dest_audio.unlink(missing_ok=True)
            raise RuntimeError(
                f"Extracted audio file is empty or missing after extraction: {dest_audio.name}"
            )

        logger.info(
            "Extracted audio track successfully: %s (size: %.2f KB)",
            dest_audio.name,
            dest_audio.stat().st_size / 1024,
        )
        return dest_audio

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        beam_size: int = 5,
    ) -> TranscriptionResult:
        """Transcribe extracted audio track using faster-whisper.

        Args:
            audio_path: Path to extracted lightweight audio track (e.g. audio.mp3).
            language: Target language ('en' for English gaming streams).
            beam_size: Beam search width (default: 5).

        Returns:
            TranscriptionResult: Structured result with word-level timestamps.

        Raises:
            FileNotFoundError: If audio file is missing.
            RuntimeError: If model transcription fails.
        """
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found for transcription: {audio_file}")

        # Safety check: enforce audio file extension to prevent passing raw video
        valid_audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
        if audio_file.suffix.lower() not in valid_audio_extensions:
            raise ValueError(
                f"Invalid file passed to transcribe(): {audio_file.name}. "
                "Only lightweight extracted audio files (.mp3, .wav) are accepted to prevent OOM."
            )

        model = self._get_model()
        logger.info("Starting transcription of %s with faster-whisper...", audio_file.name)

        try:
            segments, info = model.transcribe(
                str(audio_file),
                language=language,
                beam_size=beam_size,
                word_timestamps=True,
                vad_filter=True,  # Filter out silence/background noise
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            word_objects: List[WordTimestamp] = []
            segment_records: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for seg in segments:
                full_text_parts.append(seg.text.strip())
                seg_dict = {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                segment_records.append(seg_dict)

                if seg.words:
                    for w in seg.words:
                        cleaned_word = w.word.strip()
                        if cleaned_word:
                            word_objects.append(
                                WordTimestamp(
                                    start=round(w.start, 3),
                                    end=round(w.end, 3),
                                    text=cleaned_word,
                                    probability=round(getattr(w, "probability", 1.0), 3),
                                )
                            )

            full_text = " ".join(full_text_parts)
            logger.info(
                "Transcription completed: %d segments, %d words (audio duration: %.1fs)",
                len(segment_records),
                len(word_objects),
                info.duration,
            )

            return TranscriptionResult(
                language=info.language,
                duration=info.duration,
                words=word_objects,
                full_text=full_text,
                segments=segment_records,
            )

        except Exception as exc:
            logger.error("Failed during faster-whisper transcription of %s: %s", audio_file.name, exc)
            raise RuntimeError(f"Whisper transcription failed for {audio_file.name}: {exc}") from exc

    def process_video(
        self,
        video_path: str,
        keep_audio_file: bool = False,
    ) -> TranscriptionResult:
        """End-to-end convenience method: extracts audio and transcribes.

        Guarantees cleanup of the intermediate audio file unless keep_audio_file=True.
        """
        extracted_audio = self.extract_audio(video_path)
        try:
            return self.transcribe(str(extracted_audio))
        finally:
            if not keep_audio_file and extracted_audio.exists():
                try:
                    extracted_audio.unlink()
                    logger.info("Cleaned up intermediate audio file: %s", extracted_audio.name)
                except Exception as err:
                    logger.warning("Failed to remove intermediate audio: %s", err)
