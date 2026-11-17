"""Gemini 2.5 Flash Integration Module for Viral Segment Detection.

Analyzes gameplay/stream transcripts using Gemini 2.5 Flash to identify
high-retention gaming hooks and funny/intense moments suited for US YouTube Shorts.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("ai_analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYSTEM_PROMPT = (
    "Analyze this transcript for high-retention, fast-paced moments, intense gameplay reveals, "
    "and comedic reactions suited for US YouTube Shorts viewers. "
    "Return a JSON array of objects with keys: `start_time`, `end_time`, `hook_title`, and `viral_score`."
)


@dataclass
class ViralSegment:
    """Represents an extracted viral clip candidate."""
    start_time: float
    end_time: float
    hook_title: str
    viral_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_json_array_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """Vulnerability Prevention: Regex cleaner to safely extract a JSON array.

    Strips markdown code blocks, backticks, and any narrative preamble/postscript,
    isolating only the content between the first '[' and last ']' characters.

    Args:
        raw_text: Raw output string from the language model.

    Returns:
        List[Dict[str, Any]]: Parsed JSON list.

    Raises:
        ValueError: If no valid JSON array bounds or valid JSON is found.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Model response is empty.")

    # 1. Strip markdown code fences (```json or ```) and rogue backticks
    cleaned = re.sub(r"^```(?:json)?", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.replace("`", "").strip()

    # 2. Extract content strictly between the first '[' and the last ']'
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON array delimiters '[' and ']' found in model output.")

    json_candidate = match.group(0).strip()

    # 3. Parse JSON safely
    try:
        parsed = json.loads(json_candidate)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError as exc:
        # Secondary recovery attempt: remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([\]}])", r"\1", json_candidate)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        raise ValueError(f"Failed to decode JSON array from LLM response: {exc}") from exc


def generate_fallback_segments(
    total_duration: float = 120.0,
    segment_duration: float = 45.0,
    max_segments: int = 2,
) -> List[ViralSegment]:
    """Provide fallback default segment windows if the model fails or returns empty list.

    Args:
        total_duration: Total video/audio duration in seconds.
        segment_duration: Default clip duration in seconds (typically 30-55s).
        max_segments: Maximum number of fallback segments to generate.

    Returns:
        List[ViralSegment]: Safe default clip windows.
    """
    logger.warning("Generating default fallback segment windows (duration=%.1fs)...", total_duration)

    fallbacks: List[ViralSegment] = []
    effective_duration = min(segment_duration, total_duration)

    # Segment 1: Starting around 10% into the video to skip intro
    start_1 = min(15.0, max(0.0, total_duration - effective_duration))
    end_1 = min(start_1 + effective_duration, total_duration)
    fallbacks.append(
        ViralSegment(
            start_time=round(start_1, 2),
            end_time=round(end_1, 2),
            hook_title="INSANE GTA 6 Gameplay Reveal! 🚨 (Wait For It)",
            viral_score=8.5,
        )
    )

    if max_segments > 1 and total_duration > (effective_duration * 1.5):
        # Segment 2: Midpoint highlight
        start_2 = min(total_duration / 2, total_duration - effective_duration)
        end_2 = min(start_2 + effective_duration, total_duration)
        if start_2 > end_1 or abs(start_2 - start_1) > 20:
            fallbacks.append(
                ViralSegment(
                    start_time=round(start_2, 2),
                    end_time=round(end_2, 2),
                    hook_title="HE COULD NOT BELIEVE THIS GTA 6 GLITCH 💀",
                    viral_score=8.0,
                )
            )

    return fallbacks[:max_segments]


class GeminiSegmentDetector:
    """Detects high-retention gaming clips from transcripts using Gemini 2.5 Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Initialize the detector with the official google-genai SDK.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY env var.
            model_name: Default model identifier ('gemini-2.5-flash').
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini API calls will require an active key.")

        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Instantiate official google-genai Client."""
        if self._client is None:
            if genai is None:
                raise RuntimeError(
                    "google-genai package is not installed. Install via: pip install google-genai"
                )
            if not self.api_key:
                raise ValueError("Environment variable 'GEMINI_API_KEY' is missing or empty.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_user_prompt(
        self,
        transcript_text: str,
        total_duration: Optional[float] = None,
        video_title: Optional[str] = None,
    ) -> str:
        """Construct the prompt formatted with timestamps and context."""
        prompt_parts = []
        if video_title:
            prompt_parts.append(f"Video Title: {video_title}")
        if total_duration:
            prompt_parts.append(f"Total Video Duration: {total_duration:.1f} seconds")

        prompt_parts.append(
            "\nAnalyze the following transcript with timestamps and select the top 1 to 3 "
            "most viral, fast-paced gaming moments (optimal clip duration: 30-55 seconds):\n"
        )
        prompt_parts.append(transcript_text)
        return "\n".join(prompt_parts)

    def detect_viral_segments(
        self,
        transcript_text: str,
        total_duration: float = 120.0,
        video_title: Optional[str] = None,
        max_clips: int = 3,
    ) -> List[ViralSegment]:
        """Detect viral segments from transcript using Gemini 2.5 Flash with fallback protection.

        Args:
            transcript_text: Full transcript with timestamps.
            total_duration: Video duration in seconds.
            video_title: Optional video title for context.
            max_clips: Maximum segments to return.

        Returns:
            List[ViralSegment]: Structured viral clip segments.
        """
        if not transcript_text or not transcript_text.strip():
            logger.warning("Empty transcript provided to GeminiSegmentDetector. Using fallbacks.")
            return generate_fallback_segments(total_duration=total_duration, max_segments=max_clips)

        try:
            client = self._get_client()
            prompt = self._build_user_prompt(transcript_text, total_duration, video_title)

            logger.info("Calling Gemini 2.5 Flash for viral hook analysis...")
            
            # Configure request using google-genai SDK
            config_params = {
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.7,
            }
            if types and hasattr(types, "GenerateContentConfig"):
                config = types.GenerateContentConfig(**config_params)
            else:
                config = config_params

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            raw_text = getattr(response, "text", "") or str(response)
            logger.info("Received Gemini response (length: %d chars)", len(raw_text))

            # Vulnerability Prevention: Regex extraction and strict JSON parsing
            parsed_data = extract_json_array_from_text(raw_text)

            segments: List[ViralSegment] = []
            for item in parsed_data:
                if not isinstance(item, dict):
                    continue

                start = float(item.get("start_time", 0.0))
                end = float(item.get("end_time", start + 45.0))
                title = str(item.get("hook_title", "Insane GTA 6 Moment")).strip()
                score = float(item.get("viral_score", 8.0))

                # Sanitize duration for YouTube Shorts (must be < 60s, ideally 30-55s)
                clip_len = end - start
                if clip_len <= 5.0 or start < 0 or end > total_duration + 5:
                    logger.warning("Skipping out-of-bounds segment: start=%.1f, end=%.1f", start, end)
                    continue

                if clip_len > 58.0:
                    end = start + 55.0  # Clamp to safe Shorts length

                segments.append(
                    ViralSegment(
                        start_time=round(start, 2),
                        end_time=round(end, 2),
                        hook_title=title,
                        viral_score=round(score, 2),
                    )
                )

            if not segments:
                logger.warning("Gemini returned zero valid segments. Triggering fallback windows.")
                return generate_fallback_segments(total_duration=total_duration, max_segments=max_clips)

            # Sort descending by viral_score
            segments.sort(key=lambda s: s.viral_score, reverse=True)
            return segments[:max_clips]

        except Exception as exc:
            logger.error("Gemini 2.5 Flash analysis failed or encountered error: %s", exc)
            # Vulnerability Prevention: Graceful fallback guarantees pipeline continuity
            return generate_fallback_segments(total_duration=total_duration, max_segments=max_clips)
