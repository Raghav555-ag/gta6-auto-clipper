"""Viral Short-Form Moment Finder using Gemini 2.5 Flash and Audio Energy Analysis.

Implements short-form retention principles (0-3s Hook, Story Arc, Hook Banners, Micro SFX)
grounded on multimodal cues: speech transcripts and audio loudness/reaction spikes.
Snaps clip boundaries to word timestamps with padding to ensure zero speech clipping.
"""

import json
import logging
import math
import os
import re
import struct
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import numpy as np
except ImportError:
    np = None

from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viral_finder")

DEFAULT_SKILL_PATH = Path(__file__).parent / "skills" / "viral_editor_skill.md"


# ==============================================================================
# 1. Pydantic Schemas for Strict Gemini Structured Output
# ==============================================================================

class SfxEvent(BaseModel):
    time_sec: float = Field(
        description="Timestamp in seconds relative to the clip start (0.0s = start of clip)"
    )
    type: str = Field(
        description="Sound effect type: strictly 'whoosh', 'pop', or 'boom'"
    )


class ViralShortCandidate(BaseModel):
    start_sec: float = Field(
        description="Start timestamp in seconds within the source video"
    )
    end_sec: float = Field(
        description="End timestamp in seconds within the source video (duration: 18s - 55s)"
    )
    hook_banner: str = Field(
        description="Curiosity headline (4 to 7 words) displayed on-screen for the first 3 seconds"
    )
    title: str = Field(
        description="YouTube Shorts title with relevant hashtags like #GTA6 #Shorts"
    )
    viral_score: int = Field(
        ge=1,
        le=100,
        description="Predicted viewer retention & viral score from 1 to 100",
    )
    reason: str = Field(
        description="Editorial breakdown of the Setup -> Escalation -> Payoff arc"
    )
    sfx_events: List[SfxEvent] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 subtle micro SFX cues (whoosh/pop/boom) relative to clip start",
    )


# ==============================================================================
# 2. Audio Energy Peak Detection (Laughter / Yelling / Chaos Spikes)
# ==============================================================================

def detect_audio_energy_peaks(
    audio_path: Union[str, Path],
    window_sec: float = 0.5,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Analyze audio waveform to identify loudness spikes (laughs, screams, explosions).

    Uses pure standard library (wave + struct) or numpy if available, ensuring
    zero external compilation dependencies.

    Args:
        audio_path: Path to 16-bit PCM WAV file (e.g. 16kHz mono).
        window_sec: Duration in seconds of each energy evaluation window.
        top_n: Maximum number of peak moments to return.

    Returns:
        List[Dict[str, Any]]: High-energy peak intervals with relative loudness scores.
    """
    path = Path(audio_path).resolve()
    if not path.is_file():
        logger.warning("Audio file not found for energy analysis: %s", path)
        return []

    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if n_channels != 1 or sampwidth != 2:
                # Fallback: only analyze standard 16-bit mono PCM wav files
                logger.info("Audio is not 16-bit mono (channels=%d, width=%d). Skipping energy analysis.", n_channels, sampwidth)
                return []

            window_frames = int(framerate * window_sec)
            if window_frames <= 0 or n_frames <= 0:
                return []

            num_windows = n_frames // window_frames
            energies: List[Tuple[float, float, float]] = []

            for i in range(num_windows):
                raw_bytes = wf.readframes(window_frames)
                if not raw_bytes:
                    break

                num_samples = len(raw_bytes) // 2
                samples = struct.unpack(f"<{num_samples}h", raw_bytes)

                if np is not None:
                    arr = np.array(samples, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(arr**2)))
                else:
                    sum_sq = sum(s * s for s in samples)
                    rms = math.sqrt(sum_sq / len(samples)) if samples else 0.0

                start_t = round(i * window_sec, 2)
                end_t = round(start_t + window_sec, 2)
                energies.append((start_t, end_t, rms))

        if not energies:
            return []

        # Find threshold (top 15% energy spikes)
        rms_values = sorted([e[2] for e in energies])
        if not rms_values or rms_values[-1] == 0:
            return []

        max_rms = rms_values[-1]
        p85_idx = int(len(rms_values) * 0.85)
        p85_thresh = rms_values[p85_idx]

        # Merge contiguous energy spikes into reaction moments
        peaks: List[Dict[str, Any]] = []
        in_spike = False
        spike_start = 0.0
        spike_max = 0.0

        for start_t, end_t, rms in energies:
            if rms >= p85_thresh:
                if not in_spike:
                    in_spike = True
                    spike_start = start_t
                    spike_max = rms
                else:
                    spike_max = max(spike_max, rms)
            else:
                if in_spike:
                    in_spike = False
                    normalized_score = round(min(1.0, spike_max / max_rms), 2)
                    reaction_type = "high_shout_or_chaos" if normalized_score > 0.85 else "laughter_or_reaction"
                    peaks.append({
                        "start_sec": spike_start,
                        "end_sec": end_t,
                        "intensity": normalized_score,
                        "reaction_type": reaction_type,
                    })

        # Sort descending by intensity
        peaks.sort(key=lambda x: x["intensity"], reverse=True)
        return peaks[:top_n]

    except Exception as exc:
        logger.warning("Could not extract audio energy peaks: %s", exc)
        return []


# ==============================================================================
# 3. Speech Boundary Snapping (Prevent Mid-Word Cuts)
# ==============================================================================

def snap_to_word_boundaries(
    start_sec: float,
    end_sec: float,
    transcript_segments: List[Dict[str, Any]],
    padding_sec: float = 0.25,
    min_duration: float = 18.0,
    max_duration: float = 55.0,
    total_duration: Optional[float] = None,
) -> Tuple[float, float]:
    """Snap start_sec and end_sec to real speech/sentence boundaries with padding.

    Ensures speech is never cut off mid-word or mid-syllable, providing a natural
    0.25s verbal lead-in and clean trailing exit.

    Args:
        start_sec: Raw proposed start timestamp.
        end_sec: Raw proposed end timestamp.
        transcript_segments: List of speech segments/words with 'start', 'end', 'text'.
        padding_sec: Lead-in and lead-out buffer (default 0.25s).
        min_duration: Minimum enforced clip duration (default 18s).
        max_duration: Maximum enforced clip duration (default 55s).
        total_duration: Video duration limit.

    Returns:
        Tuple[float, float]: Snapped (start_sec, end_sec).
    """
    if not transcript_segments:
        # If no transcript available, clamp timestamps to duration limits
        dur = max(min_duration, min(max_duration, end_sec - start_sec))
        clean_start = max(0.0, start_sec)
        clean_end = clean_start + dur
        if total_duration and clean_end > total_duration:
            clean_end = total_duration
            clean_start = max(0.0, clean_end - dur)
        return round(clean_start, 2), round(clean_end, 2)

    # Flatten any nested word tokens or use segment boundaries
    word_tokens: List[Dict[str, Any]] = []
    for seg in transcript_segments:
        if "words" in seg and isinstance(seg["words"], list) and seg["words"]:
            for w in seg["words"]:
                word_tokens.append({
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "text": str(w.get("text", "")).strip(),
                })
        else:
            word_tokens.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
            })

    if not word_tokens:
        return round(max(0.0, start_sec), 2), round(end_sec, 2)

    # 1. Snap start_sec to the nearest word start (preferring starting slightly before)
    best_start = start_sec
    min_start_diff = float("inf")
    for w in word_tokens:
        diff = abs(w["start"] - start_sec)
        if diff < min_start_diff and diff <= 3.0:
            min_start_diff = diff
            # 0-3s Hook Rule: Start 0.25s before spoken word to avoid clipped attack
            best_start = max(0.0, w["start"] - padding_sec)

    # 2. Snap end_sec to the nearest word end (preferring ending slightly after)
    best_end = end_sec
    min_end_diff = float("inf")
    for w in word_tokens:
        diff = abs(w["end"] - end_sec)
        if diff < min_end_diff and diff <= 4.0:
            min_end_diff = diff
            # Natural verbal exit: add 0.25s padding after speech finishes
            best_end = w["end"] + padding_sec

    # 3. Enforce min and max duration bounds
    dur = best_end - best_start
    if dur < min_duration:
        best_end = best_start + min_duration
    elif dur > max_duration:
        best_end = best_start + max_duration

    # 4. Clamp to video boundaries if provided
    best_start = max(0.0, best_start)
    if total_duration:
        if best_end > total_duration:
            best_end = total_duration
            best_start = max(0.0, best_end - max(min_duration, dur))

    return round(best_start, 2), round(best_end, 2)


# ==============================================================================
# 4. Gemini 2.5 Flash Viral Segment Detection
# ==============================================================================

def load_system_instruction(skill_file_path: Optional[Union[str, Path]] = None) -> str:
    """Load the viral editor retention principles from skills/viral_editor_skill.md."""
    target_path = Path(skill_file_path) if skill_file_path else DEFAULT_SKILL_PATH
    target_path = target_path.resolve()

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Knowledge file '{target_path.name}' not found at {target_path}. "
            "Ensure 'skills/viral_editor_skill.md' exists."
        )

    return target_path.read_text(encoding="utf-8")


def format_transcript_prompt(
    transcript_segments: List[Dict[str, Any]],
    energy_peaks: List[Dict[str, Any]],
    video_title: Optional[str] = None,
    total_duration: Optional[float] = None,
    max_clips: int = 3,
) -> str:
    """Format speech transcripts and audio energy spikes into a rich prompt."""
    prompt_sections = []

    header = ["=== VIDEO CONTEXT ==="]
    if video_title:
        header.append(f"Video Title: {video_title}")
    if total_duration:
        header.append(f"Total Video Duration: {total_duration:.1f} seconds ({int(total_duration // 60)}m {int(total_duration % 60)}s)")
    header.append(f"Target Number of Clips: Up to {max_clips} independent viral Shorts")
    prompt_sections.append("\n".join(header))

    # Add Audio Loudness / Reaction Peaks
    if energy_peaks:
        peak_lines = ["\n=== AUDIO ENERGY SPIKES (CREATOR LAUGHTER / SHOUTS / CHAOS) ==="]
        for p in energy_peaks:
            m_s, s_s = int(p["start_sec"] // 60), int(p["start_sec"] % 60)
            m_e, s_e = int(p["end_sec"] // 60), int(p["end_sec"] % 60)
            peak_lines.append(
                f"- [{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}] {p['reaction_type'].upper()} (Intensity: {p['intensity']})"
            )
        prompt_sections.append("\n".join(peak_lines))

    # Add Full Timestamped Speech Transcript
    transcript_lines = ["\n=== TIMESTAMPED SPEECH TRANSCRIPT ==="]
    for seg in transcript_segments:
        m_s, s_s = int(seg["start"] // 60), int(seg["start"] % 60)
        m_e, s_e = int(seg["end"] // 60), int(seg["end"] % 60)
        transcript_lines.append(f"[{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}] {seg['text']}")
    prompt_sections.append("\n".join(transcript_lines))

    prompt_sections.append(
        "\n=== INSTRUCTION ===\n"
        "Apply the Viral Editor Retention Rules from your system instructions:\n"
        "1. Identify moments where speech hooks intersect with audio reactions/chaos.\n"
        "2. Ensure each clip has a distinct Setup -> Escalation -> Payoff arc (18s to 55s).\n"
        "3. Generate a 4-to-7 word hook banner for the first 3 seconds.\n"
        "4. Plan up to 3 subtle SFX cues (whoosh, pop, boom) with timestamps relative to clip start (0.0s).\n"
        "Return the output strictly matching the requested JSON schema."
    )

    return "\n".join(prompt_sections)


def find_viral_moments(
    transcript_data: Union[Dict[str, Any], List[Dict[str, Any]], str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    total_duration: Optional[float] = None,
    video_title: Optional[str] = None,
    max_clips: int = 3,
    min_sec: int = 18,
    max_sec: int = 55,
    api_key: Optional[str] = None,
    skill_doc_path: Optional[Union[str, Path]] = None,
) -> List[ViralShortCandidate]:
    """Analyze video transcript and audio energy using Gemini 2.5 Flash to select viral moments.

    Vulnerability Prevention:
        - NEVER generates dummy or hardcoded text. If GEMINI_API_KEY is missing or the API fails,
          raises a clear RuntimeError and stops.
        - Automatically snaps start_sec and end_sec to real word boundaries with 0.25s padding.

    Args:
        transcript_data: Transcript JSON dict, list of segments, JSON filepath, or raw text.
        audio_path: Optional WAV audio file to compute loudness/reaction spikes.
        total_duration: Total video length in seconds.
        video_title: Contextual title of the video.
        max_clips: Maximum number of viral clips to detect.
        min_sec: Minimum clip duration (default: 18s).
        max_sec: Maximum clip duration (default: 55s).
        api_key: Optional GEMINI_API_KEY override.
        skill_doc_path: Optional path to skills/viral_editor_skill.md.

    Returns:
        List[ViralShortCandidate]: List of validated viral short candidates.

    Raises:
        ValueError: If GEMINI_API_KEY is missing.
        RuntimeError: If google-genai is not installed, or the Gemini API call fails.
    """
    # 1. Enforce GEMINI_API_KEY existence (NO DUMMY TEXT FALLBACK)
    active_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not active_key or not active_key.strip():
        raise ValueError(
            "GEMINI_API_KEY is not set or empty. "
            "A valid Gemini API key is required to detect authentic viral moments. "
            "Please set GEMINI_API_KEY in your environment or .env file before running."
        )

    if genai is None:
        raise RuntimeError(
            "google-genai package is not installed. Install via: pip install google-genai"
        )

    # 2. Parse input transcript data
    segments: List[Dict[str, Any]] = []
    derived_duration = total_duration

    if isinstance(transcript_data, (str, Path)) and Path(transcript_data).is_file():
        with open(transcript_data, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
            if isinstance(raw_json, dict):
                segments = raw_json.get("segments", [])
                derived_duration = derived_duration or raw_json.get("duration")
            elif isinstance(raw_json, list):
                segments = raw_json
    elif isinstance(transcript_data, dict):
        segments = transcript_data.get("segments", [])
        derived_duration = derived_duration or transcript_data.get("duration")
    elif isinstance(transcript_data, list):
        segments = transcript_data
    elif isinstance(transcript_data, str):
        # Plain text
        segments = [{"start": 0.0, "end": derived_duration or 60.0, "text": transcript_data}]

    if not segments:
        raise ValueError("Transcript contains zero speech segments. Cannot perform viral detection.")

    if not derived_duration and segments:
        derived_duration = segments[-1].get("end", 60.0)

    # 3. Detect Audio Energy Peaks
    energy_peaks: List[Dict[str, Any]] = []
    if audio_path:
        logger.info("Analyzing audio energy peaks from %s...", Path(audio_path).name)
        energy_peaks = detect_audio_energy_peaks(audio_path, window_sec=0.5, top_n=15)
        logger.info("Detected %d audio loudness/reaction peak(s).", len(energy_peaks))

    # 4. Load System Instruction
    system_instruction = load_system_instruction(skill_doc_path)
    logger.info("Loaded viral editor skill instructions from %s", (skill_doc_path or DEFAULT_SKILL_PATH))

    # 5. Build User Prompt
    user_prompt = format_transcript_prompt(
        transcript_segments=segments,
        energy_peaks=energy_peaks,
        video_title=video_title,
        total_duration=derived_duration,
        max_clips=max_clips,
    )

    # 6. Call Gemini 2.5 Flash with Structured Schema
    logger.info("Calling Gemini 2.5 Flash for viral moment analysis...")
    client = genai.Client(api_key=active_key)

    config_params = {
        "system_instruction": system_instruction,
        "temperature": 0.6,
        "response_mime_type": "application/json",
        "response_schema": list[ViralShortCandidate],
    }

    try:
        if types and hasattr(types, "GenerateContentConfig"):
            config = types.GenerateContentConfig(**config_params)
        else:
            config = config_params

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=config,
        )
    except Exception as exc:
        logger.error("Gemini 2.5 Flash API call failed: %s", exc)
        raise RuntimeError(f"Gemini API execution failed: {exc}") from exc

    raw_text = getattr(response, "text", "") or str(response)
    if not raw_text.strip():
        raise RuntimeError("Gemini API returned an empty response. Cannot proceed without authentic clips.")

    # 7. Strict JSON Parsing & Pydantic Validation
    try:
        parsed_json = json.loads(raw_text)
    except json.JSONDecodeError:
        # Secondary recovery: isolate content between first '[' and last ']'
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Failed to parse structured JSON from Gemini response:\n{raw_text[:300]}")
        parsed_json = json.loads(match.group(0))

    if not isinstance(parsed_json, list) or not parsed_json:
        raise RuntimeError("Gemini returned zero candidate clips. Stopping execution.")

    validated_candidates: List[ViralShortCandidate] = []
    for item in parsed_json:
        try:
            cand = ViralShortCandidate(**item)

            # 8. Snap Boundaries with 0.25s Speech Padding (Anti-Speech Cut Safeguard)
            snapped_start, snapped_end = snap_to_word_boundaries(
                start_sec=cand.start_sec,
                end_sec=cand.end_sec,
                transcript_segments=segments,
                padding_sec=0.25,
                min_duration=float(min_sec),
                max_duration=float(max_sec),
                total_duration=derived_duration,
            )

            # Re-calibrate relative SFX timestamps if boundaries shifted
            start_shift = snapped_start - cand.start_sec
            snapped_duration = snapped_end - snapped_start
            sanitized_sfx: List[SfxEvent] = []

            for sfx in cand.sfx_events[:3]:
                # Adjust time relative to snapped clip start
                adj_time = max(0.1, min(snapped_duration - 0.2, sfx.time_sec - start_shift))
                sfx_type = sfx.type.lower().strip()
                if sfx_type not in ["whoosh", "pop", "boom"]:
                    sfx_type = "pop"
                sanitized_sfx.append(SfxEvent(time_sec=round(adj_time, 2), type=sfx_type))

            # Ensure hook banner is 4 to 7 words
            banner_words = cand.hook_banner.split()
            if len(banner_words) > 7:
                clean_banner = " ".join(banner_words[:7])
            elif len(banner_words) < 4:
                clean_banner = f"WATCH THIS: {cand.hook_banner}"
            else:
                clean_banner = cand.hook_banner

            final_cand = ViralShortCandidate(
                start_sec=snapped_start,
                end_sec=snapped_end,
                hook_banner=clean_banner,
                title=cand.title,
                viral_score=cand.viral_score,
                reason=cand.reason,
                sfx_events=sanitized_sfx,
            )
            validated_candidates.append(final_cand)

        except ValidationError as val_err:
            logger.warning("Skipping candidate failing schema validation: %s", val_err)

    if not validated_candidates:
        raise RuntimeError("No candidate clips satisfied the viral retention constraints. Aborting.")

    # Sort descending by viral score
    validated_candidates.sort(key=lambda c: c.viral_score, reverse=True)
    return validated_candidates[:max_clips]


# ==============================================================================
# 5. CLI Utility for Direct Terminal Testing
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Find viral short-form gaming moments via Gemini 2.5 Flash")
    parser.add_argument("--transcript", type=str, default="work/transcript.json", help="Path to transcript.json")
    parser.add_argument("--audio", type=str, default="work/audio_16k.wav", help="Path to audio WAV file")
    parser.add_argument("--max-clips", type=int, default=3, help="Max clips to return")
    parser.add_argument("--title", type=str, default="GTA 6 Gameplay", help="Video Title")
    args = parser.parse_args()

    try:
        results = find_viral_moments(
            transcript_data=args.transcript,
            audio_path=args.audio if Path(args.audio).exists() else None,
            video_title=args.title,
            max_clips=args.max_clips,
        )

        print("\n" + "=" * 80)
        print("VIRAL SHORT-FORM HIGHLIGHTS IDENTIFIED")
        print("=" * 80)
        for i, clip in enumerate(results, 1):
            dur = clip.end_sec - clip.start_sec
            print(f"\n[{i}] {clip.hook_banner} (Score: {clip.viral_score}/100)")
            print(f"    Timestamps:  {clip.start_sec}s -> {clip.end_sec}s (Duration: {dur:.1f}s)")
            print(f"    Title:       {clip.title}")
            print(f"    Reason:      {clip.reason}")
            print(f"    SFX Cues:    {', '.join([f'{s.type}@{s.time_sec}s' for s in clip.sfx_events]) or 'None'}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error("[ERROR] %s", e)
        sys.exit(1)
