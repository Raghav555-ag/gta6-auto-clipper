"""Studio-Grade Audio Mixing and Dynamic Ducking Module (audio_mixer.py).

Implements professional short-form audio engineering using FFmpeg filter graphs:
1. Automatic BGM Ducking: Uses FFmpeg sidechaincompress triggered by creator voice
   so music lowers (-20dB to -24dB) during speech and gently rises in natural pauses.
2. Timed Subtle SFX Layering: Places micro SFX (whoosh, pop, boom) at low volume
   (-14dB to -18dB) via adelay and amix. Automatically places a soft opening whoosh.
3. EBU R128 Loudness Normalization: Normalizes final audio with loudnorm (I=-14, TP=-1.5, LRA=11)
   to match YouTube Shorts standards without digital clipping or distortion.
4. Asset Setup & Synthetic Fallbacks: Auto-generates clean synthetic SFX/BGM fallbacks
   if assets directories are empty, logging clear instructions for custom audio.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audio_mixer")

DEFAULT_ASSETS_DIR = Path("./assets")
DEFAULT_BGM_DIR = DEFAULT_ASSETS_DIR / "bgm"
DEFAULT_SFX_DIR = DEFAULT_ASSETS_DIR / "sfx"


# ==============================================================================
# 1. FFmpeg Binary Discovery
# ==============================================================================

def get_ffmpeg_binary() -> str:
    """Locate functional FFmpeg executable."""
    env_ffmpeg = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG_BINARY")
    if env_ffmpeg and Path(env_ffmpeg).is_file():
        return str(Path(env_ffmpeg).resolve())

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    win_paths = [
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    ]
    for p in win_paths:
        if p.is_file():
            return str(p.resolve())

    raise RuntimeError("FFmpeg executable not found. Please ensure FFmpeg is on PATH.")


# ==============================================================================
# 2. Synthetic Audio Synthesis & Asset Initialization
# ==============================================================================

def write_pcm_wav(
    filepath: Union[str, Path],
    duration_sec: float,
    sample_rate: int,
    generator_func,
) -> Path:
    """Generate a clean 16-bit mono PCM WAV file from a mathematical waveform generator."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        data = bytearray()
        for i in range(n_samples):
            t = i / float(sample_rate)
            raw_val = generator_func(t, duration_sec)
            val = max(-1.0, min(1.0, raw_val))
            sample_int = int(val * 32767.0)
            data.extend(struct.pack("<h", sample_int))

        wf.writeframes(data)

    return path


def generate_synthetic_whoosh(path: Path) -> Path:
    """Generate a smooth 0.4s whoosh frequency sweep with Hann envelope."""
    def whoosh_gen(t: float, d: float) -> float:
        env = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / d))
        # Frequency rises from 250Hz to 850Hz and curves back down
        freq = 250.0 + 600.0 * math.sin(math.pi * t / d)
        tone = math.sin(2.0 * math.pi * freq * t)
        # Subtle harmonic layer
        sub = 0.3 * math.sin(4.0 * math.pi * freq * t)
        return env * 0.7 * (tone + sub)

    return write_pcm_wav(path, duration_sec=0.40, sample_rate=44100, generator_func=whoosh_gen)


def generate_synthetic_pop(path: Path) -> Path:
    """Generate a sharp 0.07s UI pop / bubble impact."""
    def pop_gen(t: float, d: float) -> float:
        env = math.exp(-40.0 * t / d)
        freq = 1350.0 - 900.0 * (t / d)
        return env * 0.85 * math.sin(2.0 * math.pi * freq * t)

    return write_pcm_wav(path, duration_sec=0.07, sample_rate=44100, generator_func=pop_gen)


def generate_synthetic_boom(path: Path) -> Path:
    """Generate a deep 0.85s sub-bass drop / cinematic impact."""
    def boom_gen(t: float, d: float) -> float:
        env = math.exp(-4.5 * t / d)
        # Deep sub-bass glide from 95Hz down to 36Hz
        freq = 95.0 - 59.0 * (t / d)
        sub = math.sin(2.0 * math.pi * freq * t)
        harmonic = 0.25 * math.sin(4.0 * math.pi * freq * t)
        return env * 0.90 * (sub + harmonic)

    return write_pcm_wav(path, duration_sec=0.85, sample_rate=44100, generator_func=boom_gen)


def generate_synthetic_bgm_loop(path: Path) -> Path:
    """Generate a soft, unobtrusive 6-second ambient chord loop (C minor pentatonic)."""
    duration_sec = 6.0
    sr = 44100
    if np is not None:
        t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False, dtype=np.float32)
        env = np.clip(t / 0.5, 0.0, 1.0) * np.clip((duration_sec - t) / 0.5, 0.0, 1.0)
        chord = (
            np.sin(2.0 * np.pi * 130.81 * t)
            + np.sin(2.0 * np.pi * 155.56 * t)
            + np.sin(2.0 * np.pi * 196.00 * t)
            + np.sin(2.0 * np.pi * 233.08 * t)
        )
        lfo = 0.7 + 0.3 * np.sin(2.0 * np.pi * 0.5 * t)
        signal = env * 0.22 * lfo * chord
        samples = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        return path

    def bgm_gen(t: float, d: float) -> float:
        loop_env = min(1.0, t / 0.5) * min(1.0, (d - t) / 0.5)
        chord_c = math.sin(2.0 * math.pi * 130.81 * t)
        chord_eb = math.sin(2.0 * math.pi * 155.56 * t)
        chord_g = math.sin(2.0 * math.pi * 196.00 * t)
        chord_bb = math.sin(2.0 * math.pi * 233.08 * t)
        lfo = 0.7 + 0.3 * math.sin(2.0 * math.pi * 0.5 * t)
        return loop_env * 0.25 * lfo * (chord_c + chord_eb + chord_g + chord_bb)

    return write_pcm_wav(path, duration_sec=6.0, sample_rate=44100, generator_func=bgm_gen)


def ensure_assets(
    assets_dir: Union[str, Path] = DEFAULT_ASSETS_DIR,
) -> Dict[str, Any]:
    """Ensure assets directories exist and synthesize pristine fallback audio if empty.

    Args:
        assets_dir: Root directory for assets.

    Returns:
        Dict[str, Any]: Summary of available BGM and SFX assets.
    """
    base = Path(assets_dir).resolve()
    bgm_dir = base / "bgm"
    sfx_dir = base / "sfx"

    bgm_dir.mkdir(parents=True, exist_ok=True)
    sfx_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup SFX Fallbacks
    sfx_files = {
        "whoosh": sfx_dir / "whoosh.wav",
        "pop": sfx_dir / "pop.wav",
        "boom": sfx_dir / "boom.wav",
    }

    generated_sfx = []
    if not sfx_files["whoosh"].exists():
        generate_synthetic_whoosh(sfx_files["whoosh"])
        generated_sfx.append("whoosh.wav")
    if not sfx_files["pop"].exists():
        generate_synthetic_pop(sfx_files["pop"])
        generated_sfx.append("pop.wav")
    if not sfx_files["boom"].exists():
        generate_synthetic_boom(sfx_files["boom"])
        generated_sfx.append("boom.wav")

    if generated_sfx:
        logger.info("Generated synthetic fallback SFX in %s: %s", sfx_dir.name, ", ".join(generated_sfx))

    # SFX README
    sfx_readme = sfx_dir / "README.md"
    if not sfx_readme.exists():
        sfx_readme.write_text(
            "# Sound Effects (SFX) Directory\n\n"
            "Drop your custom royalty-free sound effects here:\n"
            "- `whoosh.wav` or `whoosh.mp3`: Subtle camera whoosh for transitions & hook banner\n"
            "- `pop.wav` or `pop.mp3`: Quick pop for visual callouts & text pop-ins\n"
            "- `boom.wav` or `boom.mp3`: Cinematic bass drop for punchline/climax impact\n\n"
            "If any file is missing, a clean synthetic fallback is automatically generated.\n",
            encoding="utf-8",
        )

    # 2. Setup BGM Fallbacks
    bgm_candidates = list(bgm_dir.glob("*.mp3")) + list(bgm_dir.glob("*.wav")) + list(bgm_dir.glob("*.m4a"))
    if not bgm_candidates:
        default_loop = bgm_dir / "ambient_loop.wav"
        generate_synthetic_bgm_loop(default_loop)
        logger.info("Generated synthetic ambient background music loop: %s", default_loop.name)

    # BGM README
    bgm_readme = bgm_dir / "README.md"
    if not bgm_readme.exists():
        bgm_readme.write_text(
            "# Background Music (BGM) Directory\n\n"
            "Drop your royalty-free background music tracks (.mp3, .wav, .m4a) here.\n"
            "The audio mixer will automatically pick a track, loop/trim it to your clip length,\n"
            "and apply studio-grade auto-ducking (`sidechaincompress`) so the music gently dips\n"
            "whenever the creator speaks.\n",
            encoding="utf-8",
        )

    return {
        "bgm_dir": str(bgm_dir),
        "sfx_dir": str(sfx_dir),
        "sfx_available": {k: str(v) for k, v in sfx_files.items() if v.exists()},
        "bgm_tracks": [str(p) for p in (list(bgm_dir.glob("*.mp3")) + list(bgm_dir.glob("*.wav")))],
    }


def get_bgm_track(bgm_dir: Path, preferred_track: Optional[str] = None) -> Optional[Path]:
    """Retrieve an active background music file from bgm_dir."""
    if preferred_track:
        pref = Path(preferred_track).resolve()
        if pref.is_file():
            return pref
        direct_in_dir = bgm_dir / preferred_track
        if direct_in_dir.is_file():
            return direct_in_dir

    tracks = list(bgm_dir.glob("*.mp3")) + list(bgm_dir.glob("*.wav")) + list(bgm_dir.glob("*.m4a"))
    if not tracks:
        return None
    # Deterministic or random selection
    return tracks[0]


def get_sfx_path(sfx_dir: Path, sfx_type: str) -> Optional[Path]:
    """Locate sound effect file for a given type ('whoosh', 'pop', 'boom')."""
    clean_type = sfx_type.lower().strip()
    for ext in [".wav", ".mp3", ".aac", ".ogg"]:
        candidate = sfx_dir / f"{clean_type}{ext}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


# ==============================================================================
# 3. Filtergraph Assembly: Sidechain Ducking, Delay & Loudnorm
# ==============================================================================

@dataclass
class SfxCue:
    time_sec: float
    type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sanitize_sfx_cues(
    raw_cues: Optional[List[Union[Dict[str, Any], SfxCue]]],
    clip_duration: float,
    add_opening_whoosh: bool = True,
) -> List[SfxCue]:
    """Sanitize and clamp SFX cues, adding an opening hook banner whoosh at 0.0s."""
    sanitized: List[SfxCue] = []

    # 1. Soft opening whoosh for the 0-3s hook banner
    if add_opening_whoosh:
        sanitized.append(SfxCue(time_sec=0.05, type="whoosh"))

    if raw_cues:
        for cue in raw_cues:
            if isinstance(cue, SfxCue):
                t, cue_type = cue.time_sec, cue.type
            elif isinstance(cue, dict):
                t = float(cue.get("time_sec", 0.0))
                cue_type = str(cue.get("type", "pop"))
            else:
                continue

            # Filter cues beyond clip duration or invalid times
            if 0.0 <= t <= max(0.5, clip_duration - 0.2):
                clean_type = cue_type.lower().strip()
                if clean_type not in ["whoosh", "pop", "boom"]:
                    clean_type = "pop"
                # Avoid overlapping exactly on the 0.05s opening whoosh
                if abs(t - 0.05) > 0.15:
                    sanitized.append(SfxCue(time_sec=round(t, 2), type=clean_type))

    # Keep maximum 4 cues per clip to prevent noisy over-editing
    sanitized.sort(key=lambda c: c.time_sec)
    return sanitized[:4]


def build_audio_filtergraph(
    has_bgm: bool,
    sfx_files: List[Tuple[Path, float]],
    bgm_volume_db: float = -22.0,
    sfx_volume_db: float = -16.0,
) -> Tuple[List[str], str]:
    """Construct FFmpeg filtergraph for sidechain ducking, adelay SFX layering, and EBU loudnorm.

    VULNERABILITY PREVENTION:
    - Never clips audio: uses normalize=0 on amix with explicit volume attenuation.
    - Finishes with alimiter and loudnorm to guarantee YouTube Shorts compliance.

    Args:
        has_bgm: Whether BGM input is attached (Input index 1).
        sfx_files: List of (path, delay_sec) for sound effects (Input indices 2..N).
        bgm_volume_db: Base volume reduction for BGM (default: -22dB).
        sfx_volume_db: Base volume for SFX (default: -16dB).

    Returns:
        Tuple[List[str], str]: List of extra FFmpeg input args and filter_complex string.
    """
    filter_chains = []
    streams_to_mix = []

    # Stream 0: Primary Voice from Video
    # Split voice into main audio and sidechain control signal
    filter_chains.append("[0:a]aformat=channel_layouts=stereo:sample_rates=44100[voice_raw]")
    if has_bgm:
        filter_chains.append("[voice_raw]asplit=2[voice_main][voice_sidechain]")
    else:
        filter_chains.append("[voice_raw]anull[voice_main]")

    streams_to_mix.append("[voice_main]")

    # Stream 1: Background Music (Duckable via sidechaincompress)
    if has_bgm:
        bgm_vol_ratio = 10.0 ** (bgm_volume_db / 20.0)
        filter_chains.append(
            f"[1:a]aformat=channel_layouts=stereo:sample_rates=44100,volume={bgm_vol_ratio:.4f}[bgm_low]"
        )
        # Sidechain Compression: dips BGM whenever voice talks (threshold 0.03, ratio 4:1)
        filter_chains.append(
            "[bgm_low][voice_sidechain]sidechaincompress="
            "threshold=0.03:ratio=4:attack=50:release=350[bgm_ducked]"
        )
        streams_to_mix.append("[bgm_ducked]")

    # Streams 2..N: Sound Effects with adelay
    sfx_base_index = 2 if has_bgm else 1
    sfx_vol_ratio = 10.0 ** (sfx_volume_db / 20.0)

    for idx, (path, delay_sec) in enumerate(sfx_files):
        stream_idx = sfx_base_index + idx
        delay_ms = max(0, int(round(delay_sec * 1000.0)))
        filter_chains.append(
            f"[{stream_idx}:a]aformat=channel_layouts=stereo:sample_rates=44100,"
            f"volume={sfx_vol_ratio:.4f},adelay={delay_ms}|{delay_ms}[sfx_{idx}]"
        )
        streams_to_mix.append(f"[sfx_{idx}]")

    # Combine all streams with amix:
    # normalize=0 prevents automatic downward gain scaling that muffles commentary
    mix_inputs_str = "".join(streams_to_mix)
    n_inputs = len(streams_to_mix)
    filter_chains.append(
        f"{mix_inputs_str}amix=inputs={n_inputs}:duration=first:dropout_transition=2:normalize=0[mixed]"
    )

    # Master Limiter & YouTube Shorts Loudness Normalization:
    # I=-14 LUFS, True Peak -1.5dB, Loudness Range 11
    filter_chains.append(
        "[mixed]alimiter=limit=0.95:attack=5:release=50,"
        "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
    )

    filter_complex = ";\n".join(filter_chains)
    return filter_complex


# ==============================================================================
# 4. Master Audio Mixer Pipeline
# ==============================================================================

def mix_clip_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    sfx_cues: Optional[List[Union[Dict[str, Any], SfxCue]]] = None,
    bgm_path: Optional[Union[str, Path]] = None,
    assets_dir: Union[str, Path] = DEFAULT_ASSETS_DIR,
    bgm_volume_db: float = -22.0,
    sfx_volume_db: float = -16.0,
    add_opening_whoosh: bool = True,
    ffmpeg_bin: Optional[str] = None,
) -> Path:
    """Mix professional background music and sound effects into a video clip.

    Args:
        video_path: Path to rendered video clip (MP4).
        output_path: Destination path for audio-mixed video.
        sfx_cues: List of SFX event dicts or SfxCue objects.
        bgm_path: Optional explicit BGM audio file path.
        assets_dir: Directory containing bgm/ and sfx/ folders.
        bgm_volume_db: Volume for BGM in dB (default: -22dB).
        sfx_volume_db: Volume for SFX in dB (default: -16dB).
        add_opening_whoosh: Whether to trigger whoosh at 0.05s for hook banner.
        ffmpeg_bin: Optional FFmpeg binary override.

    Returns:
        Path: Path to final mixed video.
    """
    v_in = Path(video_path).resolve()
    v_out = Path(output_path).resolve()
    v_out.parent.mkdir(parents=True, exist_ok=True)

    if not v_in.is_file():
        raise FileNotFoundError(f"Input video file not found: {v_in}")

    bin_path = ffmpeg_bin or get_ffmpeg_binary()

    # 1. Initialize & Ensure Assets
    assets_info = ensure_assets(assets_dir)
    bgm_dir = Path(assets_info["bgm_dir"])
    sfx_dir = Path(assets_info["sfx_dir"])

    # 2. Determine Clip Duration via FFmpeg probe
    probe_cmd = [bin_path, "-hide_banner", "-i", str(v_in)]
    probe_proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe_proc.stderr)
    clip_duration = 30.0
    if dur_match:
        h, m, s = dur_match.groups()
        clip_duration = int(h) * 3600 + int(m) * 60 + float(s)

    # 3. Resolve BGM Track
    active_bgm = None
    if bgm_path:
        b_p = Path(bgm_path).resolve()
        if b_p.is_file():
            active_bgm = b_p
    if not active_bgm:
        active_bgm = get_bgm_track(bgm_dir)

    # 4. Resolve and Sanitize SFX Cues
    sanitized_cues = sanitize_sfx_cues(
        raw_cues=sfx_cues,
        clip_duration=clip_duration,
        add_opening_whoosh=add_opening_whoosh,
    )

    resolved_sfx: List[Tuple[Path, float]] = []
    for cue in sanitized_cues:
        s_path = get_sfx_path(sfx_dir, cue.type)
        if s_path and s_path.is_file():
            resolved_sfx.append((s_path, cue.time_sec))
        else:
            logger.warning("SFX file for type '%s' not found. Skipping cue at %.2fs.", cue.type, cue.time_sec)

    has_bgm = active_bgm is not None and active_bgm.is_file()
    if has_bgm:
        logger.info("Using BGM track: %s (auto-ducking enabled)", active_bgm.name)
    else:
        logger.info("No BGM track found. Continuing with voice & SFX.")

    logger.info("Layering %d subtle SFX cues: %s", len(resolved_sfx), [f"{p.stem}@{t}s" for p, t in resolved_sfx])

    # 5. Build FFmpeg Command
    cmd = [bin_path, "-y", "-i", str(v_in)]

    # Add BGM input (looped seamlessly to clip duration)
    if has_bgm:
        cmd.extend(["-stream_loop", "-1", "-i", str(active_bgm)])

    # Add SFX inputs
    for s_path, _ in resolved_sfx:
        cmd.extend(["-i", str(s_path)])

    # Assemble filter complex
    filter_complex = build_audio_filtergraph(
        has_bgm=has_bgm,
        sfx_files=resolved_sfx,
        bgm_volume_db=bgm_volume_db,
        sfx_volume_db=sfx_volume_db,
    )

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",  # Zero re-encoding for sharp video frames!
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(v_out),
    ])

    logger.info("Executing audio mixing & loudness normalization for %s -> %s...", v_in.name, v_out.name)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        logger.error("FFmpeg audio mixing failed:\n%s", proc.stderr)
        # Fallback safeguard: copy original video so pipeline does not crash
        logger.warning("Falling back to original unmixed video due to FFmpeg audio error.")
        if v_in != v_out:
            shutil.copy2(v_in, v_out)
        return v_out

    logger.info(
        "Successfully mixed audio with ducked BGM and normalized loudness: %s (%.2f MB)",
        v_out.name,
        v_out.stat().st_size / (1024 * 1024),
    )
    return v_out


# ==============================================================================
# 5. CLI Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Studio-grade audio mixing with BGM sidechain ducking and SFX layering"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video clip path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output mixed video path")
    parser.add_argument("--sfx-json", type=str, default=None, help="JSON file or string of sfx_events")
    parser.add_argument("--bgm", type=str, default=None, help="Custom BGM audio track")
    parser.add_argument("--assets-dir", type=str, default="./assets", help="Assets folder with bgm/ and sfx/")
    parser.add_argument("--bgm-vol", type=float, default=-22.0, help="BGM volume in dB (default: -22)")
    parser.add_argument("--sfx-vol", type=float, default=-16.0, help="SFX volume in dB (default: -16)")
    parser.add_argument("--no-opening-whoosh", action="store_true", help="Disable automatic opening whoosh")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cues_list = None
    if args.sfx_json:
        p = Path(args.sfx_json)
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cues_list = json.load(f)
            except Exception as e:
                logger.warning("Could not read --sfx-json file: %s", e)
        else:
            try:
                cues_list = json.loads(args.sfx_json)
            except Exception:
                pass

    try:
        mix_clip_audio(
            video_path=args.input,
            output_path=args.output,
            sfx_cues=cues_list,
            bgm_path=args.bgm,
            assets_dir=args.assets_dir,
            bgm_volume_db=args.bgm_vol,
            sfx_volume_db=args.sfx_vol,
            add_opening_whoosh=not args.no_opening_whoosh,
        )
    except Exception as exc:
        logger.error("[FATAL ERROR] Audio mixing failed: %s", exc)
        sys.exit(1)
