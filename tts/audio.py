"""
Audio processing helpers using FFmpeg and soundfile.

All FFmpeg calls use argument arrays (never shell=True).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError(
            "FFmpeg is not installed or not on PATH. "
            "Install FFmpeg and restart the app. See README for Windows/Linux instructions."
        )
    return path


def check_ffmpeg() -> Tuple[bool, str]:
    path = shutil.which("ffmpeg")
    if not path:
        return False, "FFmpeg not found on PATH"
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        first = (result.stdout or result.stderr or "").splitlines()[:1]
        return True, first[0] if first else path
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sr)


def _run_ffmpeg(args: Sequence[str], label: str = "ffmpeg") -> None:
    ffmpeg = ensure_ffmpeg()
    cmd = [ffmpeg, "-y", *args]
    logger.debug("Running %s: %s", label, " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        logger.error("%s failed (%s): %s", label, result.returncode, err[-2000:])
        raise FFmpegError(f"Audio processing failed ({label}). See terminal logs.")


def concat_wavs(wav_paths: Sequence[Path], output_path: Path, pause_ms: int = 280) -> None:
    """Concatenate chunk WAVs with a short natural pause between chunks."""
    if not wav_paths:
        raise ValueError("No wav paths to concatenate")
    if len(wav_paths) == 1:
        shutil.copyfile(wav_paths[0], output_path)
        return

    ensure_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="tts_concat_") as tmp:
        tmp_dir = Path(tmp)
        list_file = tmp_dir / "list.txt"
        silence = tmp_dir / "silence.wav"

        # Match first file sample rate / channels
        info = sf.info(str(wav_paths[0]))
        _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={info.samplerate}:cl=mono",
                "-t",
                f"{pause_ms / 1000.0:.3f}",
                str(silence),
            ],
            label="silence",
        )

        lines: List[str] = []
        for i, wav in enumerate(wav_paths):
            # FFmpeg concat demuxer needs escaped paths; use absolute posix-style
            lines.append(f"file '{wav.resolve().as_posix()}'")
            if i < len(wav_paths) - 1:
                lines.append(f"file '{silence.resolve().as_posix()}'")
        list_file.write_text("\n".join(lines), encoding="utf-8")

        _run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(output_path),
            ],
            label="concat",
        )


def change_speed(input_wav: Path, output_wav: Path, speed: float) -> None:
    """
    Change playback speed without large pitch distortion using FFmpeg atempo.
    atempo accepts 0.5–2.0; we clamp to the UI range 0.75–1.25.
    """
    speed = max(0.75, min(1.25, float(speed)))
    if abs(speed - 1.0) < 0.01:
        shutil.copyfile(input_wav, output_wav)
        return

    # Invert: speed 0.95 means slightly slower speech → atempo=0.95
    _run_ffmpeg(
        [
            "-i",
            str(input_wav),
            "-filter:a",
            f"atempo={speed:.4f}",
            str(output_wav),
        ],
        label="speed",
    )


def normalize_loudness(input_wav: Path, output_wav: Path) -> None:
    """Gentle loudness normalization — preserves dynamics better than hard limiting."""
    _run_ffmpeg(
        [
            "-i",
            str(input_wav),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(output_wav),
        ],
        label="loudnorm",
    )


def wav_to_mp3(input_wav: Path, output_mp3: Path, bitrate: str = "192k") -> None:
    _run_ffmpeg(
        [
            "-i",
            str(input_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-ar",
            "44100",
            str(output_mp3),
        ],
        label="mp3",
    )


def resample_wav(input_wav: Path, output_wav: Path, sample_rate: int = 44100) -> None:
    _run_ffmpeg(
        [
            "-i",
            str(input_wav),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output_wav),
        ],
        label="resample",
    )


def mix_voice_with_music(
    voice_wav: Path,
    music_path: Path,
    output_wav: Path,
    music_volume: float = 0.18,
    fade_in: float = 1.2,
    fade_out: float = 1.5,
) -> None:
    """
    Duck background music under narration.
    Does not modify the original music file.
    """
    ensure_ffmpeg()
    music_volume = max(0.05, min(0.4, float(music_volume)))

    # Duration of voice track
    info = sf.info(str(voice_wav))
    duration = float(info.duration)

    filter_complex = (
        f"[1:a]volume={music_volume:.3f},afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={max(0.0, duration - fade_out):.2f}:d={fade_out:.2f}[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[out]"
    )

    _run_ffmpeg(
        [
            "-i",
            str(voice_wav),
            "-i",
            str(music_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            "44100",
            str(output_wav),
        ],
        label="mix",
    )


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def audio_duration_seconds(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return 0.0


def split_text_into_chunks(text: str, max_chars: int = 280) -> List[str]:
    """
    Intelligent chunking for long Hindi scripts.
    Prefer: paragraphs → sentences (Hindi/English punctuation) → words.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    units: List[str] = []
    for para in paragraphs:
        units.extend(_split_sentences(para))

    chunks: List[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_by_words(unit, max_chars))
            continue
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_sentences(text: str) -> List[str]:
    # Hindi danda ।, Devanagari abbreviations, and common punctuation
    separators = set("।!?۔.\n")
    sentences: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in separators:
            piece = "".join(buf).strip()
            if piece:
                sentences.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def _split_by_words(text: str, max_chars: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(word) > max_chars:
                # Hard split extremely long tokens
                for i in range(0, len(word), max_chars):
                    chunks.append(word[i : i + max_chars])
                current = ""
            else:
                current = word
    if current:
        chunks.append(current)
    return chunks
