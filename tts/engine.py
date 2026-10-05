"""
High-level TTS engine abstraction.

The Flask app should only talk to TTSEngine — not to a specific model backend.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tts import audio as audio_utils
from tts.model import STYLE_PRESETS, build_description, create_model

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "generated"
UPLOADS_DIR = BASE_DIR / "uploads"
TEST_SENTENCE = "हर दिन एक नई शुरुआत है। खुद पर विश्वास रखिए और आगे बढ़ते रहिए।"


@dataclass
class GenerateOutput:
    job_id: str
    wav_path: Path
    mp3_path: Path
    duration_sec: float
    wav_size: int
    mp3_size: int
    generation_time_sec: float
    voice: str
    style: str
    chunks: int
    mixed: bool
    backend: str


class TTSEngine:
    def __init__(self, generated_dir: Optional[Path] = None, uploads_dir: Optional[Path] = None):
        self.generated_dir = Path(generated_dir or GENERATED_DIR)
        self.uploads_dir = Path(uploads_dir or UPLOADS_DIR)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._system_info: Optional[Dict[str, Any]] = None
        self.max_chunk_chars = int(os.environ.get("TTS_MAX_CHUNK_CHARS", "280"))
        self.max_text_chars = int(os.environ.get("TTS_MAX_TEXT_CHARS", "8000"))

    def load_model(self) -> None:
        if self.model is not None and getattr(self.model, "is_loaded", False):
            return
        self.model = create_model()
        if not self.model.is_loaded:
            self.model.load()
        self._system_info = None

    def ensure_model(self) -> None:
        if self.model is None or not self.model.is_loaded:
            self.load_model()

    @property
    def backend_id(self) -> str:
        if self.model is None:
            return (os.environ.get("TTS_BACKEND", "auto") or "auto").lower()
        return getattr(self.model, "backend_id", "unknown")

    @property
    def supports_style_captions(self) -> bool:
        if self.model is None:
            backend = (os.environ.get("TTS_BACKEND", "auto") or "auto").lower()
            return backend != "mms"
        return bool(getattr(self.model, "supports_style_captions", False))

    def get_system_info(self) -> Dict[str, Any]:
        if self._system_info is not None:
            info = dict(self._system_info)
            info["model_loaded"] = bool(self.model and self.model.is_loaded)
            info["backend"] = self.backend_id
            info["supports_style_captions"] = self.supports_style_captions
            if self.model is not None:
                info["model_id"] = getattr(self.model, "model_id", None)
            return info

        info: Dict[str, Any] = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "ram_gb": None,
            "cuda_available": False,
            "gpu_name": None,
            "vram_gb": None,
            "device_mode": "CPU Mode",
            "status_label": "System: CPU Mode",
            "ffmpeg_ok": False,
            "ffmpeg_version": None,
            "model_id": None,
            "model_loaded": False,
            "backend": self.backend_id,
            "supports_style_captions": self.supports_style_captions,
        }

        try:
            if platform.system() == "Linux" and Path("/proc/meminfo").exists():
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024**2), 1)
                        break
        except Exception:  # noqa: BLE001
            pass

        try:
            import torch

            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                try:
                    props = torch.cuda.get_device_properties(0)
                    info["vram_gb"] = round(props.total_memory / (1024**3), 1)
                except Exception:  # noqa: BLE001
                    pass
                info["device_mode"] = "NVIDIA GPU / CUDA Available"
                if info["gpu_name"]:
                    info["status_label"] = f"System: {info['gpu_name']} / CUDA Available"
                else:
                    info["status_label"] = "System: NVIDIA GPU / CUDA Available"
            else:
                info["device_mode"] = "CPU Mode"
                info["status_label"] = "System: CPU Mode"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not probe torch/CUDA: %s", exc)
            info["status_label"] = "System: CPU Mode (PyTorch not loaded yet)"

        ok, ver = audio_utils.check_ffmpeg()
        info["ffmpeg_ok"] = ok
        info["ffmpeg_version"] = ver

        if self.model is not None and self.model.is_loaded:
            info["model_loaded"] = True
            info["model_id"] = getattr(self.model, "model_id", None)
            info["backend"] = self.backend_id
            info["supports_style_captions"] = self.supports_style_captions
            device = str(getattr(self.model, "device", "") or "")
            if device.startswith("cuda"):
                info["status_label"] = info.get("status_label") or "System: NVIDIA GPU / CUDA Available"
            else:
                info["status_label"] = "System: CPU Mode"

        self._system_info = info
        return self.get_system_info()

    def refresh_system_info(self) -> Dict[str, Any]:
        self._system_info = None
        return self.get_system_info()

    def list_voices(self) -> List[Dict[str, Any]]:
        self.ensure_model()
        return self.model.list_voices()

    def list_styles(self) -> List[Dict[str, str]]:
        styles = [
            {"id": key, "label": value["label"], "direction": value["direction"]}
            for key, value in STYLE_PRESETS.items()
        ]
        return styles

    def get_style_direction(self, style_key: str) -> str:
        preset = STYLE_PRESETS.get(style_key, STYLE_PRESETS["motivational"])
        return preset["direction"]

    def generate(
        self,
        text: str,
        voice: str = "Rohit",
        style: str = "motivational",
        speed: float = 0.95,
        direction: Optional[str] = None,
        music_path: Optional[Path] = None,
        voice_only: bool = True,
    ) -> GenerateOutput:
        text = (text or "").strip()
        if not text:
            raise ValueError("Please enter Hindi text to generate speech.")
        if len(text) > self.max_text_chars:
            raise ValueError(
                f"Text is too long ({len(text)} characters). "
                f"Please keep it under {self.max_text_chars} characters."
            )

        speed = max(0.75, min(1.25, float(speed or 0.95)))
        if style not in STYLE_PRESETS:
            style = "motivational"

        audio_utils.ensure_ffmpeg()
        self.ensure_model()

        voices = {v["id"] for v in self.model.list_voices()}
        if voice not in voices:
            voice = next(iter(voices))

        description = ""
        if self.supports_style_captions:
            description = build_description(
                voice=voice if voice in {v["id"] for v in self.model.list_voices()} else "Rohit",
                style_key=style,
                custom_direction=direction,
                speed=speed,
            )
        else:
            logger.info(
                "Backend %s ignores style captions; applying speed via FFmpeg only.",
                self.backend_id,
            )

        chunks = audio_utils.split_text_into_chunks(text, max_chars=self.max_chunk_chars)
        if not chunks:
            raise ValueError("Please enter Hindi text to generate speech.")

        job_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # Avoid collisions if two jobs start in the same second
        if (self.generated_dir / f"{job_id}_voice.wav").exists():
            job_id = datetime.now().strftime("%Y-%m-%d_%H%M%S%f")[:21]

        work_dir = self.generated_dir / f"_tmp_{job_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        chunk_wavs: List[Path] = []

        try:
            logger.info(
                "Generating %s chunk(s) | backend=%s voice=%s style=%s speed=%.2f",
                len(chunks),
                self.backend_id,
                voice,
                style,
                speed,
            )
            for idx, chunk in enumerate(chunks):
                logger.info("Chunk %s/%s (%s chars)", idx + 1, len(chunks), len(chunk))
                result = self.model.generate_chunk(chunk, description)
                chunk_path = work_dir / f"chunk_{idx:03d}.wav"
                audio_utils.write_wav(chunk_path, result.audio, result.sample_rate)
                chunk_wavs.append(chunk_path)

            combined = work_dir / "combined.wav"
            audio_utils.concat_wavs(chunk_wavs, combined, pause_ms=280)

            sped = work_dir / "sped.wav"
            audio_utils.change_speed(combined, sped, speed)

            normalized = work_dir / "normalized.wav"
            try:
                audio_utils.normalize_loudness(sped, normalized)
                master_src = normalized
            except Exception as exc:  # noqa: BLE001
                logger.warning("Loudness normalization skipped: %s", exc)
                master_src = sped

            use_music = bool(music_path) and not voice_only
            if use_music:
                if not music_path or not Path(music_path).exists():
                    raise ValueError("Background music file was not found.")
                mixed = work_dir / "mixed.wav"
                audio_utils.mix_voice_with_music(master_src, Path(music_path), mixed)
                final_src = mixed
            else:
                final_src = master_src

            wav_out = self.generated_dir / f"{job_id}_voice.wav"
            mp3_out = self.generated_dir / f"{job_id}_voice.mp3"

            audio_utils.resample_wav(final_src, wav_out, sample_rate=44100)
            audio_utils.wav_to_mp3(wav_out, mp3_out, bitrate="192k")

            elapsed = time.perf_counter() - started
            return GenerateOutput(
                job_id=job_id,
                wav_path=wav_out,
                mp3_path=mp3_out,
                duration_sec=audio_utils.audio_duration_seconds(wav_out),
                wav_size=audio_utils.file_size_bytes(wav_out),
                mp3_size=audio_utils.file_size_bytes(mp3_out),
                generation_time_sec=round(elapsed, 2),
                voice=voice,
                style=style,
                chunks=len(chunks),
                mixed=use_music,
                backend=self.backend_id,
            )
        finally:
            try:
                for p in work_dir.glob("*"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                work_dir.rmdir()
            except OSError:
                logger.debug("Could not fully clean temp dir %s", work_dir)

    def generate_test(
        self,
        voice: str = "Rohit",
        style: str = "motivational",
        speed: float = 0.95,
        direction: Optional[str] = None,
    ) -> GenerateOutput:
        return self.generate(
            text=TEST_SENTENCE,
            voice=voice,
            style=style,
            speed=speed,
            direction=direction,
            music_path=None,
            voice_only=True,
        )
