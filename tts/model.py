"""
Local TTS model backends.

Primary (preferred): ai4bharat/indic-parler-tts
  - Realistic Hindi voices with description-based style control
  - Apache 2.0
  - Gated on Hugging Face (one-time accept + login / HF_TOKEN)

Fallback: facebook/mms-tts-hin
  - Ungated, small, reliable CPU Hindi TTS (VITS)
  - Single voice; no native style/emotion captions
  - CC-BY-NC 4.0 — fine for personal experiments; prefer Parler for commercial ads
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"
DEFAULT_MMS_MODEL_ID = "facebook/mms-tts-hin"

HINDI_VOICES_PARLER = [
    {"id": "Rohit", "label": "Rohit (recommended, male)", "recommended": True},
    {"id": "Divya", "label": "Divya (recommended, female)", "recommended": True},
    {"id": "Aman", "label": "Aman (male)", "recommended": False},
    {"id": "Rani", "label": "Rani (female)", "recommended": False},
]

HINDI_VOICES_MMS = [
    {"id": "MMS-Hindi", "label": "MMS Hindi (single voice)", "recommended": True},
]

STYLE_PRESETS = {
    "motivational": {
        "label": "Motivational",
        "direction": (
            "Speak in natural Indian Hindi. Use a confident, mature motivational narrator style. "
            "Maintain realistic human rhythm and pronunciation. Use natural pauses between important thoughts. "
            "Start calmly and gradually increase energy. Emphasize important words without sounding exaggerated. "
            "Avoid sounding like a newsreader, robot, or advertisement voice. "
            "Keep the delivery natural, emotionally engaging and believable."
        ),
        "caption_traits": (
            "confident and expressive motivational narration, warm mature tone, "
            "moderate pace with natural emphasis, slightly animated delivery"
        ),
    },
    "powerful": {
        "label": "Powerful Motivation",
        "direction": (
            "Deep, confident and powerful Hindi motivational narration. "
            "Start controlled and calm, then gradually increase intensity. "
            "Strong emphasis on important phrases. Natural pauses. "
            "Sound like an experienced motivational speaker, not a synthetic voice."
        ),
        "caption_traits": (
            "deep confident powerful delivery, slightly lower pitch, controlled then intense, "
            "expressive emphasis, moderate-to-slightly-slow pace"
        ),
    },
    "emotional": {
        "label": "Emotional",
        "direction": (
            "Natural and emotionally expressive Hindi narration. "
            "Speak warmly and sincerely. Use subtle emotional variation and meaningful pauses. "
            "Avoid overacting."
        ),
        "caption_traits": (
            "warm sincere emotionally expressive delivery, subtle dynamic variation, "
            "slightly slow pace with meaningful pauses, close intimate recording"
        ),
    },
    "calm_inspirational": {
        "label": "Calm Inspirational",
        "direction": (
            "Calm, warm and inspirational Hindi narration. "
            "Slow deliberate pacing, natural pauses and a reassuring human tone."
        ),
        "caption_traits": (
            "calm warm inspirational tone, slow deliberate pace, reassuring and gentle, "
            "slightly expressive, very clear audio"
        ),
    },
    "high_energy": {
        "label": "High Energy",
        "direction": (
            "Energetic and enthusiastic Hindi narration suitable for a short advertisement "
            "or social media video. Maintain clarity while increasing energy. "
            "Avoid shouting or sounding artificial."
        ),
        "caption_traits": (
            "energetic enthusiastic delivery, slightly faster pace, clear and animated, "
            "high energy without shouting, very clear audio"
        ),
    },
    "storytelling": {
        "label": "Storytelling",
        "direction": (
            "Natural Hindi storytelling narration. Warm, engaging and conversational. "
            "Vary pace gently between scenes, use soft pauses, and keep the voice human and believable."
        ),
        "caption_traits": (
            "warm storytelling narration, conversational and engaging, moderate pace, "
            "slightly expressive with natural pauses, clear close recording"
        ),
    },
    "professional_ad": {
        "label": "Professional Advertisement",
        "direction": (
            "Clear, polished Indian Hindi commercial narration. "
            "Confident and persuasive but natural. "
            "Excellent pronunciation, controlled pacing and subtle emphasis."
        ),
        "caption_traits": (
            "clear polished commercial narration, confident persuasive but natural, "
            "controlled moderate pace, excellent clarity, very clear audio"
        ),
    },
}


@dataclass
class GenerationResult:
    audio: np.ndarray
    sample_rate: int


def build_description(
    voice: str,
    style_key: str,
    custom_direction: Optional[str] = None,
    speed: float = 0.95,
) -> str:
    """Build a Parler-TTS description caption (official control surface)."""
    voice = voice if voice in {v["id"] for v in HINDI_VOICES_PARLER} else "Rohit"
    preset = STYLE_PRESETS.get(style_key, STYLE_PRESETS["motivational"])
    traits = preset["caption_traits"]

    if speed <= 0.85:
        rate_phrase = "a slow pace"
    elif speed <= 1.05:
        rate_phrase = "a moderate pace"
    else:
        rate_phrase = "a slightly fast pace"

    direction_note = (custom_direction or preset["direction"]).strip()
    return (
        f"{voice} speaks in natural Indian Hindi with {traits}. "
        f"The speaker delivers speech at {rate_phrase}. "
        f"The recording is of very high quality with very clear audio and no background noise, "
        f"with the speaker's voice sounding clear and very close up. "
        f"Style intent: {direction_note[:400]}"
    )


def _pick_device(device_preference: str) -> str:
    import torch

    pref = (device_preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but unavailable — falling back to CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class IndicParlerModel:
    """Wrapper around ai4bharat/indic-parler-tts (official API)."""

    backend_id = "parler"
    supports_style_captions = True

    def __init__(self, model_id: Optional[str] = None, device_preference: str = "auto"):
        self.model_id = model_id or os.environ.get("TTS_MODEL_ID", DEFAULT_PARLER_MODEL_ID)
        self.device_preference = (device_preference or os.environ.get("TTS_DEVICE", "auto")).lower()
        self.model = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None
        self.device = None
        self.sample_rate = 44100
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def list_voices(self) -> List[Dict[str, Any]]:
        return list(HINDI_VOICES_PARLER)

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoTokenizer

        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "parler-tts is not installed. Run: pip install git+https://github.com/huggingface/parler-tts.git"
            ) from exc

        device = _pick_device(self.device_preference)
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        logger.info("Loading Parler TTS %s on %s (%s)...", self.model_id, device, dtype)
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            ).to(device)
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
            self.sample_rate = int(getattr(self.model.config, "sampling_rate", 44100))
            self.model.eval()
            self._loaded = True
            logger.info("Parler model loaded. Sample rate=%s", self.sample_rate)
        except Exception as exc:
            message = str(exc).lower()
            if "gated" in message or "401" in message or "restricted" in message:
                raise RuntimeError(
                    "Indic Parler-TTS is a gated Hugging Face model. "
                    "Create a free HF account, accept access at "
                    "https://huggingface.co/ai4bharat/indic-parler-tts , then run "
                    "`huggingface-cli login` (or set HF_TOKEN). "
                    "Alternatively set TTS_BACKEND=mms to use the ungated MMS Hindi fallback."
                ) from exc
            if "out of memory" in message and device.startswith("cuda"):
                logger.error("GPU OOM while loading Parler: %s", exc)
                logger.info("Retrying Parler load on CPU...")
                self.device_preference = "cpu"
                self.model = None
                try:
                    import torch as _torch

                    _torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                return self.load()
            logger.exception("Failed to load Parler TTS")
            raise RuntimeError(f"Failed to load TTS model: {exc}") from exc

    def generate_chunk(self, text: str, description: str) -> GenerationResult:
        if not self._loaded:
            self.load()

        import torch

        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text chunk")

        try:
            description_inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
            prompt_inputs = self.prompt_tokenizer(text, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                generation = self.model.generate(
                    input_ids=description_inputs.input_ids,
                    attention_mask=description_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )

            audio = generation.detach().cpu().numpy().squeeze().astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            return GenerationResult(audio=audio, sample_rate=self.sample_rate)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise RuntimeError(
                    "Insufficient memory during generation. Try a shorter script or CPU mode."
                ) from exc
            logger.exception("Parler generation failed")
            raise RuntimeError("Speech generation failed. Check the terminal logs for details.") from exc


class MmsHindiModel:
    """
    Fallback: facebook/mms-tts-hin via official Transformers VitsModel API.
    Single speaker; style captions are ignored by the network (speed still via FFmpeg).
    """

    backend_id = "mms"
    supports_style_captions = False

    def __init__(self, model_id: Optional[str] = None, device_preference: str = "auto"):
        self.model_id = model_id or os.environ.get("TTS_MMS_MODEL_ID", DEFAULT_MMS_MODEL_ID)
        self.device_preference = (device_preference or os.environ.get("TTS_DEVICE", "auto")).lower()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.sample_rate = 16000
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def list_voices(self) -> List[Dict[str, Any]]:
        return list(HINDI_VOICES_MMS)

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoTokenizer, VitsModel

        device = _pick_device(self.device_preference)
        self.device = device

        logger.info("Loading MMS Hindi TTS %s on %s...", self.model_id, device)
        try:
            self.model = VitsModel.from_pretrained(self.model_id).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.sample_rate = int(getattr(self.model.config, "sampling_rate", 16000))
            self.model.eval()
            self._loaded = True
            logger.info("MMS model loaded. Sample rate=%s", self.sample_rate)
        except Exception as exc:
            logger.exception("Failed to load MMS TTS")
            raise RuntimeError(f"Failed to load MMS Hindi TTS model: {exc}") from exc

    def generate_chunk(self, text: str, description: str = "") -> GenerationResult:
        # description intentionally unused — MMS has no caption conditioning
        if not self._loaded:
            self.load()

        import torch

        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text chunk")

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs).waveform
            audio = output.detach().cpu().numpy().squeeze().astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            return GenerationResult(audio=audio, sample_rate=self.sample_rate)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise RuntimeError(
                    "Insufficient memory during generation. Try a shorter script."
                ) from exc
            logger.exception("MMS generation failed")
            raise RuntimeError("Speech generation failed. Check the terminal logs for details.") from exc


def create_model(device_preference: Optional[str] = None):
    """
    Select backend:
      TTS_BACKEND=parler|mms|auto  (default: auto)
    auto tries Parler; on gated/auth failure falls back to MMS.
    """
    backend = (os.environ.get("TTS_BACKEND", "auto") or "auto").lower().strip()
    pref = device_preference or os.environ.get("TTS_DEVICE", "auto")

    if backend == "mms":
        return MmsHindiModel(device_preference=pref)

    if backend == "parler":
        return IndicParlerModel(device_preference=pref)

    # auto
    parler = IndicParlerModel(device_preference=pref)
    try:
        parler.load()
        return parler
    except Exception as exc:
        logger.warning("Parler backend unavailable (%s). Falling back to MMS Hindi.", exc)
        mms = MmsHindiModel(device_preference=pref)
        mms.load()
        return mms


# Backwards-compatible alias used in docs/comments
HINDI_VOICES = HINDI_VOICES_PARLER
