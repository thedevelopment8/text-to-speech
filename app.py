"""
Hindi Motivational Voice Generator — local Flask app.

Run:
    python app.py

Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from tts.engine import TTSEngine
from tts.audio import FFmpegError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated"
UPLOADS_DIR = BASE_DIR / "uploads"
ALLOWED_MUSIC_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hindi_tts")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

max_mb = int(os.environ.get("MAX_UPLOAD_MB", "25"))
app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

engine = TTSEngine(generated_dir=GENERATED_DIR, uploads_dir=UPLOADS_DIR)

# Safe filename pattern for downloads (timestamp_voice.ext)
SAFE_AUDIO_NAME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6,12}_voice\.(mp3|wav)$")


def _user_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _result_payload(result) -> dict:
    return {
        "ok": True,
        "job_id": result.job_id,
        "wav_url": f"/api/audio/{result.wav_path.name}",
        "mp3_url": f"/api/audio/{result.mp3_path.name}",
        "download_mp3": f"/api/download/{result.mp3_path.name}",
        "download_wav": f"/api/download/{result.wav_path.name}",
        "duration_sec": round(result.duration_sec, 2),
        "wav_size": result.wav_size,
        "mp3_size": result.mp3_size,
        "generation_time_sec": result.generation_time_sec,
        "voice": result.voice,
        "style": result.style,
        "chunks": result.chunks,
        "mixed": result.mixed,
        "backend": result.backend,
    }


@app.errorhandler(413)
def too_large(_err):
    return _user_error(f"Upload too large. Maximum size is {max_mb} MB.", 413)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/system")
def api_system():
    info = engine.refresh_system_info()
    return jsonify({"ok": True, "system": info})


@app.route("/api/voices")
def api_voices():
    try:
        engine.ensure_model()
    except Exception as exc:  # noqa: BLE001
        logger.error("Model not ready for /api/voices: %s", exc)
        return _user_error(str(exc), 500)
    return jsonify(
        {
            "ok": True,
            "voices": engine.list_voices(),
            "styles": engine.list_styles(),
            "backend": engine.backend_id,
            "supports_style_captions": engine.supports_style_captions,
            "test_sentence": "हर दिन एक नई शुरुआत है। खुद पर विश्वास रखिए और आगे बढ़ते रहिए।",
        }
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Synchronous generation for v1.
    Structured so a background job queue can wrap this later.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        voice = (data.get("voice") or "Rohit").strip()
        style = (data.get("style") or "motivational").strip()
        direction = data.get("direction")
        voice_only = bool(data.get("voice_only", True))
        music_id = (data.get("music_id") or "").strip()

        try:
            speed = float(data.get("speed", 0.95))
        except (TypeError, ValueError):
            speed = 0.95

        if not text:
            return _user_error("Please paste Hindi text before generating.")

        music_path = None
        if music_id and not voice_only:
            music_path = _resolve_upload(music_id)
            if music_path is None:
                return _user_error("Uploaded music file not found. Please upload again.")

        result = engine.generate(
            text=text,
            voice=voice,
            style=style,
            speed=speed,
            direction=direction,
            music_path=music_path,
            voice_only=voice_only,
        )

        return jsonify(_result_payload(result))
    except ValueError as exc:
        return _user_error(str(exc), 400)
    except FFmpegError as exc:
        logger.error("FFmpeg error: %s", exc)
        return _user_error(str(exc), 500)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return _user_error(str(exc), 500)
    except Exception:
        logger.exception("Unexpected generation failure")
        return _user_error("Speech generation failed. Check the terminal for details.", 500)


@app.route("/api/test-voice", methods=["POST"])
def api_test_voice():
    try:
        data = request.get_json(silent=True) or {}
        voice = (data.get("voice") or "Rohit").strip()
        style = (data.get("style") or "motivational").strip()
        direction = data.get("direction")
        try:
            speed = float(data.get("speed", 0.95))
        except (TypeError, ValueError):
            speed = 0.95

        result = engine.generate_test(
            voice=voice,
            style=style,
            speed=speed,
            direction=direction,
        )
        payload = _result_payload(result)
        payload["preview"] = True
        return jsonify(payload)
    except ValueError as exc:
        return _user_error(str(exc), 400)
    except FFmpegError as exc:
        return _user_error(str(exc), 500)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return _user_error(str(exc), 500)
    except Exception:
        logger.exception("Test voice failed")
        return _user_error("Test voice generation failed. Check the terminal for details.", 500)


@app.route("/api/upload-music", methods=["POST"])
def api_upload_music():
    if "music" not in request.files:
        return _user_error("No music file provided.")
    file = request.files["music"]
    if not file or not file.filename:
        return _user_error("No music file selected.")

    original = secure_filename(file.filename)
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_MUSIC_EXT:
        return _user_error(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_MUSIC_EXT))}"
        )

    music_id = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / music_id
    file.save(dest)
    logger.info("Saved uploaded music as %s (from %s)", music_id, original)
    return jsonify(
        {
            "ok": True,
            "music_id": music_id,
            "filename": original,
        }
    )


@app.route("/api/audio/<path:filename>")
def api_audio(filename: str):
    name = Path(filename).name
    if not SAFE_AUDIO_NAME.match(name):
        return _user_error("Invalid audio filename.", 400)
    path = GENERATED_DIR / name
    if not path.exists():
        return _user_error("Audio file not found.", 404)
    return send_from_directory(GENERATED_DIR, name, as_attachment=False)


@app.route("/api/download/<path:filename>")
def api_download(filename: str):
    name = Path(filename).name
    if not SAFE_AUDIO_NAME.match(name):
        return _user_error("Invalid audio filename.", 400)
    path = GENERATED_DIR / name
    if not path.exists():
        return _user_error("Audio file not found.", 404)
    return send_from_directory(GENERATED_DIR, name, as_attachment=True)


def _resolve_upload(music_id: str) -> Path | None:
    name = Path(music_id).name
    if name != music_id:
        return None
    if ".." in name or "/" in name or "\\" in name:
        return None
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_MUSIC_EXT:
        return None
    # UUID hex + ext
    stem = Path(name).stem
    if not re.fullmatch(r"[0-9a-f]{32}", stem):
        return None
    path = UPLOADS_DIR / name
    if not path.exists() or not path.is_file():
        return None
    # Ensure resolved path stays inside uploads
    try:
        path.resolve().relative_to(UPLOADS_DIR.resolve())
    except ValueError:
        return None
    return path


def _preload_model():
    """Load model once at startup (or skip gracefully if deps missing)."""
    preload = os.environ.get("TTS_PRELOAD", "1") != "0"
    if not preload:
        logger.info("TTS_PRELOAD=0 — model will load on first generation.")
        return
    try:
        logger.info("Preloading TTS model...")
        engine.load_model()
        engine.refresh_system_info()
        logger.info("TTS model ready.")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Model preload failed (%s). The UI will still start; "
            "generation will retry loading on demand.",
            exc,
        )


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    sys_info = engine.get_system_info()
    logger.info("%s", sys_info.get("status_label", "System: unknown"))
    if not sys_info.get("ffmpeg_ok"):
        logger.warning("FFmpeg not detected — install it before generating audio.")

    # Avoid double-preload with Flask reloader
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_DEBUG") != "1":
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not debug:
            _preload_model()

    print("\n  Hindi Motivational Voice Generator")
    print(f"  Open: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, threaded=True)
