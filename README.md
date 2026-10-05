# Hindi Motivational Voice Generator

Local Hindi text-to-speech web app for motivational / advertisement voiceovers.
Runs entirely on your machine — **no cloud TTS APIs**.

**Preferred model:** [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) (Indic Parler-TTS)  
**Fallback model:** [facebook/mms-tts-hin](https://huggingface.co/facebook/mms-tts-hin) (Meta MMS Hindi)  
**Stack:** Python 3.11+, Flask, PyTorch, Hugging Face Transformers, FFmpeg

---

## Features

- Paste long Hindi (Devanagari) scripts and generate realistic speech locally
- Hindi voices from the Parler model card: **Rohit, Divya, Aman, Rani**
- Style presets that map to Parler-TTS description captions (pace, expressivity, clarity)
- Editable speech direction (Advanced)
- Speed control **0.75x – 1.25x** (caption bias + FFmpeg `atempo`)
- Optional background music with ducking + fade in/out
- WAV master + high-quality MP3 download
- Automatic GPU (CUDA) use when available, CPU fallback otherwise
- Intelligent chunking for long scripts
- Automatic fallback to ungated MMS Hindi if Parler access is unavailable

---

## Requirements

| Component | Notes |
|-----------|--------|
| Python | **3.11 or 3.12 recommended** (3.13+ may work; avoid very new versions if wheels are missing) |
| FFmpeg | Required for MP3 conversion, speed, concat, and music mix |
| RAM | ~8 GB minimum on CPU; **16 GB+ recommended** (Parler is larger than MMS) |
| GPU (optional) | NVIDIA GPU + CUDA for much faster Parler generation |
| Disk | Several GB for Python packages + Hugging Face model cache |
| HF account | Required once for gated Indic Parler-TTS (free) |

---

## 1. Install FFmpeg

### Windows

1. Download a static build from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) (or [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html))
2. Extract and add the `bin` folder to your **PATH**
3. Open a **new** Command Prompt / PowerShell and verify:

```bat
ffmpeg -version
```

### macOS

```bash
brew install ffmpeg
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

---

## 2. Create a virtual environment

From the project root:

```bash
python -m venv venv
```

### Windows

```bat
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## 3. Install PyTorch

Install the build that matches your hardware **before** (or with) the rest of the requirements.

### CPU only (Windows / Linux)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### NVIDIA CUDA 12.1 (Windows / Linux)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check other CUDA versions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Verify CUDA (optional)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Install project dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, Transformers, and the official **parler-tts** package from GitHub:

```text
pip install git+https://github.com/huggingface/parler-tts.git
```

If the Git install fails on Windows, install Git for Windows first, then retry.

---

## 5. Model download / Hugging Face access

### Preferred: Indic Parler-TTS (gated, best quality + styles)

`ai4bharat/indic-parler-tts` requires a free Hugging Face account and one-time access approval:

1. Join: [https://huggingface.co/join](https://huggingface.co/join)
2. Open [https://huggingface.co/ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) and accept the terms
3. Create a token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Log in:

```bash
huggingface-cli login
```

Or put the token in `.env`:

```text
HF_TOKEN=hf_your_token_here
TTS_BACKEND=parler
```

Cache locations:

- Windows: `C:\Users\<you>\.cache\huggingface\`
- Linux/macOS: `~/.cache\huggingface/`

### Fallback: MMS Hindi (ungated)

If Parler is unavailable, set:

```bat
set TTS_BACKEND=mms
python app.py
```

| Backend | Voices | Style captions | License |
|---------|--------|----------------|---------|
| Parler (preferred) | Rohit, Divya, Aman, Rani | Yes | Apache 2.0 |
| MMS fallback | Single Hindi voice | No (FFmpeg speed only) | CC-BY-NC 4.0 — prefer Parler for commercial ads |

Default `TTS_BACKEND=auto` tries Parler, then falls back to MMS.

---

## 6. Run the app

```bash
python app.py
```

Open:

**http://127.0.0.1:5000**

The UI status line shows either:

- `System: NVIDIA GPU / CUDA Available`, or
- `System: CPU Mode`

plus the active backend (`parler` or `mms`).

---

## Usage tips

1. Paste Hindi text (Devanagari). Do not transliterate.
2. Pick a voice (Rohit / Divya recommended for Hindi on Parler).
3. Pick a style preset, or edit **Advanced → Speech Direction**.
4. Click **Test Voice** for a short preview before long ads.
5. Click **Generate Voice**, then play / download MP3 or WAV.
6. Optional: upload background music and uncheck **Voice Only**.

### How style control works

Indic Parler-TTS does **not** expose separate emotion knobs. It controls delivery via a **description caption** (speaking rate, expressivity, pitch, clarity, named speaker). This app turns your style preset + direction into that caption — the official API.

On the MMS fallback, style dropdowns are informational only; speed still applies via FFmpeg.

### How to change voice / style later

- **UI:** Voice and Style dropdowns, or edit Advanced speech direction.
- **Code:** `tts/model.py` → `HINDI_VOICES_PARLER`, `STYLE_PRESETS`, and `build_description()`.
- **Swap models:** keep the Flask app the same; replace logic inside `tts/`.

---

## Project layout

```text
.
├── app.py                 # Flask entrypoint
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── tts/
│   ├── engine.py          # TTSEngine abstraction
│   ├── model.py           # Parler + MMS backends
│   └── audio.py           # FFmpeg + chunking
├── templates/index.html
├── static/css/style.css
├── static/js/app.js
├── generated/             # output WAV/MP3
├── uploads/               # temp music uploads
└── models/                # optional local weights
```

---

## Troubleshooting

### Gated repo / 401 / must be authenticated

Accept the model terms on Hugging Face, then `huggingface-cli login`, or use `TTS_BACKEND=mms`.

### `FFmpeg is not installed`

Install FFmpeg and ensure `ffmpeg` is on PATH. Restart the terminal and the app.

### `parler-tts is not installed`

```bash
pip install git+https://github.com/huggingface/parler-tts.git
```

### CUDA unavailable / falls back to CPU

Normal if you have no NVIDIA GPU or drivers/CUDA mismatch. Generation still works, but is **much slower**. Force CPU with:

```bat
set TTS_DEVICE=cpu
python app.py
```

PowerShell:

```powershell
$env:TTS_DEVICE="cpu"
python app.py
```

### Out of memory (GPU or RAM)

- Close other apps
- Use shorter chunks (`TTS_MAX_CHUNK_CHARS=200` in `.env`)
- Prefer GPU with enough VRAM, or a machine with more RAM
- Or use the smaller MMS backend: `TTS_BACKEND=mms`

### First run is very slow

Parler (~0.9B params) downloads once, then loads into memory. On CPU, a short paragraph can take **several minutes**. MMS is much faster/lighter.

### Garbled / robotic speech

- Prefer recommended voices **Rohit** or **Divya** for Hindi (Parler)
- Keep punctuation (use `।` and commas for natural pauses)
- Avoid extremely long single chunks — the app auto-splits

### `pip` / Python version errors

Use Python 3.11 or 3.12. Recreate the venv if you switched interpreters.

---

## Security notes (local personal use)

- Uploads are limited in size and extension
- Downloads only allow generated `*_voice.mp3` / `*_voice.wav` names
- FFmpeg is invoked with argument lists (`shell=False`)
- No accounts, analytics, or external TTS APIs

---

## License

Application code in this repo is for your personal local use.  
Indic Parler-TTS / Parler-TTS: **Apache 2.0**.  
MMS-TTS Hindi: **CC-BY-NC 4.0** (non-commercial). Prefer Parler for commercial advertisement use.
