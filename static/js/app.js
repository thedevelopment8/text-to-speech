(() => {
  const els = {
    scriptText: document.getElementById("scriptText"),
    charCount: document.getElementById("charCount"),
    wordCount: document.getElementById("wordCount"),
    estDuration: document.getElementById("estDuration"),
    voiceSelect: document.getElementById("voiceSelect"),
    styleSelect: document.getElementById("styleSelect"),
    speedRange: document.getElementById("speedRange"),
    speedValue: document.getElementById("speedValue"),
    speechDirection: document.getElementById("speechDirection"),
    advancedToggle: document.getElementById("advancedToggle"),
    advancedPanel: document.getElementById("advancedPanel"),
    generateBtn: document.getElementById("generateBtn"),
    testBtn: document.getElementById("testBtn"),
    errorBox: document.getElementById("errorBox"),
    loadingBox: document.getElementById("loadingBox"),
    loadingText: document.getElementById("loadingText"),
    resultPanel: document.getElementById("resultPanel"),
    audioPlayer: document.getElementById("audioPlayer"),
    resultMeta: document.getElementById("resultMeta"),
    downloadMp3: document.getElementById("downloadMp3"),
    downloadWav: document.getElementById("downloadWav"),
    systemStatus: document.getElementById("systemStatus"),
    musicUploadWrap: document.getElementById("musicUploadWrap"),
    musicFile: document.getElementById("musicFile"),
    musicStatus: document.getElementById("musicStatus"),
    voiceOnly: document.getElementById("voiceOnly"),
  };

  const state = {
    styles: {},
    musicId: null,
    busy: false,
  };

  function formatBytes(n) {
    if (!n && n !== 0) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  }

  function formatDuration(sec) {
    if (!sec && sec !== 0) return "—";
    const s = Math.max(0, Math.round(sec));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m > 0 ? `${m}m ${r}s` : `${r}s`;
  }

  function countWords(text) {
    const t = text.trim();
    if (!t) return 0;
    return t.split(/\s+/).filter(Boolean).length;
  }

  function updateTextStats() {
    const text = els.scriptText.value;
    const chars = text.length;
    const words = countWords(text);
    // ~2.4 Hindi words/sec for motivational pacing estimate
    const est = words > 0 ? Math.max(1, Math.round(words / 2.4)) : 0;
    els.charCount.textContent = `${chars} character${chars === 1 ? "" : "s"}`;
    els.wordCount.textContent = `${words} word${words === 1 ? "" : "s"}`;
    els.estDuration.textContent = `~${est}s estimated`;
  }

  function showError(msg) {
    els.errorBox.hidden = false;
    els.errorBox.textContent = msg || "Something went wrong.";
  }

  function clearError() {
    els.errorBox.hidden = true;
    els.errorBox.textContent = "";
  }

  function setBusy(busy, message) {
    state.busy = busy;
    els.generateBtn.disabled = busy;
    els.testBtn.disabled = busy;
    els.loadingBox.hidden = !busy;
    if (message) els.loadingText.textContent = message;
  }

  function musicMode() {
    const checked = document.querySelector('input[name="musicMode"]:checked');
    return checked ? checked.value : "none";
  }

  function syncMusicUI() {
    const mode = musicMode();
    els.musicUploadWrap.hidden = mode !== "upload";
    if (mode === "none") {
      els.voiceOnly.checked = true;
    }
  }

  async function loadSystem() {
    try {
      const res = await fetch("/api/system");
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Failed to load system info");
      const s = data.system;
      let label = s.status_label || "System: CPU Mode";
      if (s.backend) label += ` · ${s.backend}`;
      if (!s.ffmpeg_ok) label += " · FFmpeg missing";
      els.systemStatus.textContent = label;
      els.systemStatus.classList.toggle("ok", !!s.ffmpeg_ok);
    } catch (err) {
      els.systemStatus.textContent = "System: status unavailable";
    }
  }

  async function loadVoices() {
    const res = await fetch("/api/voices");
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Failed to load voices");

    els.voiceSelect.innerHTML = "";
    (data.voices || []).forEach((v, idx) => {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.label;
      if (v.recommended || idx === 0) opt.selected = true;
      els.voiceSelect.appendChild(opt);
    });

    els.styleSelect.innerHTML = "";
    state.styles = {};
    (data.styles || []).forEach((s) => {
      state.styles[s.id] = s;
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.label;
      if (s.id === "motivational") opt.selected = true;
      els.styleSelect.appendChild(opt);
    });

    applyStyleDirection();

    if (data.supports_style_captions === false) {
      const note = document.createElement("p");
      note.className = "hint";
      note.id = "styleSupportHint";
      note.textContent =
        "Current backend (MMS Hindi) has a single voice and does not support style captions. Speed still applies via FFmpeg. For Rohit/Divya + style control, enable Indic Parler-TTS (HF login).";
      const styleControl = els.styleSelect.closest(".control");
      if (styleControl && !document.getElementById("styleSupportHint")) {
        styleControl.appendChild(note);
      }
    }
  }

  function applyStyleDirection() {
    const style = state.styles[els.styleSelect.value];
    if (style && style.direction) {
      els.speechDirection.value = style.direction;
    }
  }

  function showResult(data) {
    els.resultPanel.hidden = false;
    const src = data.mp3_url || data.wav_url;
    els.audioPlayer.src = `${src}?t=${Date.now()}`;
    els.audioPlayer.load();
    els.downloadMp3.href = data.download_mp3;
    els.downloadWav.href = data.download_wav;
    els.resultMeta.innerHTML = `
      <span>Duration: ${formatDuration(data.duration_sec)}</span>
      <span>MP3: ${formatBytes(data.mp3_size)}</span>
      <span>WAV: ${formatBytes(data.wav_size)}</span>
      <span>Generated in ${data.generation_time_sec}s</span>
      <span>Voice: ${data.voice}</span>
      <span>Chunks: ${data.chunks}</span>
      <span>Backend: ${data.backend || "—"}</span>
      ${data.mixed ? "<span>Mixed with music</span>" : "<span>Voice only</span>"}
    `;
    els.resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function uploadMusicIfNeeded() {
    if (musicMode() !== "upload" || els.voiceOnly.checked) {
      return null;
    }
    if (state.musicId) return state.musicId;
    const file = els.musicFile.files && els.musicFile.files[0];
    if (!file) {
      throw new Error("Please upload a background music file, or choose None / Voice Only.");
    }
    const form = new FormData();
    form.append("music", file);
    const res = await fetch("/api/upload-music", { method: "POST", body: form });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Music upload failed");
    state.musicId = data.music_id;
    els.musicStatus.textContent = `Uploaded: ${data.filename}`;
    return state.musicId;
  }

  async function generate(isTest) {
    clearError();
    if (state.busy) return;

    const payload = {
      voice: els.voiceSelect.value,
      style: els.styleSelect.value,
      speed: parseFloat(els.speedRange.value),
      direction: els.speechDirection.value.trim(),
      voice_only: els.voiceOnly.checked || musicMode() === "none",
    };

    if (!isTest) {
      payload.text = els.scriptText.value.trim();
      if (!payload.text) {
        showError("Please paste Hindi text before generating.");
        return;
      }
    }

    setBusy(
      true,
      isTest
        ? "Generating short test voice locally…"
        : "Generating speech locally… first run may download the model; CPU can take several minutes."
    );

    try {
      if (!isTest && !payload.voice_only) {
        payload.music_id = await uploadMusicIfNeeded();
      }

      const endpoint = isTest ? "/api/test-voice" : "/api/generate";
      const controller = new AbortController();
      // Long timeout for CPU generation
      const timeout = setTimeout(() => controller.abort(), 45 * 60 * 1000);

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      showResult(data);
    } catch (err) {
      if (err.name === "AbortError") {
        showError("Generation timed out. Try a shorter script or use a GPU.");
      } else {
        showError(err.message || "Generation failed.");
      }
    } finally {
      setBusy(false);
    }
  }

  // Events
  els.scriptText.addEventListener("input", updateTextStats);
  els.speedRange.addEventListener("input", () => {
    els.speedValue.textContent = `${parseFloat(els.speedRange.value).toFixed(2)}x`;
  });
  els.styleSelect.addEventListener("change", applyStyleDirection);
  els.advancedToggle.addEventListener("click", () => {
    const open = els.advancedPanel.hidden;
    els.advancedPanel.hidden = !open;
    els.advancedToggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.querySelectorAll('input[name="musicMode"]').forEach((el) => {
    el.addEventListener("change", syncMusicUI);
  });
  els.musicFile.addEventListener("change", () => {
    state.musicId = null;
    const file = els.musicFile.files && els.musicFile.files[0];
    els.musicStatus.textContent = file
      ? `Selected: ${file.name} (will upload on generate)`
      : "Voice stays dominant; music is ducked with fade in/out.";
  });
  els.voiceOnly.addEventListener("change", () => {
    if (!els.voiceOnly.checked && musicMode() === "none") {
      const upload = document.querySelector('input[name="musicMode"][value="upload"]');
      if (upload) upload.checked = true;
      syncMusicUI();
    }
  });
  els.generateBtn.addEventListener("click", () => generate(false));
  els.testBtn.addEventListener("click", () => generate(true));

  // Init
  updateTextStats();
  syncMusicUI();
  Promise.all([loadSystem(), loadVoices()]).catch((err) => {
    showError(err.message || "Failed to initialize UI.");
  });
})();
