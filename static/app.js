const state = {
  defaults: {},
  history: [],
  currentRun: null,
  currentStem: null,
  isProcessing: false,
  audioElement: null,
};

const selectors = {
  status: document.getElementById("status"),
  form: document.getElementById("process-form"),
  audioFile: document.getElementById("audio-file"),
  audioPath: document.getElementById("audio-path"),
  audioLink: document.getElementById("audio-link"),
  audioPlayer: document.getElementById("audio-player"),
  audioFilename: document.getElementById("audio-filename"),
  speakerEditor: document.getElementById("speaker-editor"),
  outputSummary: document.getElementById("output-summary"),
  outputTranscript: document.getElementById("output-transcript"),
  outputTopics: document.getElementById("output-topics"),
  outputSummaries: document.getElementById("output-summaries"),
  outputDiarization: document.getElementById("output-diarization"),
  historyList: document.getElementById("history-list"),
  refreshHistory: document.getElementById("refresh-history"),
};

function setStatus(message, isError = false) {
  if (!selectors.status) {
    return;
  }
  selectors.status.textContent = message || "\u00a0";
  selectors.status.classList.toggle("error", isError);
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function ensureBlock(container, title, bodyHtml, options = {}) {
  if (!container) {
    return;
  }
  const { size = "md", theme = "default" } = options;
  if (!bodyHtml) {
    container.classList.remove("active");
    container.removeAttribute("data-size");
    container.removeAttribute("data-theme");
    container.innerHTML = "";
    return;
  }
  container.classList.add("active");
  container.dataset.size = size;
  if (theme && theme !== "default") {
    container.dataset.theme = theme;
  } else {
    container.removeAttribute("data-theme");
  }
  container.innerHTML = `
    <div class="block-header">
      <h3>${title}</h3>
    </div>
    <div class="block-body">${bodyHtml}</div>
  `;
}

function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "";
  }
  const minutes = Math.floor(value / 60);
  const seconds = (value - minutes * 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${seconds}`;
}

function renderTimestampButton(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  const display = formatSeconds(value);
  return `<button type="button" class="timestamp" data-time="${value}">${escapeHtml(display)}</button>`;
}

function applyDefaults() {
  const { defaults } = state;
  document.querySelectorAll("[data-config-key]").forEach((element) => {
    const key = element.dataset.configKey;
    if (!(key in defaults)) {
      return;
    }
    const def = defaults[key];
    if (def === null || typeof def === "undefined") {
      if (element.type === "checkbox") {
        element.checked = false;
      } else {
        element.value = "";
      }
      return;
    }
    if (element.type === "checkbox") {
      element.checked = Boolean(def);
      return;
    }
    if (Array.isArray(def)) {
      element.value = def.join(",");
      return;
    }
    element.value = def;
  });
}

async function loadDefaults() {
  const response = await fetch("/api/defaults");
  if (!response.ok) {
    throw new Error("Unable to load defaults.");
  }
  state.defaults = await response.json();
  applyDefaults();
}

function collectConfig() {
  const config = {};
  document.querySelectorAll("[data-config-key]").forEach((element) => {
    const key = element.dataset.configKey;
    if (!key) {
      return;
    }
    if (element.type === "checkbox") {
      config[key] = element.checked;
      return;
    }
    const value = element.value.trim();
    if (value.length === 0) {
      config[key] = "";
      return;
    }
    config[key] = value;
  });

  const audioLink = selectors.audioLink?.value.trim();
  if (audioLink) {
    config.audio_link = audioLink;
  }
  return config;
}

async function submitForm(event) {
  event.preventDefault();
  if (state.isProcessing) {
    return;
  }
  setStatus("Processing audio. This may take a while...");
  state.isProcessing = true;

  const formData = new FormData();
  const config = collectConfig();
  const audioPathValue = selectors.audioPath?.value.trim();
  if (audioPathValue) {
    formData.append("audio_path", audioPathValue);
  }

  const file = selectors.audioFile?.files?.[0];
  if (file) {
    formData.append("audio_file", file);
  }

  formData.append("config_json", JSON.stringify(config));

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Processing failed.");
    }
    const result = await response.json();
    renderResult(result);
    await fetchHistory();
    setStatus("Processing complete.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Processing failed.", true);
  } finally {
    state.isProcessing = false;
  }
}

function renderFileLinks(files) {
  if (!files || typeof files !== "object") {
    return "";
  }
  const entries = Object.entries(files);
  if (!entries.length) {
    return "<p>No output files recorded.</p>";
  }
  const links = entries
    .map(([label, meta]) => {
      const url = meta.url || "";
      const filename = meta.filename || label;
      const linkHtml = url
        ? `<a href="${url}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`
        : `<span>${escapeHtml(label)}</span>`;
      return `<li>${linkHtml} <span class="note">${escapeHtml(filename)}</span></li>`;
    })
    .join("");
  return `<ul>${links}</ul>`;
}

function setAudioSource(info) {
  const audio = selectors.audioPlayer;
  const label = selectors.audioFilename;
  if (!audio || !label) {
    return;
  }
  if (!info || !info.url) {
    audio.removeAttribute("src");
    audio.removeAttribute("data-src");
    audio.load();
    label.textContent = "Audio unavailable for playback.";
    return;
  }
  if (audio.getAttribute("data-src") !== info.url) {
    audio.setAttribute("src", info.url);
    audio.setAttribute("data-src", info.url);
    audio.load();
  }
  label.textContent = info.filename || "Audio file";
}

function renderResult(result) {
  if (!result || typeof result !== "object") {
    return;
  }
  const { audio_stem: stem, mode, created_at: createdAt, files, data, config, source_audio: sourceAudio } = result;

  state.currentRun = result;
  state.currentStem = stem || null;

  setAudioSource(sourceAudio);

  const created = createdAt ? new Date(createdAt).toLocaleString() : "unknown";
  const configSummary = config
    ? Object.entries(config)
        .filter(([key]) => key !== "audio_path" && key !== "output_dir")
        .map(([key, value]) => `<li><strong>${escapeHtml(key)}</strong>: ${escapeHtml(String(value ?? ""))}</li>`)
        .join("")
    : "";

  ensureBlock(
    selectors.outputSummary,
    "Run Summary",
    `<p><strong>Audio:</strong> ${escapeHtml(stem || "unknown")}</p>
     <p><strong>Mode:</strong> ${escapeHtml(mode || "all")} &middot; <strong>Created:</strong> ${escapeHtml(created)}</p>
     <details>
       <summary>Configuration</summary>
       <ul>${configSummary}</ul>
     </details>
     <details open>
       <summary>Download outputs</summary>
       ${renderFileLinks(files)}
     </details>`,
    { size: "md" }
  );

  renderTranscript(data?.transcript_text, data?.transcript_segments);
  renderTopicSegments(data?.topic_segments);
  renderSummaries(data?.topic_summaries);
  renderDiarization(data?.diarization_segments);
  renderSpeakerEditor(result);
}

function renderTranscript(transcriptText, segments) {
  if (Array.isArray(segments) && segments.length) {
    const rows = segments
      .map((segment, index) => {
        const start = renderTimestampButton(segment.start);
        const end = renderTimestampButton(segment.end);
        const speaker = segment.speaker ? escapeHtml(segment.speaker) : "—";
        const text = escapeHtml(segment.text || "");
        return `<tr>
          <td>${index + 1}</td>
          <td>${start}</td>
          <td>${end}</td>
          <td>${speaker}</td>
          <td>${text}</td>
        </tr>`;
      })
      .join("");
    ensureBlock(
      selectors.outputTranscript,
      `Transcript (${segments.length} segments)`,
      `<div class="table-wrapper">
         <table class="output-table">
           <thead>
             <tr>
               <th>#</th>
               <th>Start</th>
               <th>End</th>
               <th>Speaker</th>
               <th>Text</th>
             </tr>
           </thead>
           <tbody>${rows}</tbody>
        </table>
       </div>`,
      { size: "xl" }
    );
    return;
  }

  if (transcriptText) {
    const lines = transcriptText.split(/\r?\n/).filter(Boolean).length;
    ensureBlock(
      selectors.outputTranscript,
      "Transcript",
      `<details open>
       <summary>${lines} lines</summary>
       <pre>${escapeHtml(transcriptText)}</pre>
     </details>`,
      { size: "xl" }
    );
    return;
  }

  ensureBlock(selectors.outputTranscript, "Transcript", "<p>No transcript available.</p>", { size: "xl" });
}

function renderTopicSegments(segments) {
  if (!Array.isArray(segments) || segments.length === 0) {
    ensureBlock(selectors.outputTopics, "Topic Segments", "<p>No topic segments detected.</p>", { size: "xl" });
    return;
  }
  const rows = segments
    .map((segment, index) => {
      const start = renderTimestampButton(segment.start);
      const end = renderTimestampButton(segment.end);
      const preview = (segment.text || "").slice(0, 160);
      const speaker = segment.primary_speaker ? escapeHtml(segment.primary_speaker) : "—";
      return `<tr>
        <td>${index + 1}</td>
        <td>${start}</td>
        <td>${end}</td>
        <td>${speaker}</td>
        <td>${escapeHtml(preview)}${segment.text && segment.text.length > 160 ? "…" : ""}</td>
      </tr>`;
    })
    .join("");

  ensureBlock(
    selectors.outputTopics,
    `Topic Segments (${segments.length})`,
    `<div class="table-wrapper">
       <table class="output-table">
         <thead>
           <tr>
             <th>#</th>
             <th>Start</th>
             <th>End</th>
             <th>Primary Speaker</th>
             <th>Preview</th>
           </tr>
         </thead>
         <tbody>${rows}</tbody>
        </table>
     </div>`,
    { size: "xl" }
  );
}

function renderSummaries(summaries) {
  if (!Array.isArray(summaries) || summaries.length === 0) {
    ensureBlock(selectors.outputSummaries, "Summaries", "<p>No summaries generated.</p>", {
      size: "md",
      theme: "accent",
    });
    return;
  }
  const cards = summaries
    .map((summary, index) => {
      const start = renderTimestampButton(summary.start);
      const end = renderTimestampButton(summary.end);
      const primary = summary.primary_speaker ? ` (${escapeHtml(summary.primary_speaker)})` : "";
      const speakerSummaries = summary.speaker_summaries
        ? Object.entries(summary.speaker_summaries)
            .filter(([, text]) => text)
            .map(
              ([speaker, text]) =>
                `<li><strong>${escapeHtml(speaker)}:</strong> ${escapeHtml(String(text))}</li>`
            )
            .join("")
        : "";
      return `<article>
        <h4>${index + 1}. ${start} → ${end}${primary}</h4>
        <p>${escapeHtml(summary.summary || "")}</p>
        ${
          speakerSummaries
            ? `<details>
                 <summary>Speaker details</summary>
                 <ul>${speakerSummaries}</ul>
               </details>`
            : ""
        }
      </article>`;
    })
    .join("");

  ensureBlock(
    selectors.outputSummaries,
    `Summaries (${summaries.length})`,
    cards,
    { size: "md", theme: "accent" }
  );
}

function renderDiarization(segments) {
  if (!Array.isArray(segments) || segments.length === 0) {
    ensureBlock(selectors.outputDiarization, "Diarization", "<p>No diarization segments detected.</p>", {
      size: "md",
    });
    ensureBlock(selectors.outputDiarization, "Diarization", "<p>No diarization segments detected.</p>", {
      size: "md",
    });
    return;
  }
  const rows = segments
    .map((segment, index) => {
      const start = renderTimestampButton(segment.start);
      const end = renderTimestampButton(segment.end);
      const speaker = segment.speaker ? escapeHtml(segment.speaker) : "Unknown";
      return `<tr>
        <td>${index + 1}</td>
        <td>${start}</td>
        <td>${end}</td>
        <td>${speaker}</td>
      </tr>`;
    })
    .join("");

  ensureBlock(
    selectors.outputDiarization,
    `Diarization Segments (${segments.length})`,
    `<div class="table-wrapper">
        <table class="output-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Start</th>
              <th>End</th>
              <th>Speaker</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
     </div>`,
    { size: "md" }
  );
}

function collectSpeakers(data) {
  const speakers = new Set();
  if (!data) {
    return speakers;
  }
  const push = (value) => {
    if (typeof value === "string" && value.trim()) {
      speakers.add(value.trim());
    }
  };

  (data.transcript_segments || []).forEach((segment) => push(segment.speaker));
  (data.topic_segments || []).forEach((segment) => {
    push(segment.primary_speaker);
    (segment.turns || []).forEach((turn) => push(turn.speaker));
    if (segment.speaker_counts) {
      Object.keys(segment.speaker_counts).forEach(push);
    }
  });
  (data.topic_summaries || []).forEach((summary) => {
    push(summary.primary_speaker);
    if (summary.speaker_summaries) {
      Object.keys(summary.speaker_summaries).forEach(push);
    }
  });
  (data.diarization_segments || []).forEach((segment) => push(segment.speaker));

  push("Unknown");
  return speakers;
}

function renderSpeakerEditor(result) {
  const container = selectors.speakerEditor;
  if (!container) {
    return;
  }

  const speakers = collectSpeakers(result?.data);
  if (!speakers.size || !state.currentStem) {
    container.classList.remove("active", "speaker-block");
    container.removeAttribute("data-size");
    container.removeAttribute("data-theme");
    container.innerHTML = "";
    return;
  }

  const rows = [...speakers]
    .filter((speaker) => speaker && speaker.length)
    .sort((a, b) => a.localeCompare(b))
    .map(
      (speaker) => `<div class="speaker-row">
        <span class="speaker-tag">${escapeHtml(speaker)}</span>
        <input type="text" data-speaker="${escapeHtml(speaker)}" value="${escapeHtml(speaker)}" />
      </div>`
    )
    .join("");

  container.classList.add("active", "speaker-block");
  container.dataset.size = "md";
  container.dataset.theme = "accent";
  container.innerHTML = `
    <div class="block-header">
      <h3>Speaker Labels</h3>
    </div>
    <div class="block-body">
      <p class="note">Rename diarized speakers to friendlier titles. Changes update transcripts, summaries, and JSON files.</p>
      <div class="speaker-grid">${rows}</div>
      <button type="button" class="button button--primary apply-speaker-btn" data-action="apply-speakers">Apply Names</button>
    </div>
  `;
}

async function applySpeakerRenames() {
  const container = selectors.speakerEditor;
  if (!container || !state.currentStem) {
    return;
  }
  const inputs = container.querySelectorAll("input[data-speaker]");
  const replacements = {};
  inputs.forEach((input) => {
    const original = input.dataset.speaker;
    const value = input.value.trim();
    if (original && value && value !== original) {
      replacements[original] = value;
    }
  });

  if (!Object.keys(replacements).length) {
    setStatus("No speaker names were changed.");
    return;
  }

  try {
    setStatus("Updating speaker names...");
    const response = await fetch(`/api/runs/${encodeURIComponent(state.currentStem)}/rename-speakers`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ replacements }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Failed to update speaker names.");
    }
    const payload = await response.json();
    renderResult(payload);
    await fetchHistory();
    setStatus("Speaker names updated.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Failed to update speaker names.", true);
  }
}

async function fetchHistory() {
  const response = await fetch("/api/runs");
  if (!response.ok) {
    throw new Error("Unable to load history.");
  }
  const payload = await response.json();
  state.history = payload.runs || [];
  renderHistory();
}

function renderHistory() {
  if (!selectors.historyList) {
    return;
  }
  selectors.historyList.innerHTML = "";
  if (!state.history.length) {
    selectors.historyList.innerHTML = "<li>No runs yet.</li>";
    return;
  }
  const fragment = document.createDocumentFragment();
  state.history.forEach((run) => {
    const item = document.createElement("li");
    item.dataset.stem = run.audio_stem;
    const created = run.created_at ? new Date(run.created_at).toLocaleString() : "unknown";
    item.innerHTML = `
      <div class="history-entry-title">${escapeHtml(run.audio_stem || "unknown")}</div>
      <div class="history-entry-details">
        <span>${escapeHtml(run.mode || "all")}</span>
        <span>${escapeHtml(created)}</span>
      </div>
    `;
    fragment.appendChild(item);
  });
  selectors.historyList.appendChild(fragment);
}

async function loadHistoryEntry(stem) {
  if (!stem) {
    return;
  }
  setStatus(`Loading results for ${stem}...`);
  const response = await fetch(`/api/runs/${encodeURIComponent(stem)}`);
  if (!response.ok) {
    setStatus(`Unable to load run for ${stem}.`, true);
    return;
  }
  const result = await response.json();
  renderResult(result);
  setStatus("");
}

function handleTimestampClick(event) {
  const trigger = event.target.closest("[data-time]");
  if (!trigger) {
    return;
  }
  const seconds = Number(trigger.dataset.time);
  if (Number.isNaN(seconds) || !state.audioElement) {
    return;
  }
  if (!state.audioElement.src) {
    return;
  }
  event.preventDefault();
  state.audioElement.currentTime = seconds;
  if (state.audioElement.paused) {
    state.audioElement.play().catch(() => {});
  }
}

async function init() {
  state.audioElement = selectors.audioPlayer || null;

  try {
    await loadDefaults();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load defaults. Check the server logs.", true);
  }
  try {
    await fetchHistory();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load history.", true);
  }

  selectors.form?.addEventListener("submit", submitForm);
  selectors.historyList?.addEventListener("click", (event) => {
    const target = event.target.closest("li[data-stem]");
    if (!target) {
      return;
    }
    loadHistoryEntry(target.dataset.stem);
  });
  selectors.refreshHistory?.addEventListener("click", async () => {
    setStatus("Refreshing history...");
    try {
      await fetchHistory();
      setStatus("History refreshed.");
    } catch (error) {
      console.error(error);
      setStatus("Failed to refresh history.", true);
    }
  });
  selectors.speakerEditor?.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-action='apply-speakers']");
    if (!trigger) {
      return;
    }
    applySpeakerRenames();
  });

  document.addEventListener("click", handleTimestampClick);
}

document.addEventListener("DOMContentLoaded", init);
