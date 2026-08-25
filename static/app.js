const state = { mode: "document" };

const sourceCol = document.getElementById("source-col");
const modeTabs = document.querySelectorAll(".mode-tab");
const sourceFileInput = document.getElementById("source-file");
const responseFileInput = document.getElementById("response-file");
const sourceFilename = document.getElementById("source-filename");
const responseFilename = document.getElementById("response-filename");
const sourceText = document.getElementById("source-text");
const responseText = document.getElementById("response-text");
const runBtn = document.getElementById("run-btn");
const modelChoice = document.getElementById("model-choice");
const statusRow = document.getElementById("status-row");
const statusText = document.getElementById("status-text");
const results = document.getElementById("results");
const evidenceList = document.getElementById("evidence-list");
const correctionPanel = document.getElementById("correction-panel");
const correctionText = document.getElementById("correction-text");
const allClear = document.getElementById("all-clear");

modeTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    modeTabs.forEach(t => { t.classList.remove("is-active"); t.setAttribute("aria-selected", "false"); });
    tab.classList.add("is-active");
    tab.setAttribute("aria-selected", "true");
    state.mode = tab.dataset.mode;
    sourceCol.style.display = state.mode === "document" ? "flex" : "none";
  });
});

sourceFileInput.addEventListener("change", () => {
  sourceFilename.textContent = sourceFileInput.files[0] ? sourceFileInput.files[0].name : "";
});
responseFileInput.addEventListener("change", () => {
  responseFilename.textContent = responseFileInput.files[0] ? responseFileInput.files[0].name : "";
});

const statusMessages = [
  "Extracting atomic claims…",
  "Cross-referencing against source…",
  "Scoring faithfulness…",
  "Drafting corrected response…",
];

let statusInterval = null;
function startStatusCycle() {
  let i = 0;
  statusText.textContent = statusMessages[0];
  statusInterval = setInterval(() => {
    i = (i + 1) % statusMessages.length;
    statusText.textContent = statusMessages[i];
  }, 2600);
}
function stopStatusCycle() {
  clearInterval(statusInterval);
}

runBtn.addEventListener("click", async () => {
  const respVal = responseText.value.trim();
  if (!respVal && !responseFileInput.files[0]) {
    alert("Please paste or upload the LLM response.");
    return;
  }
  if (state.mode === "document" && !sourceText.value.trim() && !sourceFileInput.files[0]) {
    alert("Please paste or upload the source document.");
    return;
  }

  runBtn.disabled = true;
  results.hidden = true;
  statusRow.hidden = false;
  startStatusCycle();

  try {
    const formData = new FormData();
    formData.append("llm_response", responseText.value);
    formData.append("model", modelChoice.value);
    if (responseFileInput.files[0]) formData.append("response_file", responseFileInput.files[0]);

    let endpoint = "/api/check-live";
    if (state.mode === "document") {
      endpoint = "/api/check";
      formData.append("source_text", sourceText.value);
      if (sourceFileInput.files[0]) formData.append("source_file", sourceFileInput.files[0]);
    }

    const res = await fetch(endpoint, { method: "POST", body: formData });
    if (!res.ok) {
      let detail = `Server error: ${res.status}`;
      try {
        const errBody = await res.json();
        if (errBody.error) detail = errBody.error;
        else if (errBody.detail) detail = JSON.stringify(errBody.detail);
      } catch (_) {}
      throw new Error(detail);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("Something went wrong: " + err.message);
    console.error(err);
  } finally {
    stopStatusCycle();
    statusRow.hidden = true;
    runBtn.disabled = false;
  }
});

function renderResults(data) {
  const { score, results: claimResults, corrected_response } = data;

  document.getElementById("stat-supported").textContent = score.supported;
  document.getElementById("stat-contradicted").textContent = score.contradicted;
  document.getElementById("stat-unsupported").textContent = score.unsupported;
  document.getElementById("stat-unverified").textContent = score.unverified || 0;
  document.getElementById("stat-total").textContent = score.total_claims;

  const pct = score.faithfulness_score;
  const gaugeFill = document.getElementById("gauge-fill");
  const gaugeNumber = document.getElementById("gauge-number");

  const circumference = 427;
  const offset = circumference - (pct / 100) * circumference;
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = pct >= 70 ? "#2E9E83" : pct >= 40 ? "#B8791F" : "#C1473C";

  animateCount(gaugeNumber, 0, Math.round(pct), 900);

  evidenceList.innerHTML = "";
  claimResults.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = `evidence-card evidence-card--${r.label}`;
    card.style.animationDelay = `${i * 70}ms`;

    const scoreVal = r.label === "supported" ? r.entailment_score
      : r.label === "contradicted" ? r.contradiction_score
      : r.label === "unverified" ? null
      : r.neutral_score;

    const scoreDisplay = scoreVal === null ? "no evidence" : scoreVal.toFixed(3);

    card.innerHTML = `
      <span class="evidence-claim">${escapeHtml(r.claim)}</span>
      <span class="evidence-meta">
        <span class="evidence-score">${scoreDisplay}</span>
        <span class="evidence-stamp evidence-stamp--${r.label}">${r.label}</span>
      </span>
    `;
    evidenceList.appendChild(card);
  });

  if (corrected_response) {
    correctionText.textContent = corrected_response;
    correctionPanel.hidden = false;
    allClear.hidden = true;
  } else {
    correctionPanel.hidden = true;
    allClear.hidden = false;
  }

  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function animateCount(el, from, to, duration) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
