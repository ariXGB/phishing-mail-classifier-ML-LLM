const MODEL_LABELS = { lr: "Logistic Regression", nb: "Naive Bayes", xgb: "XGBoost", minilm: "MiniLM" };
const DEFAULT_BASE = "http://localhost:8000";

function getBase(){
  return localStorage.getItem("phishscan_api_base") || DEFAULT_BASE;
}
function setBase(v){
  localStorage.setItem("phishscan_api_base", v);
}

// ---- tabs ----
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".pane").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("pane-" + tab.dataset.tab).classList.add("active");
  });
});

// ---- model chips ----
const chipWrap = document.getElementById("modelChips");
const selectedModels = new Set(["minilm"]);
Object.entries(MODEL_LABELS).forEach(([key, label]) => {
  const chip = document.createElement("label");
  chip.className = "chip" + (selectedModels.has(key) ? " checked" : "");
  chip.innerHTML = `<input type="checkbox" ${selectedModels.has(key) ? "checked" : ""}> ${label}`;
  chip.querySelector("input").addEventListener("change", (e) => {
    if (e.target.checked){ selectedModels.add(key); chip.classList.add("checked"); }
    else { selectedModels.delete(key); chip.classList.remove("checked"); }
  });
  chipWrap.appendChild(chip);
});

// ---- config pane ----
const apiBaseInput = document.getElementById("apiBase");
apiBaseInput.value = getBase();
apiBaseInput.addEventListener("change", () => {
  setBase(apiBaseInput.value.trim().replace(/\/$/, ""));
  checkLive();
});

async function checkLive(){
  const pill = document.getElementById("livePill");
  const text = document.getElementById("liveText");
  const configStatus = document.getElementById("configStatus");
  try {
    const res = await fetch(getBase() + "/docs", { method: "GET" });
    if (res.ok){
      pill.className = "live-pill ok";
      text.textContent = "api online";
      configStatus.textContent = "✓ reachable";
    } else {
      throw new Error("bad status");
    }
  } catch (e){
    pill.className = "live-pill bad";
    text.textContent = "api unreachable";
    configStatus.textContent = "✗ unreachable — check config tab";
  }
}
checkLive();

// ---- single message ----
const analyzeBtn = document.getElementById("analyzeBtn");
const scanLine = document.getElementById("scanLine");
const singleResults = document.getElementById("singleResults");
const singleError = document.getElementById("singleError");

analyzeBtn.addEventListener("click", async () => {
  const text = document.getElementById("msgInput").value.trim();
  singleError.innerHTML = "";
  singleResults.innerHTML = "";

  if (!text){
    singleError.innerHTML = `<div class="err">Enter some message text first.</div>`;
    return;
  }
  if (selectedModels.size === 0){
    singleError.innerHTML = `<div class="err">Select at least one model.</div>`;
    return;
  }

  analyzeBtn.disabled = true;
  scanLine.classList.add("active");

  try {
    const form = new FormData();
    form.append("text", text);
    selectedModels.forEach(m => form.append("model_names", m));

    const res = await fetch(getBase() + "/predict-text", { method: "POST", body: form });
    if (!res.ok){
      const detail = await res.text();
      throw new Error(`${res.status} ${res.statusText} — ${detail}`);
    }
    const data = await res.json();

    data.forEach(item => {
      const isPhishing = Number(item.prediction) === 1;
      const row = document.createElement("div");
      row.className = "result-row " + (isPhishing ? "phishing" : "legit");
      const confText = item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "n/a";
      row.innerHTML = `
        <span class="prompt">&gt;</span>
        <span class="model">${MODEL_LABELS[item.model_name] || item.model_name}</span>
        <span class="verdict">${isPhishing ? "⚠ likely phishing" : "✓ likely legitimate"}</span>
        <span class="conf">conf ${confText}</span>
      `;
      singleResults.appendChild(row);
    });
  } catch (e){
    singleError.innerHTML = `<div class="err">Request failed: ${e.message}</div>`;
  } finally {
    analyzeBtn.disabled = false;
    scanLine.classList.remove("active");
  }
});

// ---- batch csv ----
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzoneText");
const fileInput = document.getElementById("fileInput");
const batchConfigRow = document.getElementById("batchConfigRow");
const textColumnSelect = document.getElementById("textColumn");
const batchModelSelect = document.getElementById("batchModel");
const batchBtn = document.getElementById("batchBtn");
const batchError = document.getElementById("batchError");
const batchOutput = document.getElementById("batchOutput");

let currentFile = null;
let currentColumns = [];

Object.entries(MODEL_LABELS).forEach(([key, label]) => {
  const opt = document.createElement("option");
  opt.value = key; opt.textContent = label;
  batchModelSelect.appendChild(opt);
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => {
  e.preventDefault(); dropzone.classList.remove("drag");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file){
  currentFile = file;
  dropzoneText.innerHTML = `<b>${file.name}</b> selected — click to change`;
  batchError.innerHTML = "";
  batchOutput.innerHTML = "";

  const reader = new FileReader();
  reader.onload = (e) => {
    const firstLine = e.target.result.split(/\r?\n/)[0] || "";
    currentColumns = firstLine.split(",").map(c => c.trim().replace(/^"|"$/g, ""));
    textColumnSelect.innerHTML = "";
    currentColumns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col; opt.textContent = col;
      textColumnSelect.appendChild(opt);
    });
    batchConfigRow.style.display = "block";
  };
  reader.readAsText(file);
}

batchBtn.addEventListener("click", async () => {
  batchError.innerHTML = "";
  batchOutput.innerHTML = "";
  if (!currentFile){
    batchError.innerHTML = `<div class="err">Upload a CSV first.</div>`;
    return;
  }

  batchBtn.disabled = true;
  batchBtn.textContent = "▶ scoring rows…";

  try {
    const form = new FormData();
    form.append("file", currentFile);
    form.append("model_name", batchModelSelect.value);
    form.append("text_column", textColumnSelect.value);

    const res = await fetch(getBase() + "/predict-csv", { method: "POST", body: form });
    if (!res.ok){
      const detail = await res.text();
      throw new Error(`${res.status} ${res.statusText} — ${detail}`);
    }
    const rows = await res.json();
    renderBatchResults(rows);
  } catch (e){
    batchError.innerHTML = `<div class="err">Request failed: ${e.message}</div>`;
  } finally {
    batchBtn.disabled = false;
    batchBtn.textContent = "▶ run batch prediction";
  }
});

function renderBatchResults(rows){
  if (!rows.length){
    batchOutput.innerHTML = `<div class="err">No rows returned.</div>`;
    return;
  }
  const cols = Object.keys(rows[0]);

  let html = `<div style="font-family:var(--mono);font-size:12.5px;color:var(--safe);margin-top:14px;">✓ scored ${rows.length} rows</div>`;
  html += `<div class="table-wrap"><table><thead><tr>`;
  cols.forEach(c => html += `<th>${c}</th>`);
  html += `</tr></thead><tbody>`;
  rows.forEach(r => {
    html += "<tr>" + cols.map(c => `<td>${r[c] ?? ""}</td>`).join("") + "</tr>";
  });
  html += `</tbody></table></div>`;
  batchOutput.innerHTML = html;

  const csvContent = [cols.join(",")]
    .concat(rows.map(r => cols.map(c => JSON.stringify(r[c] ?? "")).join(",")))
    .join("\n");
  const blob = new Blob([csvContent], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "phishing_predictions.csv";
  link.className = "download-btn";
  link.textContent = "⬇ download results as csv";
  batchOutput.appendChild(link);
}
