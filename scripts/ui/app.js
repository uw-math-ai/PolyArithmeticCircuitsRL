// app.js — Circuit RL inference demo client

const state = {
  polys: [],         // List<PolyInfo>
  models: [],        // List<ModelInfo>
  results: {},       // {label -> {idx -> ResultPayload}}
  eventSource: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const els = {
  runBtn: $("#run-btn"),
  searchSims: $("#search-sims"),
  modeToggle: $("#mode-toggle"),
  modelCards: $("#model-cards"),
  polyTables: $("#poly-tables"),
};

function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

// ---------------------------------------------------------------------------
// Initial data load
// ---------------------------------------------------------------------------
async function loadInitialData() {
  try {
    const [suiteRes, modelsRes] = await Promise.all([
      fetch("/api/test-suite").then((r) => r.json()),
      fetch("/api/models").then((r) => r.json()),
    ]);
    state.polys = suiteRes.polynomials;
    state.models = modelsRes.models.filter(
      (m) => m.kind === "baseline" || m.exists
    );
    renderModelCards();
    renderTables();
  } catch (e) {
    els.polyTables.innerHTML =
      `<p class="empty-state">Failed to load suite: ${e}</p>`;
  }
}

// ---------------------------------------------------------------------------
// Model summary cards
// ---------------------------------------------------------------------------
function renderModelCards() {
  els.modelCards.innerHTML = "";
  for (const m of state.models) {
    const card = document.createElement("div");
    card.className = `model-card kind-${m.kind}`;
    card.id = `card-${m.label}`;
    card.innerHTML = `
      <div class="kind-tag">${m.kind === "baseline" ? "Baseline" : "Checkpoint"}</div>
      <div class="display-name">${m.display}</div>
      <div class="info-line" id="info-${m.label}">&nbsp;</div>
      <div class="progress-track">
        <div class="progress-fill" id="bar-${m.label}"></div>
      </div>
      <div class="stat-row">
        <div class="stat">
          <span class="stat-value" id="stat-greedy-${m.label}">–</span>
          <span class="stat-label">Greedy ✓</span>
        </div>
        <div class="stat">
          <span class="stat-value" id="stat-search-${m.label}">–</span>
          <span class="stat-label">Search ✓</span>
        </div>
        <div class="stat">
          <span class="stat-value" id="stat-time-${m.label}">–</span>
          <span class="stat-label">Time</span>
        </div>
      </div>
    `;
    els.modelCards.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Polynomial tables (grouped)
// ---------------------------------------------------------------------------
function renderTables() {
  els.polyTables.innerHTML = "";

  // Preserve group order based on the suite ordering.
  const groupOrder = [];
  const groups = {};
  for (const p of state.polys) {
    if (!groups[p.group]) {
      groups[p.group] = [];
      groupOrder.push(p.group);
    }
    groups[p.group].push(p);
  }

  for (const groupName of groupOrder) {
    const sec = document.createElement("div");
    sec.className = "poly-table-group";
    sec.innerHTML = `<h3 class="group-title">${groupName}</h3>`;

    const table = document.createElement("table");
    table.className = "poly-table";
    const colHeaders = state.models
      .map((m) => `<th class="col-model">${m.display}</th>`)
      .join("");
    table.innerHTML = `
      <thead>
        <tr>
          <th class="col-poly">Polynomial</th>
          <th class="col-bmin" title="Minimum cost across all five closed-form baselines">Min op</th>
          ${colHeaders}
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const tbody = table.querySelector("tbody");
    for (const p of groups[groupName]) {
      const row = document.createElement("tr");
      row.dataset.idx = p.index;
      const cells = state.models
        .map(
          (m) =>
            `<td class="result-cell pending" data-model="${m.label}" data-idx="${p.index}">–</td>`
        )
        .join("");
      row.innerHTML = `
        <td class="poly-cell">
          <span class="latex">\\(${p.latex}\\)</span>
          <span class="prime-badge">F<sub>${p.prime}</sub></span>
          <span class="name">${p.name}</span>
        </td>
        <td class="bmin-cell">${p.baseline_min}</td>
        ${cells}
      `;
      tbody.appendChild(row);
    }
    sec.appendChild(table);
    els.polyTables.appendChild(sec);
  }

  if (window.renderMathInElement) {
    renderMathInElement(els.polyTables, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}

// ---------------------------------------------------------------------------
// Cell updates
// ---------------------------------------------------------------------------
function getCell(label, idx) {
  return document.querySelector(
    `.result-cell[data-model="${label}"][data-idx="${idx}"]`
  );
}

function updateCell(label, idx, result) {
  const cell = getCell(label, idx);
  if (!cell) return;
  const mode = getMode();
  const g = { cost: result.greedy_cost, ok: result.greedy_success };
  const s = { cost: result.search_cost, ok: result.search_success };
  const ttGreedy = `Greedy: ${g.cost} ${g.ok ? "✓" : "✗"}`;
  const ttSearch = `Search: ${s.cost} ${s.ok ? "✓" : "✗"}`;
  const ttTime = `Latency: ${result.elapsed_ms}ms · nodes=${result.node_expansions}, hits=${result.transposition_hits}`;
  cell.title = `${ttGreedy}\n${ttSearch}\n${ttTime}`;
  cell.classList.remove("running", "pending");
  if (mode === "both") {
    cell.classList.remove("success", "fail");
    cell.innerHTML = `
      <div class="dual">
        <span class="part g ${g.ok ? "ok" : "bad"}">
          <span class="tag">G</span>${g.cost}${g.ok ? "✓" : "✗"}
        </span>
        <span class="part s ${s.ok ? "ok" : "bad"}">
          <span class="tag">S</span>${s.cost}${s.ok ? "✓" : "✗"}
        </span>
      </div>
    `;
  } else {
    const sel = mode === "search" ? s : g;
    cell.className = `result-cell ${sel.ok ? "success" : "fail"}`;
    cell.innerHTML = `${sel.cost} <span class="icon">${sel.ok ? "✓" : "✗"}</span>`;
  }
}

function refreshAllCells() {
  for (const label in state.results) {
    for (const idx in state.results[label]) {
      updateCell(label, idx, state.results[label][idx]);
    }
  }
}

$$('input[name="mode"]').forEach((el) =>
  el.addEventListener("change", refreshAllCells)
);

// ---------------------------------------------------------------------------
// Per-model summary stats
// ---------------------------------------------------------------------------
function updateStats(label) {
  const results = Object.values(state.results[label] || {});
  const total = state.polys.length;
  const gOk = results.filter((r) => r.greedy_success).length;
  const sOk = results.filter((r) => r.search_success).length;
  $(`#stat-greedy-${label}`).textContent = `${gOk}/${total}`;
  $(`#stat-search-${label}`).textContent = `${sOk}/${total}`;
  $(`#bar-${label}`).style.width = `${(100 * results.length) / total}%`;
}

function resetEvaluationUI() {
  $$(".result-cell").forEach((cell) => {
    cell.className = "result-cell pending";
    cell.textContent = "–";
    cell.title = "";
  });
  state.models.forEach((m) => {
    $(`#stat-greedy-${m.label}`).textContent = "–";
    $(`#stat-search-${m.label}`).textContent = "–";
    $(`#stat-time-${m.label}`).textContent = "–";
    $(`#bar-${m.label}`).style.width = "0%";
    $(`#info-${m.label}`).innerHTML = "&nbsp;";
  });
  state.results = {};
}

function formatInfo(label, info) {
  if (!info) return "";
  if (info.description) return info.description;
  const parts = [];
  if (info.cycle !== undefined && info.cycle !== null) parts.push(`cycle ${info.cycle}`);
  if (info.params) parts.push(`${(info.params / 1e6).toFixed(1)}M params`);
  if (info.hidden_dim) parts.push(`h=${info.hidden_dim}`);
  if (info.holdout_gain != null) parts.push(`holdout gain ${info.holdout_gain.toFixed(2)}`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------------------
// Run inference
// ---------------------------------------------------------------------------
els.runBtn.addEventListener("click", () => {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  resetEvaluationUI();

  const url = new URL("/api/evaluate", window.location.origin);
  url.searchParams.set("search_sims", els.searchSims.value);
  state.models.forEach((m) => url.searchParams.append("models", m.label));

  els.runBtn.disabled = true;
  els.runBtn.querySelector(".btn-label").textContent = "Running…";

  const es = new EventSource(url.toString());
  state.eventSource = es;

  es.addEventListener("session-start", () => {});

  es.addEventListener("model-start", (e) => {
    const data = JSON.parse(e.data);
    $(`#info-${data.label}`).textContent = formatInfo(data.label, data.info);
    $$(`.result-cell[data-model="${data.label}"]`).forEach((c) =>
      c.classList.add("running")
    );
  });

  es.addEventListener("result", (e) => {
    const data = JSON.parse(e.data);
    if (!state.results[data.label]) state.results[data.label] = {};
    state.results[data.label][data.poly_index] = data;
    updateCell(data.label, data.poly_index, data);
    updateStats(data.label);
  });

  es.addEventListener("model-done", (e) => {
    const data = JSON.parse(e.data);
    $(`#stat-time-${data.label}`).textContent = `${data.elapsed_sec}s`;
    $$(`.result-cell[data-model="${data.label}"]`).forEach((c) =>
      c.classList.remove("running")
    );
    $(`#bar-${data.label}`).style.width = "100%";
  });

  es.addEventListener("model-error", (e) => {
    const data = JSON.parse(e.data);
    $(`#info-${data.label}`).innerHTML =
      `<span class="error-msg">${data.error}</span>`;
  });

  es.addEventListener("session-complete", () => {
    es.close();
    state.eventSource = null;
    els.runBtn.disabled = false;
    els.runBtn.querySelector(".btn-label").textContent = "Run Inference";
  });

  es.onerror = () => {
    es.close();
    state.eventSource = null;
    els.runBtn.disabled = false;
    els.runBtn.querySelector(".btn-label").textContent = "Run Inference";
  };
});

loadInitialData();
