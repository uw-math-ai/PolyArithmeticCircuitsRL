// app.js — Circuit RL inference demo client

const state = {
  polys: [],          // List<PolyInfo>
  allModels: [],      // List<ModelInfo>  (heuristic + every discovered ckpt)
  selectedCkpts: new Set(),  // labels of checkpoints currently picked
  results: {},        // {label -> {idx -> ResultPayload}}
  circuits: {},       // {label -> {idx -> {greedy_circuit, search_circuit, greedy_circuit_cost, search_circuit_cost}}}
  eventSource: null,
  checkpointDir: "",
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const els = {
  runBtn:       $("#run-btn"),
  searchSims:   $("#search-sims"),
  modeToggle:   $("#mode-toggle"),
  modelCards:   $("#model-cards"),
  polyTables:   $("#poly-tables"),
  chipRow:      $("#chip-row"),
  pickerSub:    $("#picker-sub"),
  pickAll:      $("#pick-all"),
  pickClear:    $("#pick-clear"),
  pickRefresh:  $("#pick-refresh"),
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
    state.allModels = modelsRes.models;
    state.checkpointDir = modelsRes.checkpoint_dir || "";

    // Default selection: every discovered checkpoint.
    state.selectedCkpts = new Set(
      state.allModels.filter((m) => m.kind === "checkpoint").map((m) => m.label)
    );

    renderPicker();
    rebuildResultsViews();
  } catch (e) {
    els.polyTables.innerHTML = `<p class="empty-state">Failed to load: ${e}</p>`;
  }
}

async function refreshModels() {
  els.pickRefresh.disabled = true;
  els.pickRefresh.textContent = "↻ Refreshing…";
  try {
    const res = await fetch("/api/models").then((r) => r.json());
    state.allModels = res.models;
    state.checkpointDir = res.checkpoint_dir || "";
    // Preserve current selection where possible.
    const labels = new Set(state.allModels.map((m) => m.label));
    for (const l of [...state.selectedCkpts]) {
      if (!labels.has(l)) state.selectedCkpts.delete(l);
    }
    renderPicker();
    rebuildResultsViews();
  } finally {
    els.pickRefresh.disabled = false;
    els.pickRefresh.textContent = "↻ Refresh";
  }
}

// ---------------------------------------------------------------------------
// Checkpoint picker (chip-style)
// ---------------------------------------------------------------------------
function ckptList() {
  return state.allModels.filter((m) => m.kind === "checkpoint");
}

function renderPicker() {
  const ckpts = ckptList();
  els.chipRow.innerHTML = "";
  if (ckpts.length === 0) {
    els.chipRow.innerHTML =
      `<p class="empty-state-small">No checkpoints found in ${
        state.checkpointDir || "checkpoint dir"
      }.</p>`;
  } else {
    for (const m of ckpts) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.dataset.label = m.label;
      if (state.selectedCkpts.has(m.label)) chip.classList.add("selected");
      const meta = m.meta || {};
      const metaParts = [];
      if (meta.cycle !== undefined && meta.cycle !== null)
        metaParts.push(`cycle ${meta.cycle}`);
      if (m.size_mb) metaParts.push(`${m.size_mb}MB`);
      chip.innerHTML = `
        <span class="chip-tick">✓</span>
        <span class="chip-label">${m.display}</span>
        ${metaParts.length ? `<span class="chip-meta">${metaParts.join(" · ")}</span>` : ""}
      `;
      chip.addEventListener("click", () => {
        if (state.selectedCkpts.has(m.label)) {
          state.selectedCkpts.delete(m.label);
        } else {
          state.selectedCkpts.add(m.label);
        }
        chip.classList.toggle("selected");
        updatePickerSub();
        rebuildResultsViews();
      });
      els.chipRow.appendChild(chip);
    }
  }
  updatePickerSub();
}

function updatePickerSub() {
  const ckpts = ckptList();
  const n = state.selectedCkpts.size;
  els.pickerSub.textContent =
    `${ckpts.length} found in ${state.checkpointDir || "<unknown>"} · ${n} selected`;
}

els.pickAll.addEventListener("click", () => {
  state.selectedCkpts = new Set(ckptList().map((m) => m.label));
  renderPicker();
  rebuildResultsViews();
});
els.pickClear.addEventListener("click", () => {
  state.selectedCkpts.clear();
  renderPicker();
  rebuildResultsViews();
});
els.pickRefresh.addEventListener("click", refreshModels);

// ---------------------------------------------------------------------------
// Active models = heuristic (always) + selected checkpoints (in suite order)
// ---------------------------------------------------------------------------
function activeModels() {
  const out = [];
  for (const m of state.allModels) {
    if (m.kind === "baseline") {
      out.push(m);
    } else if (state.selectedCkpts.has(m.label)) {
      out.push(m);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Per-model summary cards
// ---------------------------------------------------------------------------
function renderModelCards() {
  els.modelCards.innerHTML = "";
  for (const m of activeModels()) {
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
          <span class="stat-label">Greedy optimal</span>
        </div>
        <div class="stat">
          <span class="stat-value" id="stat-search-${m.label}">–</span>
          <span class="stat-label">Search optimal</span>
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
// Polynomial tables (grouped, scrollable when wide)
// ---------------------------------------------------------------------------
function renderTables() {
  els.polyTables.innerHTML = "";

  const groupOrder = [];
  const groups = {};
  for (const p of state.polys) {
    if (!groups[p.group]) {
      groups[p.group] = [];
      groupOrder.push(p.group);
    }
    groups[p.group].push(p);
  }

  const models = activeModels();

  for (const groupName of groupOrder) {
    const sec = document.createElement("div");
    sec.className = "poly-table-group";
    sec.innerHTML = `<h3 class="group-title">${groupName}</h3>`;

    const wrap = document.createElement("div");
    wrap.className = "poly-table-wrap";

    const table = document.createElement("table");
    table.className = "poly-table";
    const colHeaders = models
      .map((m) => `<th class="col-model">${m.display}</th>`)
      .join("");
    table.innerHTML = `
      <thead>
        <tr>
          <th class="col-poly">Polynomial</th>
          <th class="col-bmin" title="Provably-minimal gate count (binary +/×; only the constant 1 is free; reuse is free). Hover for the optimal circuit.">Min op</th>
          ${colHeaders}
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const tbody = $("tbody", table);
    for (const p of groups[groupName]) {
      const row = document.createElement("tr");
      row.dataset.idx = p.index;
      const cells = models
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
    wrap.appendChild(table);
    sec.appendChild(wrap);
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

  // Re-paint any cached results into the freshly-built table.
  for (const label in state.results) {
    for (const idx in state.results[label]) {
      updateCell(label, idx, state.results[label][idx]);
    }
    updateStats(label);
  }
}

function rebuildResultsViews() {
  renderModelCards();
  renderTables();
}

// ---------------------------------------------------------------------------
// Cell updates
// ---------------------------------------------------------------------------
function getCell(label, idx) {
  return document.querySelector(
    `.result-cell[data-model="${label}"][data-idx="${idx}"]`
  );
}

// Badge shown next to a cost: ✓ at optimal, otherwise +N over optimal.
function deltaBadge(delta) {
  return delta <= 0 ? "✓" : `+${delta}`;
}

function updateCell(label, idx, result) {
  const cell = getCell(label, idx);
  if (!cell) return;
  const mode = getMode();
  const opt = result.optimal;
  const g = { cost: result.greedy_cost, tier: result.greedy_tier };
  const s = { cost: result.search_cost, tier: result.search_tier };
  const desc = (c) => (c - opt <= 0 ? "optimal" : `${c - opt} op${c - opt > 1 ? "s" : ""} from optimal`);
  const tt = [
    `Optimal: ${opt}`,
    `Greedy: ${g.cost} (${desc(g.cost)})`,
    `Search: ${s.cost} (${desc(s.cost)})`,
    `Latency: ${result.elapsed_ms}ms · nodes=${result.node_expansions}, hits=${result.transposition_hits}`,
  ];
  cell.title = tt.join("\n");
  cell.classList.remove("running", "pending");
  if (mode === "both") {
    cell.className = "result-cell";
    cell.innerHTML = `
      <div class="dual">
        <span class="part g tier-${g.tier}">
          <span class="tag">G</span>${g.cost}<span class="icon">${deltaBadge(g.cost - opt)}</span>
        </span>
        <span class="part s tier-${s.tier}">
          <span class="tag">S</span>${s.cost}<span class="icon">${deltaBadge(s.cost - opt)}</span>
        </span>
      </div>
    `;
  } else {
    const sel = mode === "search" ? s : g;
    cell.className = `result-cell tier-${sel.tier}`;
    cell.innerHTML = `${sel.cost} <span class="icon">${deltaBadge(sel.cost - opt)}</span>`;
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
  const gOk = results.filter((r) => r.greedy_tier === 0).length;
  const sOk = results.filter((r) => r.search_tier === 0).length;
  const gEl = $(`#stat-greedy-${label}`);
  const sEl = $(`#stat-search-${label}`);
  const bar = $(`#bar-${label}`);
  if (gEl) gEl.textContent = `${gOk}/${total}`;
  if (sEl) sEl.textContent = `${sOk}/${total}`;
  if (bar) bar.style.width = `${(100 * results.length) / total}%`;
}

function resetEvaluationUI() {
  $$(".result-cell").forEach((cell) => {
    cell.className = "result-cell pending";
    cell.textContent = "–";
    cell.title = "";
  });
  for (const m of activeModels()) {
    const g = $(`#stat-greedy-${m.label}`);
    const s = $(`#stat-search-${m.label}`);
    const t = $(`#stat-time-${m.label}`);
    const i = $(`#info-${m.label}`);
    const b = $(`#bar-${m.label}`);
    if (g) g.textContent = "–";
    if (s) s.textContent = "–";
    if (t) t.textContent = "–";
    if (b) b.style.width = "0%";
    if (i) i.innerHTML = "&nbsp;";
  }
  state.results = {};
  state.circuits = {};
}

function formatInfo(info) {
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
  for (const m of activeModels()) url.searchParams.append("models", m.label);

  els.runBtn.disabled = true;
  els.runBtn.querySelector(".btn-label").textContent = "Running…";

  const es = new EventSource(url.toString());
  state.eventSource = es;

  es.addEventListener("session-start", () => {});

  es.addEventListener("model-start", (e) => {
    const data = JSON.parse(e.data);
    const info = $(`#info-${data.label}`);
    if (info) info.textContent = formatInfo(data.info);
    $$(`.result-cell[data-model="${data.label}"]`).forEach((c) =>
      c.classList.add("running")
    );
  });

  es.addEventListener("result", (e) => {
    const data = JSON.parse(e.data);
    if (!state.results[data.label]) state.results[data.label] = {};
    state.results[data.label][data.poly_index] = data;
    if (!state.circuits[data.label]) state.circuits[data.label] = {};
    state.circuits[data.label][data.poly_index] = {
      greedy_circuit: data.greedy_circuit,
      search_circuit: data.search_circuit,
      greedy_circuit_cost: data.greedy_circuit_cost,
      search_circuit_cost: data.search_circuit_cost,
    };
    updateCell(data.label, data.poly_index, data);
    updateStats(data.label);
  });

  es.addEventListener("model-done", (e) => {
    const data = JSON.parse(e.data);
    const t = $(`#stat-time-${data.label}`);
    const b = $(`#bar-${data.label}`);
    if (t) t.textContent = `${data.elapsed_sec}s`;
    if (b) b.style.width = "100%";
    $$(`.result-cell[data-model="${data.label}"]`).forEach((c) =>
      c.classList.remove("running")
    );
  });

  es.addEventListener("model-error", (e) => {
    const data = JSON.parse(e.data);
    const info = $(`#info-${data.label}`);
    if (info) info.innerHTML = `<span class="error-msg">${data.error}</span>`;
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

// ---------------------------------------------------------------------------
// Circuit hover popover
// ---------------------------------------------------------------------------
const _pop = document.getElementById("circ-popover");
if (!_pop) {
  throw new Error("BUG: #circ-popover not found — ensure the div precedes this script in index.html");
}
const _popTitle = _pop.querySelector(".circ-popover-title");
const _popSub   = _pop.querySelector(".circ-popover-sub");
const _popBody  = _pop.querySelector(".circ-popover-body");

function _positionPopover(anchorEl) {
  const rect = anchorEl.getBoundingClientRect();
  const margin = 10;
  const pw = _pop.offsetWidth;
  const ph = _pop.offsetHeight;

  // Prefer below; flip above if it would overflow.
  let top = rect.bottom + margin;
  if (top + ph > window.innerHeight - margin) {
    top = rect.top - ph - margin;
  }
  top = Math.max(margin, top);

  // Center horizontally on the anchor cell, clamped to viewport.
  let left = rect.left + rect.width / 2 - pw / 2;
  left = Math.max(margin, Math.min(window.innerWidth - pw - margin, left));

  _pop.style.top  = top  + "px";
  _pop.style.left = left + "px";
}

function showCircPopover(anchorEl, circuit, title, subtitle) {
  _popTitle.textContent = title;
  _popSub.textContent   = subtitle;
  _popBody.innerHTML    = "";

  if (window.renderCircuit) {
    const { el } = window.renderCircuit(circuit);
    _popBody.appendChild(el);
  } else {
    _popBody.textContent = "(circuit renderer unavailable)";
  }

  _pop.classList.add("visible");
  _positionPopover(anchorEl);
}

function hideCircPopover() {
  _pop.classList.remove("visible");
}

function _showResultCell(cell) {
  const label = cell.dataset.model;
  const idx   = parseInt(cell.dataset.idx, 10);
  const circs = (state.circuits[label] || {})[idx];
  if (!circs) return;

  const mode     = getMode();
  const useSearch = mode !== "greedy";
  const circuit  = useSearch ? circs.search_circuit  : circs.greedy_circuit;
  const cost     = useSearch ? circs.search_circuit_cost : circs.greedy_circuit_cost;
  if (!circuit) return;

  const model    = state.allModels.find((m) => m.label === label);
  const name     = model ? model.display : label;
  const modeTag  = useSearch ? "Search" : "Greedy";
  showCircPopover(cell, circuit, `${name} · ${modeTag}`, `Circuit cost: ${cost}`);
}

function _showBminCell(cell) {
  const row = cell.closest("tr");
  if (!row) return;
  const idx  = parseInt(row.dataset.idx, 10);
  const poly = state.polys[idx];
  if (!poly || !poly.optimal_circuit) return;
  showCircPopover(cell, poly.optimal_circuit, "Optimal (baseline)",
                  `Circuit cost: ${poly.optimal_circuit_cost}`);
}

// Use event delegation on the entire poly-tables section for efficiency.
els.polyTables.addEventListener("mouseover", (e) => {
  const rc = e.target.closest(".result-cell");
  const bc = e.target.closest(".bmin-cell");
  if (rc && !rc.classList.contains("pending") && !rc.classList.contains("running")) {
    _showResultCell(rc);
  } else if (bc) {
    _showBminCell(bc);
  } else if (!e.target.closest("#circ-popover")) {
    hideCircPopover();
  }
});

els.polyTables.addEventListener("mouseout", (e) => {
  const to = e.relatedTarget;
  if (!to || (!to.closest(".result-cell") && !to.closest(".bmin-cell"))) {
    hideCircPopover();
  }
});

loadInitialData();
