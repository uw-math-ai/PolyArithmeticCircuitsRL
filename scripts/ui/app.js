// app.js — Circuit RL inference demo client

const state = {
  polys: [],          // List<PolyInfo>
  allModels: [],      // List<ModelInfo>  (heuristic + every discovered ckpt)
  selectedCkpts: new Set(),  // labels of checkpoints currently picked
  results: {},        // {label -> {idx -> ResultPayload}}
  circuits: {},       // {label -> {idx -> {greedy_circuit, search_circuit, greedy_circuit_cost, search_circuit_cost}}}
  eventSource: null,
  checkpointDir: "",
  activeTab: "inference",
  game: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function modelDomId(label) {
  return `model-${String(label).replace(/[^A-Za-z0-9_-]/g, "_")}`;
}

function modelByLabel(label) {
  return state.allModels.find((m) => m.label === label);
}

function modelFieldLabel(model) {
  return model?.meta?.field_label || "unknown field";
}

function modelSupportsPrime(model, prime) {
  const primes = model?.meta?.supported_primes;
  return !Array.isArray(primes) || primes.includes(prime);
}

function unsupportedText(model) {
  const primes = model?.meta?.supported_primes;
  return Array.isArray(primes) && primes.length
    ? primes.map((p) => `F${p}`).join(", ") + " only"
    : "field unknown";
}

function unsupportedTitle(model, poly) {
  const source = model?.meta?.field_source || "checkpoint metadata";
  return `${model?.display || model?.label || "Model"} supports ${modelFieldLabel(model)} (${source}); row is F${poly.prime}.`;
}

function resultCellsForLabel(label) {
  return $$(".result-cell").filter((cell) => cell.dataset.model === label);
}

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
  tabButtons:   $$(".tab-btn"),
  inferenceTab: $("#tab-inference"),
  playTab:      $("#tab-play"),
  playPoly:     $("#play-poly"),
  playModel:    $("#play-model"),
  playK:        $("#play-k"),
  playStart:    $("#play-start"),
  playForfeit:  $("#play-forfeit"),
  playStatus:   $("#play-status"),
  playActivePoly: $("#play-active-poly"),
  playAccCost:  $("#play-acc-cost"),
  playRefCost:  $("#play-ref-cost"),
  playFrontier: $("#play-frontier"),
  playCandidates: $("#play-candidates"),
  playDirect:   $("#play-direct"),
  playAgentSub: $("#play-agent-sub"),
  playAgentMove: $("#play-agent-move"),
  playAgentCircuit: $("#play-agent-circuit"),
  playCircuitSub: $("#play-circuit-sub"),
  playCurrentCircuit: $("#play-current-circuit"),
  playHistorySub: $("#play-history-sub"),
  playHistory: $("#play-history"),
};

function getMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function escapeHTML(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed: ${res.status}`);
  return data;
}

function setActiveTab(tab) {
  state.activeTab = tab;
  document.body.dataset.activeTab = tab;
  els.tabButtons.forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  els.inferenceTab.classList.toggle("active", tab === "inference");
  els.playTab.classList.toggle("active", tab === "play");
}

els.tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
});
setActiveTab("inference");

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
    renderPlayControls();
  } catch (e) {
    els.polyTables.innerHTML = `<p class="empty-state">Failed to load: ${e}</p>`;
    els.playStatus.textContent = `Failed to load: ${e}`;
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
    renderPlayControls();
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
      if (meta.field_label) metaParts.push(meta.field_label);
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
// Play tab
// ---------------------------------------------------------------------------
function preferredPlayModelLabel() {
  const sac = state.allModels.find((m) => m.label === "SAC");
  if (sac) return sac.label;
  const ckpt = state.allModels.find((m) => m.kind === "checkpoint");
  return ckpt ? ckpt.label : "heuristic";
}

function selectedPlayPoly() {
  const idx = Number(els.playPoly.value || 0);
  return state.polys.find((p) => p.index === idx) || state.polys[0];
}

function selectedPlayModel() {
  return modelByLabel(els.playModel.value) || state.allModels[0];
}

function renderPlayControls() {
  if (!els.playPoly || !els.playModel) return;
  const prevPoly = els.playPoly.value;
  const prevModel = els.playModel.value || preferredPlayModelLabel();

  els.playPoly.innerHTML = state.polys
    .map((p) => `<option value="${p.index}">F${p.prime} · ${escapeHTML(p.name)}</option>`)
    .join("");
  if (prevPoly && state.polys.some((p) => String(p.index) === prevPoly)) {
    els.playPoly.value = prevPoly;
  }

  els.playModel.innerHTML = state.allModels
    .map((m) => {
      const field = m.meta?.field_label ? ` · ${m.meta.field_label}` : "";
      return `<option value="${escapeHTML(m.label)}">${escapeHTML(m.display)}${escapeHTML(field)}</option>`;
    })
    .join("");
  if (state.allModels.some((m) => m.label === prevModel)) {
    els.playModel.value = prevModel;
  } else {
    els.playModel.value = preferredPlayModelLabel();
  }
  updatePlayCompatibility();
}

function updatePlayCompatibility() {
  const poly = selectedPlayPoly();
  const model = selectedPlayModel();
  if (!poly || !model) return;
  const supported = modelSupportsPrime(model, poly.prime);
  els.playStart.disabled = !supported;
  if (state.game && state.game.status === "active") return;
  els.playStatus.textContent = supported
    ? "Ready"
    : `${model.display} supports ${modelFieldLabel(model)}; selected row is F${poly.prime}.`;
}

function renderMath(root) {
  if (!root || !window.renderMathInElement) return;
  renderMathInElement(root, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}

function mountCircuit(container, circuit) {
  container.innerHTML = "";
  if (!circuit) {
    container.innerHTML = `<p class="empty-state-small">No circuit.</p>`;
    return;
  }
  if (window.renderCircuit) {
    const { el } = window.renderCircuit(circuit);
    container.appendChild(el);
  } else {
    container.textContent = "(circuit renderer unavailable)";
  }
}

function moveSummaryHTML(move, opts = {}) {
  if (!move) return `<p class="empty-state-small">No move.</p>`;
  const prior = move.prior == null ? "" : `<span>${(100 * move.prior).toFixed(1)}%</span>`;
  const badge = move.is_agent ? `<span class="move-badge">Agent</span>` : "";
  const cost = move.direct_cost == null ? "" : `<span>${move.direct_cost} ops</span>`;
  const selected = opts.selected ? " selected" : "";
  const agent = move.is_agent ? " agent-pick" : "";
  return `
    <div class="move-card${selected}${agent}">
      <div class="move-card-top">
        <strong>${escapeHTML(move.label)}</strong>
        <span class="move-meta">${badge}${prior}${cost}</span>
      </div>
      <div class="move-polys">
        <span>\\(${move.g.latex}\\)</span>
        <span class="move-plus">${move.kind === "factor" ? "factor" : "+"}</span>
        <span>\\(${move.h.latex}\\)</span>
      </div>
      <div class="move-source">${escapeHTML(move.source)}${move.score_hint != null ? ` · hint ${Number(move.score_hint).toFixed(2)}` : ""}</div>
    </div>
  `;
}

function statusLabel(game) {
  if (!game) return "Ready";
  if (game.status === "complete") return `Termination reached · cost ${game.acc_cost}`;
  if (game.status === "forfeit") return `Forfeited · committed cost ${game.acc_cost}`;
  return `Active · ${game.frontier.length} frontier item${game.frontier.length === 1 ? "" : "s"}`;
}

function renderGame(game) {
  state.game = game;
  const active = game.active;
  els.playStatus.textContent = statusLabel(game);
  els.playForfeit.disabled = game.status !== "active";
  els.playDirect.disabled = game.status !== "active" || !active;
  els.playAccCost.textContent = String(game.acc_cost);
  els.playRefCost.textContent = String(game.reference_cost);
  els.playActivePoly.innerHTML = active
    ? `Active: \\(${active.latex}\\) · F${active.prime} · support ${active.support}`
    : game.status === "complete" ? "No unresolved frontier." : "Session closed.";

  els.playFrontier.innerHTML = game.frontier.length
    ? game.frontier.map((p, i) => `
        <span class="frontier-pill${i === 0 ? " active" : ""}">
          <span>${i === 0 ? "active" : `#${i + 1}`}</span>
          <strong>\\(${p.latex}\\)</strong>
        </span>
      `).join("")
    : `<span class="frontier-pill done"><strong>Done</strong></span>`;

  if (game.status === "active" && game.candidates.length) {
    els.playCandidates.innerHTML = game.candidates.map((move) => `
      <button class="move-button" type="button" data-action-index="${move.index}">
        ${moveSummaryHTML(move)}
      </button>
    `).join("");
    $$(".move-button", els.playCandidates).forEach((btn) => {
      btn.addEventListener("click", () => stepGame("candidate", Number(btn.dataset.actionIndex)));
    });
  } else if (game.status === "active") {
    els.playCandidates.innerHTML = `<p class="empty-state-small">No split candidates for the active frontier item.</p>`;
  } else {
    els.playCandidates.innerHTML = `<p class="empty-state-small">${game.status === "complete" ? "Build complete." : "Game forfeited."}</p>`;
  }

  els.playAgentSub.textContent = game.policy_value == null
    ? "Direct construction."
    : `Policy value ${Number(game.policy_value).toFixed(3)}`;
  els.playAgentMove.innerHTML = moveSummaryHTML(game.agent_move, { selected: true });
  mountCircuit(els.playAgentCircuit, game.agent_preview?.circuit);

  els.playCircuitSub.textContent = `Rendered cost ${game.current_circuit_cost} · committed cost ${game.acc_cost}`;
  mountCircuit(els.playCurrentCircuit, game.current_circuit);

  els.playHistorySub.textContent = `${game.history.length} step${game.history.length === 1 ? "" : "s"}`;
  els.playHistory.innerHTML = game.history.length
    ? game.history.map((h) => `
      <li>
        <div class="hist-top">
          <span class="hist-actor ${h.actor}">${escapeHTML(h.actor)}</span>
          <strong>${escapeHTML(h.action?.label || h.action_kind)}</strong>
          <span class="hist-cost">${h.acc_cost_after} cost</span>
        </div>
        <div class="hist-poly">\\(${h.active.latex}\\)</div>
      </li>
    `).join("")
    : `<li class="empty-history">No steps yet.</li>`;

  renderMath(els.playTab);
}

async function startGame() {
  const poly = selectedPlayPoly();
  const model = selectedPlayModel();
  if (!poly || !model) return;
  els.playStart.disabled = true;
  els.playStatus.textContent = "Starting...";
  try {
    const game = await postJSON("/api/play/start", {
      poly_index: poly.index,
      model: model.label,
      k: Number(els.playK.value || 12),
    });
    renderGame(game);
  } catch (e) {
    els.playStatus.textContent = e.message;
  } finally {
    updatePlayCompatibility();
  }
}

async function stepGame(action, actionIndex = null) {
  if (!state.game?.session_id) return;
  els.playStatus.textContent = "Applying move...";
  const payload = { session_id: state.game.session_id, action };
  if (actionIndex != null) payload.action_index = actionIndex;
  try {
    renderGame(await postJSON("/api/play/step", payload));
  } catch (e) {
    els.playStatus.textContent = e.message;
  }
}

async function forfeitGame() {
  if (!state.game?.session_id) return;
  try {
    renderGame(await postJSON("/api/play/forfeit", {
      session_id: state.game.session_id,
    }));
  } catch (e) {
    els.playStatus.textContent = e.message;
  }
}

els.playPoly.addEventListener("change", updatePlayCompatibility);
els.playModel.addEventListener("change", updatePlayCompatibility);
els.playStart.addEventListener("click", startGame);
els.playDirect.addEventListener("click", () => stepGame("direct"));
els.playForfeit.addEventListener("click", forfeitGame);

// ---------------------------------------------------------------------------
// Per-model summary cards
// ---------------------------------------------------------------------------
function renderModelCards() {
  els.modelCards.innerHTML = "";
  for (const m of activeModels()) {
    const domId = modelDomId(m.label);
    const card = document.createElement("div");
    card.className = `model-card kind-${m.kind}`;
    card.id = `card-${domId}`;
    card.innerHTML = `
      <div class="kind-tag">${m.kind === "baseline" ? "Baseline" : "Checkpoint"}</div>
      <div class="display-name">${m.display}</div>
      <div class="info-line" id="info-${domId}">&nbsp;</div>
      <div class="progress-track">
        <div class="progress-fill" id="bar-${domId}"></div>
      </div>
      <div class="stat-row">
        <div class="stat">
          <span class="stat-value" id="stat-greedy-${domId}">–</span>
          <span class="stat-label">Greedy target</span>
        </div>
        <div class="stat">
          <span class="stat-value" id="stat-search-${domId}">–</span>
          <span class="stat-label">Search target</span>
        </div>
        <div class="stat">
          <span class="stat-value" id="stat-time-${domId}">–</span>
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
          <th class="col-bmin" title="Reference gate count (binary +/×; only the constant 1 is free; reuse is free). Hover for the reference circuit.">Ref op</th>
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
        .map((m) => {
          if (!modelSupportsPrime(m, p.prime)) {
            return `<td class="result-cell skipped" data-model="${m.label}" data-idx="${p.index}" title="${unsupportedTitle(m, p)}">${unsupportedText(m)}</td>`;
          }
          return `<td class="result-cell pending" data-model="${m.label}" data-idx="${p.index}">–</td>`;
        })
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
  return $$(".result-cell")
    .find((cell) => cell.dataset.model === label && cell.dataset.idx === String(idx));
}

// Badge shown next to a cost: ✓ at reference target, otherwise +N over target.
function deltaBadge(delta) {
  return delta <= 0 ? "✓" : `+${delta}`;
}

function updateSkippedCell(label, idx, data) {
  const cell = getCell(label, idx);
  const model = modelByLabel(label);
  if (!cell || !model) return;
  cell.className = "result-cell skipped";
  cell.textContent = unsupportedText(model);
  cell.title = data.reason || unsupportedTitle(model, state.polys[idx]);
}

function updateCell(label, idx, result) {
  const cell = getCell(label, idx);
  if (!cell) return;
  const mode = getMode();
  const opt = result.optimal;
  const g = { cost: result.greedy_cost, tier: result.greedy_tier };
  const s = { cost: result.search_cost, tier: result.search_tier };
  const desc = (c) => (c - opt <= 0 ? "matches reference" : `${c - opt} op${c - opt > 1 ? "s" : ""} from reference`);
  const tt = [
    `Reference: ${opt}`,
    `Greedy: ${g.cost} (${desc(g.cost)})`,
    `Search: ${s.cost} (${desc(s.cost)})`,
    `Latency: ${result.elapsed_ms}ms · nodes=${result.node_expansions}, hits=${result.transposition_hits}`,
  ];
  if (result.budget_capped) {
    tt.push(`Capped budget: sims=${result.search_sims_used}, k=${result.k_candidates_used}, depth=${result.search_max_depth}`);
  }
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
  const model = modelByLabel(label);
  const total = state.polys.filter((p) => modelSupportsPrime(model, p.prime)).length;
  const gOk = results.filter((r) => r.greedy_tier === 0).length;
  const sOk = results.filter((r) => r.search_tier === 0).length;
  const domId = modelDomId(label);
  const gEl = $(`#stat-greedy-${domId}`);
  const sEl = $(`#stat-search-${domId}`);
  const bar = $(`#bar-${domId}`);
  if (gEl) gEl.textContent = total ? `${gOk}/${total}` : "–";
  if (sEl) sEl.textContent = total ? `${sOk}/${total}` : "–";
  if (bar) bar.style.width = total ? `${(100 * results.length) / total}%` : "0%";
}

function resetEvaluationUI() {
  $$(".result-cell").forEach((cell) => {
    const model = modelByLabel(cell.dataset.model);
    const poly = state.polys[Number(cell.dataset.idx)];
    if (model && poly && !modelSupportsPrime(model, poly.prime)) {
      cell.className = "result-cell skipped";
      cell.textContent = unsupportedText(model);
      cell.title = unsupportedTitle(model, poly);
    } else {
      cell.className = "result-cell pending";
      cell.textContent = "–";
      cell.title = "";
    }
  });
  for (const m of activeModels()) {
    const domId = modelDomId(m.label);
    const g = $(`#stat-greedy-${domId}`);
    const s = $(`#stat-search-${domId}`);
    const t = $(`#stat-time-${domId}`);
    const i = $(`#info-${domId}`);
    const b = $(`#bar-${domId}`);
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
  if (info.description) {
    return info.field_label ? `${info.description} · field ${info.field_label}` : info.description;
  }
  const parts = [];
  if (info.cycle !== undefined && info.cycle !== null) parts.push(`cycle ${info.cycle}`);
  if (info.field_label) parts.push(`field ${info.field_label}`);
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
    const info = $(`#info-${modelDomId(data.label)}`);
    if (info) info.textContent = formatInfo(data.info);
    resultCellsForLabel(data.label).forEach((c) =>
      { if (!c.classList.contains("skipped")) c.classList.add("running"); }
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

  es.addEventListener("skipped", (e) => {
    const data = JSON.parse(e.data);
    updateSkippedCell(data.label, data.poly_index, data);
  });

  es.addEventListener("model-done", (e) => {
    const data = JSON.parse(e.data);
    const domId = modelDomId(data.label);
    const t = $(`#stat-time-${domId}`);
    const b = $(`#bar-${domId}`);
    if (t) t.textContent = `${data.elapsed_sec}s`;
    if (b) b.style.width = "100%";
    resultCellsForLabel(data.label).forEach((c) =>
      c.classList.remove("running")
    );
  });

  es.addEventListener("model-error", (e) => {
    const data = JSON.parse(e.data);
    const info = $(`#info-${modelDomId(data.label)}`);
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
  showCircPopover(cell, poly.optimal_circuit, "Reference circuit",
                  `Circuit cost: ${poly.optimal_circuit_cost}`);
}

// Use event delegation on the entire poly-tables section for efficiency.
els.polyTables.addEventListener("mouseover", (e) => {
  const rc = e.target.closest(".result-cell");
  const bc = e.target.closest(".bmin-cell");
  if (rc && !rc.classList.contains("pending") && !rc.classList.contains("running") && !rc.classList.contains("skipped")) {
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
