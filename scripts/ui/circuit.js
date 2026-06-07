// circuit.js — renders an arithmetic-circuit tree/DAG as a clean SVG node-link graph.
//
// A circuit node is { op, latex, children?, exp?, scalar?, cost?, id?, ref? }.
// renderCircuit(node) returns a DOM element: an SVG edge layer with absolutely
// positioned HTML node chips on top (so KaTeX renders normally inside them).
//
// Before layout, renderCircuit normalizes the tree so only + and × nodes appear:
//   pow(f, k)          → repeated-squaring chain of × nodes  (f×f, (f×f)×(f×f), …)
//   mul(scalar=k, [f]) → binary-doubling chain of + nodes    (f+f, (f+f)+(f+f), …)
//
// Shared subexpressions (a value reused in several places, e.g. x² in
// x²y²+x²+y²) are expressed with id/ref: define a node once with {"id": "x2",…}
// and reference it elsewhere with {"op": "ref", "ref": "x2"}.  Such a node is
// drawn once with fan-out edges to every consumer — a DAG, not a tree.  Pure
// trees use the original tidy-tree layout unchanged.

(function (global) {
  const LEVEL_H = 84;   // vertical gap between tree levels
  const LEAF_W = 82;    // horizontal slot width per leaf
  const NODE_R = 19;    // half-height used for edge attach points
  const PAD = 26;       // canvas padding

  // ── id/ref linking ──────────────────────────────────────────────────────────

  function deepCopy(n) { return JSON.parse(JSON.stringify(n)); }

  // Replace every {op:"ref", ref:id} child with the actual defining node object,
  // turning the JSON into a real in-memory DAG (shared object identity).
  function linkRefs(root) {
    const defs = {};
    (function collect(n) {
      if (!n || typeof n !== "object") return;
      if (n.id != null) defs[n.id] = n;
      (n.children || []).forEach(collect);
    })(root);
    const seen = new Set();
    (function relink(n) {
      if (!n || seen.has(n)) return;
      seen.add(n);
      if (n.children) {
        n.children = n.children.map((c) =>
          c && c.op === "ref" ? defs[c.ref] : c
        );
        n.children.forEach(relink);
      }
    })(root);
    return root;
  }

  // ── Normalization helpers ─────────────────────────────────────────────────

  // pow(base, exp) → chain of × nodes using repeated squaring.
  // Even steps use a _share node (single child, two edges drawn) to avoid
  // duplicating the subtree — the child is computed once, used twice.
  function expandPow(base, exp) {
    if (exp <= 1) return base;
    if (exp === 2) return { op: "mul", latex: "×", _share: true, children: [base] };
    const half = Math.floor(exp / 2);
    const h = expandPow(base, half);
    if (exp % 2 === 0) {
      return { op: "mul", latex: "×", _share: true, children: [h] };
    }
    // odd exponent: pow(f, exp-1) × f — f is shared; draw a DAG arc on the right
    return { op: "mul", latex: "×", _extra_edges: [base], children: [expandPow(base, exp - 1)] };
  }

  // k × product → binary-doubling chain of + nodes.
  // Doubling steps use _share (single child, two edges) to avoid subtree copies.
  function expandScalar(product, k) {
    if (k <= 1) return product;
    if (k === 2) return { op: "add", latex: "+", _share: true, children: [product] };
    const half = Math.floor(k / 2);
    const h = expandScalar(product, half);
    if (k % 2 === 0) {
      return { op: "add", latex: "+", _share: true, children: [h] };
    }
    return { op: "add", latex: "+", _extra_edges: [product], children: [expandScalar(product, k - 1)] };
  }

  // Build a constant c (≥2) as a binary-doubling DAG of 1s, e.g. 4 = (1+1)+(1+1)
  // reusing the shared 2.  Free constant is only 1 — every other must be made.
  // Cached per value so a constant is built once and shared across the diagram.
  function buildConst(v, cc) {
    if (v <= 1) return { op: "leaf", latex: String(v) };
    if (cc.has(v)) return cc.get(v);
    const a = Math.floor(v / 2);
    const b = v - a;
    const node = { op: "add", latex: String(v), children: [buildConst(a, cc), buildConst(b, cc)] };
    cc.set(v, node);
    return node;
  }

  // Recursively rewrite the tree to contain only +, ×, leaf, direct nodes.
  // Memoized on node identity so shared (linked) nodes normalize once and stay
  // shared in the output — preserving the DAG structure.  ``cc`` caches built
  // integer constants so they are shared across the whole diagram.
  function normalize(node, memo, cc) {
    if (!node) return node;
    if (memo.has(node)) return memo.get(node);
    if (node.op === "leaf") {
      // A constant leaf >1 must be constructed from 1s (only 1 is free).
      const m = /^\d+$/.test(node.latex) ? parseInt(node.latex, 10) : null;
      const result = m !== null && m >= 2 ? buildConst(m, cc) : node;
      memo.set(node, result);
      return result;
    }
    if (node.op === "direct") {
      memo.set(node, node);
      return node;
    }

    const kids = (node.children || []).map((c) => normalize(c, memo, cc));
    let result;

    if (node.op === "pow") {
      const base = kids[0] || { op: "leaf", latex: "1" };
      result = expandPow(base, node.exp || 1);
    } else if (node.op === "mul") {
      let product;
      if (kids.length === 0) {
        product = { op: "leaf", latex: "1" };
      } else if (kids.length === 1) {
        product = kids[0];
      } else {
        product = { op: "mul", latex: node.latex, children: kids };
      }
      const k = node.scalar;
      result = !k || k === 1 ? product : expandScalar(product, k);
    } else {
      // add — propagate normalized children
      if (kids.length === 0) result = { op: "leaf", latex: "0" };
      else if (kids.length === 1) result = kids[0];
      else result = { op: node.op, latex: node.latex, children: kids };
    }

    memo.set(node, result);
    return result;
  }

  // ── Rendering helpers ──────────────────────────────────────────────────────

  function katex(latex) {
    if (global.katex) {
      try {
        return global.katex.renderToString(latex, { throwOnError: false });
      } catch (e) {
        /* fall through */
      }
    }
    return `<span>${latex}</span>`;
  }

  function nodeInner(n) {
    switch (n.op) {
      case "leaf":
        return `<span class="circ-val">${katex(n.latex)}</span>`;
      case "add":
        return `<span class="circ-sym">+</span>`;
      case "mul":
        return `<span class="circ-sym">×</span>`;
      case "direct":
        return `
          <div class="circ-direct-inner">
            <span class="circ-val">${katex(n.latex)}</span>
            <span class="circ-direct-tag">direct build · ${n.cost} ops</span>
          </div>`;
      default:
        return `<span class="circ-val">${katex(n.latex || "")}</span>`;
    }
  }

  // Cubic bezier with two control points.
  function bez(x1, y1, c1x, c1y, c2x, c2y, x2, y2) {
    return `<path class="circ-edge" d="M${x1},${y1} C${c1x},${c1y} ${c2x},${c2y} ${x2},${y2}" />`;
  }

  // ── Layouts (each sets n._px / n._py pixel centers on every node) ───────────

  // Tidy-tree layout for pure trees (no shared nodes). Unchanged behaviour.
  function layoutTree(root) {
    let leafCursor = 0;
    let maxDepth = 0;
    (function assign(node, depth) {
      node._d = depth;
      maxDepth = Math.max(maxDepth, depth);
      const kids = node.children || [];
      if (kids.length === 0) {
        node._slot = leafCursor;
        leafCursor += 1;
      } else {
        kids.forEach((k) => assign(k, depth + 1));
        node._slot = (kids[0]._slot + kids[kids.length - 1]._slot) / 2;
      }
    })(root, 0);

    const leaves = Math.max(1, leafCursor);
    let dagExtra = 0;
    (function checkDag(n) {
      if (n._extra_edges && n._extra_edges.length) dagExtra = 56;
      (n.children || []).forEach(checkDag);
    })(root);
    const svgW = (leaves - 1) * LEAF_W + LEAF_W + PAD * 2 + dagExtra;
    const svgH = maxDepth * LEVEL_H + NODE_R * 2 + PAD * 2;

    (function setpx(n) {
      n._px = n._slot * LEAF_W + LEAF_W / 2 + PAD;
      n._py = n._d * LEVEL_H + NODE_R + PAD;
      (n.children || []).forEach(setpx);
    })(root);

    return { svgW, svgH };
  }

  // Layered layout for DAGs (a node may have several parents). Depth is the
  // longest path from the root, so a shared node sinks below all its parents;
  // x positions are barycenters of children, de-overlapped within each layer.
  function layoutDag(root, uniq) {
    // Longest-path depth from root.
    const topo = [];
    const ts = new Set();
    (function dfs(n) {
      if (ts.has(n)) return;
      ts.add(n);
      (n.children || []).forEach(dfs);
      topo.push(n);
    })(root);
    const order = topo.slice().reverse(); // parents before children

    const depth = new Map();
    uniq.forEach((n) => depth.set(n, 0));
    for (const n of order) {
      const d = depth.get(n) || 0;
      for (const c of n.children || []) {
        depth.set(c, Math.max(depth.get(c) || 0, d + 1));
      }
    }
    let maxDepth = 0;
    depth.forEach((v) => (maxDepth = Math.max(maxDepth, v)));

    // DFS first-visit index — stable tiebreak for ordering within a layer.
    const dfsIndex = new Map();
    let di = 0;
    (function dfo(n) {
      if (dfsIndex.has(n)) return;
      dfsIndex.set(n, di++);
      (n.children || []).forEach(dfo);
    })(root);

    const layers = [];
    for (let d = 0; d <= maxDepth; d++) layers.push([]);
    uniq.forEach((n) => layers[depth.get(n)].push(n));

    // Assign slots bottom-up: leaves get sequential slots, parents barycenter.
    const slot = new Map();
    let nextLeafSlot = 0;
    for (let d = maxDepth; d >= 0; d--) {
      const layer = layers[d]
        .slice()
        .sort((a, b) => dfsIndex.get(a) - dfsIndex.get(b));
      for (const n of layer) {
        const kids = n.children || [];
        if (kids.length === 0) {
          slot.set(n, nextLeafSlot++);
        } else {
          let s = 0;
          for (const c of kids) s += slot.get(c);
          slot.set(n, s / kids.length);
        }
      }
      // De-overlap within the layer, preserving left-to-right order.
      const sorted = layer
        .slice()
        .sort((a, b) => slot.get(a) - slot.get(b) || dfsIndex.get(a) - dfsIndex.get(b));
      for (let i = 1; i < sorted.length; i++) {
        if (slot.get(sorted[i]) < slot.get(sorted[i - 1]) + 1) {
          slot.set(sorted[i], slot.get(sorted[i - 1]) + 1);
        }
      }
    }

    let maxSlot = 0;
    slot.forEach((v) => (maxSlot = Math.max(maxSlot, v)));
    const svgW = maxSlot * LEAF_W + LEAF_W + PAD * 2;
    const svgH = maxDepth * LEVEL_H + NODE_R * 2 + PAD * 2;

    uniq.forEach((n) => {
      n._px = slot.get(n) * LEAF_W + LEAF_W / 2 + PAD;
      n._py = depth.get(n) * LEVEL_H + NODE_R + PAD;
    });

    return { svgW, svgH };
  }

  function renderCircuit(rootIn) {
    // Work on a private copy so we never mutate the stored circuit.
    let root = linkRefs(deepCopy(rootIn));
    root = normalize(root, new Map(), new Map());

    // Unique node set (a DAG node is listed once even with several parents).
    const uniq = [];
    const useen = new Set();
    (function visit(n) {
      if (useen.has(n)) return;
      useen.add(n);
      uniq.push(n);
      (n.children || []).forEach(visit);
    })(root);

    // DAG iff some node is the child of more than one parent.
    const parentCount = new Map();
    for (const n of uniq) {
      for (const c of n.children || []) {
        parentCount.set(c, (parentCount.get(c) || 0) + 1);
      }
    }
    const isDag = uniq.some((n) => (parentCount.get(n) || 0) > 1);

    const { svgW, svgH } = isDag ? layoutDag(root, uniq) : layoutTree(root);

    // ---- edges -----------------------------------------------------------
    let edges = "";
    for (const n of uniq) {
      if (n._share) {
        // single child, drawn with TWO edges — "node × itself" / "node + itself"
        const c = n.children[0];
        const x1 = n._px, y1 = n._py + NODE_R;
        const x2 = c._px, y2 = c._py - NODE_R;
        const my = (y1 + y2) / 2;
        const off = 7;
        edges += bez(x1 - off, y1, x1 - off, my, x2, my, x2, y2);
        edges += bez(x1 + off, y1, x1 + off, my, x2, my, x2, y2);
      } else {
        // Group children by identity: a child referenced k times (e.g. a×a, or
        // 4 = 2+2 reusing the shared 2) gets k edges, fanned out so they read as
        // distinct inputs rather than overlapping.
        const groups = new Map();
        for (const c of n.children || []) groups.set(c, (groups.get(c) || 0) + 1);
        for (const [c, cnt] of groups) {
          const y1 = n._py + NODE_R;
          const x2 = c._px, y2 = c._py - NODE_R;
          const my = (y1 + y2) / 2;
          for (let t = 0; t < cnt; t++) {
            const off = cnt === 1 ? 0 : (t - (cnt - 1) / 2) * 14;
            const x1 = n._px + off;
            edges += bez(x1, y1, x1, my, x2, my, x2, y2);
          }
        }
      }
      // DAG wrap-around arcs (e.g. the third factor of an odd power).
      for (const t of n._extra_edges || []) {
        const x1 = n._px + NODE_R + 4, y1 = n._py;
        const x2 = t._px + NODE_R + 4, y2 = t._py;
        const rx = Math.max(x1, x2) + 32;
        edges += bez(x1, y1, rx, y1, rx, y2, x2, y2);
      }
    }

    // ---- assemble --------------------------------------------------------
    const wrap = document.createElement("div");
    wrap.className = "circ-wrap";
    wrap.style.width = svgW + "px";
    wrap.style.height = svgH + "px";
    wrap.innerHTML = `<svg class="circ-svg" width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">${edges}</svg>`;

    for (const n of uniq) {
      const el = document.createElement("div");
      el.className = `circ-node op-${n.op}`;
      el.style.left = n._px + "px";
      el.style.top = n._py + "px";
      if (n.latex) el.title = n.latex.replace(/[{}]/g, "");
      el.innerHTML = nodeInner(n);
      wrap.appendChild(el);
    }
    return { el: wrap, width: svgW, height: svgH };
  }

  global.renderCircuit = renderCircuit;
})(window);
