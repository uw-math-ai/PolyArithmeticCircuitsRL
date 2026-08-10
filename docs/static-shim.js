// static-shim.js — makes the live inference demo run on GitHub Pages.
//
// The interactive demo talks to a Flask backend at /api/*. GitHub Pages serves
// only static files, so this shim intercepts those calls and replays the
// snapshot JSON baked by scripts/build_static_snapshot.py into data/*.json.
//
// It monkeypatches window.fetch (for /api/test-suite and /api/models) and
// window.EventSource (for the /api/evaluate SSE stream). app.js itself is
// unmodified — it cannot tell it is talking to a static snapshot.
//
// Live human-vs-agent play (/api/play/*) needs live server state and is not
// available in the static snapshot; those calls return a friendly 501.
(function () {
  "use strict";

  const originalFetch = window.fetch.bind(window);

  function jsonResponse(obj, status = 200) {
    return new Response(JSON.stringify(obj), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }

  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.startsWith("/api/test-suite")) {
      return originalFetch("data/test-suite.json");
    }
    if (url.startsWith("/api/models")) {
      return originalFetch("data/models.json");
    }
    if (url.startsWith("/api/play/")) {
      return Promise.resolve(
        jsonResponse(
          { error: "Live play is disabled in the static snapshot." },
          501
        )
      );
    }
    return originalFetch(input, init);
  };

  // Minimal EventSource replacement that replays the baked evaluate stream.
  // app.js registers listeners synchronously right after construction; the
  // real fetch of data/evaluate.json is async, so listeners are always in
  // place before the first event is dispatched.
  class SnapshotEventSource {
    constructor(url) {
      this._listeners = {};
      this.onerror = null;
      this._closed = false;
      this._run(url);
    }

    addEventListener(type, cb) {
      (this._listeners[type] || (this._listeners[type] = [])).push(cb);
    }

    close() {
      this._closed = true;
    }

    _emit(type, data) {
      if (this._closed) return;
      const evt = { data: JSON.stringify(data) };
      (this._listeners[type] || []).forEach((cb) => cb(evt));
    }

    async _run(url) {
      let requested;
      try {
        const u = new URL(url, window.location.href);
        requested = new Set(u.searchParams.getAll("models"));
      } catch (_e) {
        requested = new Set();
      }
      let snap;
      try {
        snap = await originalFetch("data/evaluate.json").then((r) => r.json());
      } catch (e) {
        if (this.onerror) this.onerror(e);
        return;
      }
      for (const ev of snap.events || []) {
        if (this._closed) return;
        // Model-scoped events carry a label; drop those the caller did not
        // select. Session-level events (no label) always pass through.
        const label = ev.data && ev.data.label;
        if (label && requested.size && !requested.has(label)) continue;
        this._emit(ev.event, ev.data);
        // Tiny stagger so the table visibly fills in, mimicking streaming.
        await new Promise((r) => setTimeout(r, 12));
      }
    }
  }

  window.EventSource = SnapshotEventSource;
})();
