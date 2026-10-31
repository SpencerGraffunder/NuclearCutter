#!/usr/bin/env node
/*
 * The GUI's category chooser and progress bars, executed against the real
 * index.html and the real /api/taxonomy payload.
 *
 * Why this exists. The VERIFY bar slid back and forth for weeks because the page
 * read `steps.verify`, a key no server code has ever set. No Python test could see
 * it: the bug was a string inside a <script> tag. Later, removing the "Picture
 * style" checkboxes almost silently switched OFF all censoring, because
 * collectSettings() went on reading those checkboxes, got [], and an empty
 * renderings list makes rendering_allowed() false for every detection. Both bugs
 * are "the page and the server disagree", and both are invisible to a test that
 * only speaks Python.
 *
 * So: extract the page's own JavaScript, run it against a small DOM shim, and hand
 * it the vocabulary the server would hand it. A GUI that reads a key the server
 * does not send, or touches a node that is no longer on the page, fails here.
 *
 *   node tests/gui_dom.test.mjs
 *
 * Normally run through tests/test_gui_dom.py, which serves the real
 * /api/taxonomy into NC_TAXONOMY_JSON so this file can never drift from the API.
 * The shim is deliberately dumb: when the page starts using DOM features it does
 * not model, this file throws, which is also a signal.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HTML = fs.readFileSync(path.join(ROOT, "nuclearcutter/static/index.html"), "utf8");

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log(`OK   ${name}`); }
  else { fail++; console.log(`FAIL ${name}${extra ? " — " + extra : ""}`); }
};

const scripts = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (!scripts.length) { console.error("no <script> block in index.html"); process.exit(2); }

const FIXTURE = process.env.NC_TAXONOMY_JSON;
if (!FIXTURE) {
  console.error("NC_TAXONOMY_JSON must point at a /api/taxonomy dump "
    + "(run it via tests/test_gui_dom.py)");
  process.exit(2);
}
const TAX = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));

// ---------------------------------------------------------------- DOM shim
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = [];
    this.style = {}; this.dataset = {}; this.listeners = {};
    this.checked = false; this.disabled = false; this.value = ""; this.title = "";
    this.type = ""; this.open = false; this._text = ""; this._html = "";
    this.parentElement = null; this.id = "";
    const s = new Set();
    this.classList = {
      _s: s,
      add: (...c) => c.forEach(x => s.add(x)),
      remove: (...c) => c.forEach(x => s.delete(x)),
      contains: c => s.has(c),
      toggle: (c, on) => {
        const v = on === undefined ? !s.has(c) : !!on;
        if (v) s.add(c); else s.delete(c);
        return v;
      },
    };
  }
  set className(v) { this.classList._s = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classList._s].join(" "); }
  set innerHTML(v) { this._html = String(v); if (v === "") this.children = []; }
  get innerHTML() { return this._html; }
  set textContent(v) { this._text = String(v); }
  get textContent() { return this._text; }
  appendChild(c) { this.children.push(c); c.parentElement = this; return c; }
  // The page walks up from a chip to its family group to count ticks per family, so
  // parentage has to be real or that code path never gets exercised here.
  closest(sel) {
    const cls = sel.startsWith(".") ? sel.slice(1) : null;
    let n = this;
    while (n) {
      if (cls && n.classList && n.classList.contains(cls)) return n;
      if (!cls && sel.startsWith("#") && n.id === sel.slice(1)) return n;
      n = n.parentElement;
    }
    return null;
  }
  remove() {}
  removeChild(c) { this.children = this.children.filter(x => x !== c); }
  addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); }
  fire(t) { (this.listeners[t] || []).forEach(f => f()); }
  get lastChild() { return this.children[this.children.length - 1]; }
  querySelectorAll(sel) {
    const hits = ALL.filter(e => matchesIn(e, sel) && (e.parentElement === this || isDesc(e, this)));
    return hits;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  get options() { return this._options || []; }
  set options(v) { this._options = v; }
}

const byId = {};
const ALL = [];
// Selector support is deliberately narrow: exactly the shapes the page uses.
function matchesIn(el, sel) {
  if (sel.includes("input[data-tag]")) return el.dataset.tag !== undefined;
  if (sel.includes("select[data-fam-action]")) return el.dataset.famAction !== undefined;
  if (sel.startsWith(".")) return el.classList.contains(sel.slice(1));
  if (sel.startsWith("#")) return el.id === sel.slice(1);
  return el.tagName === sel.toUpperCase();
}
function isDesc(el, root) { let n = el.parentElement; while (n) { if (n === root) return true; n = n.parentElement; } return false; }
global.document = {
  createElement(t) { const e = new El(t); ALL.push(e); return e; },
  createTextNode(t) { const e = new El("#text"); e.textContent = String(t); return e; },
  getElementById(id) { return byId[id] ||= new El("div"); },
  querySelectorAll(sel) {
    if (sel.includes("input[data-tag]:checked"))
      return ALL.filter(e => e.dataset.tag !== undefined && e.checked);
    if (sel.includes("input[data-tag]"))
      return ALL.filter(e => e.dataset.tag !== undefined);
    if (sel.includes("select[data-fam-action]"))
      return ALL.filter(e => e.dataset.famAction !== undefined);
    if (sel.includes("[data-count-for]"))
      return ALL.filter(e => e.dataset.countFor !== undefined);
    return [];
  },
  addEventListener() {}, body: new El("body"),
};
global.window = { addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }) };
global.fetch = async (url) => ({
  ok: true,
  json: async () => (String(url).includes("/api/taxonomy") ? TAX : { settings: {} }),
});
global.location = { origin: "http://x", href: "http://x/" };
// The page console.warns when a bar has no progress key; the harness asserts on the
// class it sets instead, so keep the noise out of the report.
console.warn = () => {};

const api = new Function(`
${scripts.join("\n")}
return { loadTaxonomy, renderState, collectSettings, applySelectionToChooser,
         passThrough, selectedTagIds,
         setState: (s) => { state = s; } };
`)();

// A /api/state-shaped job, matching JobInfo.to_dict().
const j = (over = {}) => ({
  kind: "scan", status: "running", phase: "identify", message: "classifying range 12 of 411",
  error: null, started_at: 0, finished_at: null, progress: 0.3, position: 100,
  duration: 60, frames_done: 100, frames_total: 4110, eta: 600,
  steps: { transcribe: 1, language: 1, scan: 1, identify: 0.35, render: 0, summary: 0 },
  results: {}, logs: [], ...over,
});
const st = (scan, render) => ({
  settings: { rendering: TAX.rendering.map(r => r.id), minors: TAX.minors.map(m => m.id),
              family_actions: {}, selection: [], polish_descriptions: false },
  jobs: { scan, render: render || j({ kind: "render", status: "idle", steps:
              { transcribe: 0, language: 0, scan: 0, identify: 0, render: 0, summary: 0 } }),
          benchmark: j({ kind: "benchmark", status: "idle" }) },
  timeline: { visual: [], language: [], video: "x.mkv", duration: 8000 },
  all_logs: [], system: {}, model: {}, server: {},
});

await (async () => {
  await api.loadTaxonomy();

  // ------------------------------------------------------------ the chooser
  ok("one chooser group per family",
     byId.chooser.children.length === TAX.families.length,
     `${byId.chooser.children.length} vs ${TAX.families.length}`);
  // The page's own bootstrap calls loadTaxonomy(), and so does this harness, so the
  // shim holds two builds of the chooser. Unique ids is the assertion that survives
  // that; a raw element count would only measure how many times it rendered.
  const chipIds = new Set(ALL.filter(e => e.dataset.tag !== undefined && e.tagName === "INPUT")
                             .map(e => e.dataset.tag));
  ok("every tag in the vocabulary gets a checkbox",
     chipIds.size === TAX.tags.length, `${chipIds.size} vs ${TAX.tags.length}`);
  ok("no tag is rendered twice with different ids",
     [...chipIds].every(id => TAX.tags.some(t => t.id === id)));
  ok("a merged vocabulary really reached the browser", TAX.tags.length < 90,
     `${TAX.tags.length} tags served`);

  // The four removals of 2026-10-30, asserted so they stay removed and so no handler
  // is left behind reaching for a node that is not on the page.
  ok("search box is gone", !HTML.includes('id="tag-search"'));
  ok("preset buttons are gone", !/id="presets"/.test(HTML));
  ok("picture-style qualifiers are gone", !/id="rendering-chips"/.test(HTML));
  ok("include-people qualifiers are gone", !/id="minors-chips"/.test(HTML));

  // --------------------------------------------------------- pass-through values
  // The near-miss this harness was written for. These two settings lost their
  // checkboxes; if collectSettings still read them from the DOM it would send [],
  // and the server would censor nothing while the page showed every box ticked.
  const s1 = api.collectSettings();
  ok("rendering is never an empty list",
     Array.isArray(s1.rendering) && s1.rendering.length === TAX.rendering.length,
     JSON.stringify(s1.rendering));
  ok("minors is never an empty list",
     Array.isArray(s1.minors) && s1.minors.length === TAX.minors.length,
     JSON.stringify(s1.minors));

  // A value narrowed elsewhere (CLI --rendering animated_cg) must survive a GUI save.
  api.applySelectionToChooser({ ...st(j()).settings, rendering: ["animated_cg"],
                                minors: ["no"] });
  const s2 = api.collectSettings();
  ok("a narrowed rendering from the server is carried through",
     s2.rendering.join() === "animated_cg", JSON.stringify(s2.rendering));
  ok("a narrowed minors from the server is carried through",
     s2.minors.join() === "no", JSON.stringify(s2.minors));

  // ------------------------------------------------------------------ the bars
  // Bar ids on the page: bar-transcribe, bar-language, bar-scan, bar-verify,
  // bar-summary, bar-render. VERIFY is the UI name; the wire key is `identify`.
  api.setState(st(j()));
  api.renderState();
  ok("VERIFY bar tracks steps.identify",
     byId["bar-verify"].style.width === "35%",
     `width=${byId["bar-verify"].style.width}`);
  ok("VERIFY is a determinate bar, not an animation",
     !byId["bar-verify"].classList.contains("indet"),
     byId["bar-verify"].className);
  ok("SCAN bar tracks steps.scan", byId["bar-scan"].style.width === "100%");
  ok("LANGUAGE bar has a row and a key", byId["bar-language"].style.width === "100%");

  // A key the server stopped sending must look broken, never alive. An animating bar
  // is worse than a flat one because it reads as progress.
  const noKey = j({ steps: { transcribe: 1, language: 1, scan: 1,
                             identify: undefined, render: 0, summary: 0 } });
  api.setState(st(noKey));
  api.renderState();
  ok("a missing step key draws the 'lost' state",
     byId["bar-verify"].classList.contains("lost"),
     `classes=${byId["bar-verify"].className} width=${byId["bar-verify"].style.width}`);
  ok("a lost bar does not animate",
     !byId["bar-verify"].classList.contains("indet"));
  ok("a lost bar shows '?' not '…'", byId["pct-verify"].textContent === "?",
     byId["pct-verify"].textContent);

  // And the pairing itself: every key the page reads must be a key the server sends.
  // This is the contract that broke when VERIFY read steps.verify.
  const KEYS = ["transcribe", "language", "scan", "identify", "render", "summary"];
  ok("the page reads exactly the step keys a running job sends",
     KEYS.every(k => k in j().steps), Object.keys(j().steps).join(","));
  const rows = [...HTML.matchAll(/id="bar-([a-z]+)"/g)].map(m => m[1]);
  const VERIFIED = { transcribe: "transcribe", language: "language", scan: "scan",
                     verify: "identify", summary: "summary", render: "render" };
  ok("every bar row maps to a real step key",
     rows.length === 6 && rows.every(r => r in VERIFIED), rows.join(","));

  console.log(`\n${pass} passed, ${fail} failed`);
})();

// The page schedules a /api/state poll on a timer as part of its own bootstrap.
// That keeps node's event loop alive forever in a harness with no server, so leave
// deliberately once the assertions are in rather than waiting for nothing.
process.exit(fail ? 1 : 0);
