// Behavioural test for web/static/ask.js history loading (roborev #6470,
// #6472): runs the real script against a tiny DOM stand-in and a fetch
// whose response we release by hand, so the race is exercised, not grepped.
// Usage: node tests/js/ask_history.test.mjs [path/to/ask.js]
import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const src = readFileSync(process.argv[2] || new URL("../../web/static/ask.js", import.meta.url), "utf8");

class El {
  constructor(id = "") {
    this.id = id; this.children = []; this.parent = null; this.dataset = {};
    this.className = ""; this.textContent = ""; this.innerHTML = ""; this.value = "";
    this.disabled = false; this.checked = false; this.listeners = {};
    const self = this;
    this.classList = { add(c) { self.className = (self.className + " " + c).trim(); } };
  }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  setAttribute() {} removeAttribute() {} focus() {}
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  prepend(c) { c.parent = this; this.children.unshift(c); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((x) => x !== this); }
  querySelectorAll(sel) {
    const cls = sel.replace(/^\./, "");
    return this.children.filter((c) => c.className.split(/\s+/).includes(cls));
  }
}

function setup() {
  const els = {};
  for (const id of ["ask-form", "ask-question", "ask-submit", "ask-status", "ask-meta", "ask-answers"]) els[id] = new El(id);
  Object.assign(els["ask-form"].dataset, { askUrl: "/ask", historyUrl: "/questions/data", static: "false" });
  const pending = [];
  const ctx = {
    console,
    document: { getElementById: (id) => els[id] || null, createElement: () => new El() },
    fetch: (url) => new Promise((resolve) => pending.push({ url, resolve })),
  };
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  return { ctx, els, pending };
}

const ENTRY = [{ status: "answered", question: "old q", answer: "old a", ts: "2026-10-20T20:00:00", source: "discord", model: "m" }];
const respond = (p, body) => p.resolve({ ok: true, json: async () => body });
const settle = () => new Promise((r) => setTimeout(r, 0));
const earlier = (els) => els["ask-answers"].querySelectorAll(".qa-earlier").length;

// Control: the harness itself works, a resolved load appends its Q&A.
{
  const { ctx, els, pending } = setup();
  ctx.AskBox.setSession("rtnc/2026-10-20");
  assert.equal(pending.length, 1);
  assert.match(pending[0].url, /session=rtnc%2F2026-10-20/);
  respond(pending[0], ENTRY); await settle();
  assert.equal(earlier(els), 1, "control: history should render");
}

// The race: session goes untagged while the previous load is in flight.
{
  const { ctx, els, pending } = setup();
  ctx.AskBox.setSession("rtnc/2026-10-20");
  ctx.AskBox.setSession(null);
  respond(pending[0], ENTRY); await settle();
  assert.equal(earlier(els), 0, "stale history appended after the session went untagged");
}

// A load superseded by a newer session's load is dropped too.
{
  const { ctx, els, pending } = setup();
  ctx.AskBox.setSession("a/2026-10-13");
  ctx.AskBox.setSession("b/2026-10-20");
  respond(pending[1], ENTRY); await settle();
  respond(pending[0], [{ ...ENTRY[0], question: "stale" }]); await settle();
  const qs = els["ask-answers"].children.map((c) => c.children[0].textContent);
  assert.deepEqual(qs, ["old q"], "only the newer session's history should show");
}
console.log("ask.js history: 3 checks passed");
