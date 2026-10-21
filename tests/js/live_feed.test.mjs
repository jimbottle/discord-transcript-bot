// Behavioural tests for web/static/live-feed.js and feed-order.js: the real
// scripts in a node vm against a tiny DOM whose container simulates
// scrolling (each line 20px tall, 100px viewport).
// Usage: node tests/js/live_feed.test.mjs
import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const read = (f) => readFileSync(new URL("../../web/static/" + f, import.meta.url), "utf8");
const LINE = 20, VIEW = 100;

class El {
  constructor(tag = "div") {
    this.tag = tag; this.children = []; this.parent = null; this.id = ""; this.className = "";
    this._text = ""; this.dataset = {}; this.attrs = {}; this.listeners = {}; this._scrollTop = 0;
    this.type = "";
  }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this._text = v; this.children = []; }
  get firstChild() { return this.children[0] || null; }
  appendChild(c) {
    if (c.tag === "#fragment") { c.children.forEach((k) => this.appendChild(k)); c.children = []; return c; }
    c.parent = this; this.children.push(c); return c;
  }
  insertBefore(c, ref) {
    const nodes = c.tag === "#fragment" ? c.children.splice(0) : [c];
    const at = ref ? this.children.indexOf(ref) : this.children.length;
    nodes.forEach((n) => (n.parent = this));
    this.children.splice(at < 0 ? this.children.length : at, 0, ...nodes);
    return c;
  }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); c.parent = null; return c; }
  remove() { if (this.parent) this.parent.removeChild(this); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  click() { (this.listeners.click || []).forEach((f) => f()); }
  get scrollHeight() { return this.children.filter((c) => c.className === "entry").length * LINE; }
  get clientHeight() { return VIEW; }
  get scrollTop() { return this._scrollTop; }
  set scrollTop(v) { this._scrollTop = Math.max(0, Math.min(v, Math.max(0, this.scrollHeight - VIEW))); }
}

function env(storage) {
  const ctx = {
    console,
    document: {
      createElement: (t) => new El(t),
      createTextNode: (t) => { const n = new El("#text"); n._text = t; return n; },
      createDocumentFragment: () => new El("#fragment"),
    },
    localStorage: storage,
  };
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(read("feed-order.js"), ctx);
  vm.runInContext(read("live-feed.js"), ctx);
  return ctx;
}
const lines = (n) => Array.from({ length: n }, (_, i) => ({ time: "20:00:" + String(i).padStart(2, "0"), player: "P", character: "C", text: "line " + i }));
const shown = (feed) => feed.children.filter((c) => c.className === "entry").map((c) => c.children[1].textContent);
const mem = () => { const m = {}; return { getItem: (k) => (k in m ? m[k] : null), setItem: (k, v) => (m[k] = v) }; };

// newest-first: newest on top, pinned at top, reader's place kept when scrolled
{
  const ctx = env(mem()); const feed = new El();
  const f = ctx.createLiveFeed(feed, { order: "newest" });
  f.update(lines(10), "a.txt");
  assert.deepEqual(shown(feed).slice(0, 2), ["line 9", "line 8"]);
  assert.equal(feed.scrollTop, 0);
  f.update(lines(11), "a.txt");
  assert.equal(shown(feed)[0], "line 10");
  assert.equal(feed.scrollTop, 0, "pinned to the newest line");
  feed.scrollTop = 100; // reader scrolled down to re-read
  f.update(lines(13), "a.txt");
  assert.equal(feed.scrollTop, 140, "offset grows by the two inserted lines");
}

// oldest-first: newest at the bottom, follows at bottom, stays put when scrolled up
{
  const ctx = env(mem()); const feed = new El();
  const f = ctx.createLiveFeed(feed, { order: "oldest" });
  f.update(lines(10), "a.txt");
  assert.deepEqual(shown(feed).slice(0, 2), ["line 0", "line 1"]);
  assert.equal(feed.scrollTop, 100, "starts at the bottom (newest)");
  f.update(lines(11), "a.txt");
  assert.equal(shown(feed).at(-1), "line 10");
  assert.equal(feed.scrollTop, 120, "follows the newest line");
  feed.scrollTop = 20; // reader scrolled up
  f.update(lines(14), "a.txt");
  assert.equal(feed.scrollTop, 20, "reader's place kept");
}

// switching order rebuilds in the new order and jumps to the follow end
{
  const ctx = env(mem()); const feed = new El();
  const f = ctx.createLiveFeed(feed, { order: "newest" });
  f.update(lines(8), "a.txt");
  f.setOrder("oldest");
  assert.deepEqual(shown(feed), lines(8).map((l) => l.text));
  assert.equal(feed.scrollTop, 60);
  f.setOrder("newest");
  assert.deepEqual(shown(feed), lines(8).map((l) => l.text).reverse());
  assert.equal(feed.scrollTop, 0);
  f.update(lines(9), "a.txt"); // keeps growing correctly after a switch
  assert.equal(shown(feed)[0], "line 8");
  assert.equal(shown(feed).length, 9);
}

// new session file resets; empty state shown then cleared; times are 12-hour
{
  const ctx = env(mem()); const feed = new El();
  const f = ctx.createLiveFeed(feed, { order: "newest" });
  f.update(lines(3), "a.txt");
  f.update([], "b.txt", "Session file is empty so far.");
  assert.equal(feed.children.length, 1);
  assert.equal(feed.children[0].textContent, "Session file is empty so far.");
  f.update(lines(2), "b.txt");
  assert.deepEqual(shown(feed), ["line 1", "line 0"]);
  assert.equal(feed.children.length, 2, "empty-state removed");
  assert.match(feed.children[0].children[0].textContent, / · 8:00 PM$/);
}

// FeedOrder: default, remembered choice, toggle state, broken storage
{
  const store = mem();
  const ctx = env(store);
  assert.equal(ctx.FeedOrder.get("k", "newest"), "newest");
  ctx.FeedOrder.set("k", "oldest");
  assert.equal(ctx.FeedOrder.get("k", "newest"), "oldest");
  store.setItem("k", "garbage");
  assert.equal(ctx.FeedOrder.get("k", "newest"), "newest", "invalid stored value ignored");

  const box = new El(); const picked = [];
  ctx.FeedOrder.mountToggle(box, "newest", (o) => picked.push(o));
  const [newest, oldest] = box.children;
  assert.equal(newest.attrs["aria-pressed"], "true");
  assert.equal(oldest.attrs["aria-pressed"], "false");
  oldest.click(); oldest.click(); // second click on the active one is a no-op
  assert.deepEqual(picked, ["oldest"]);
  assert.equal(oldest.attrs["aria-pressed"], "true");

  const broken = env({ getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } });
  assert.equal(broken.FeedOrder.get("k", "oldest"), "oldest");
  broken.FeedOrder.set("k", "newest"); // must not throw
}
console.log("live feed + order toggle: all checks passed");
