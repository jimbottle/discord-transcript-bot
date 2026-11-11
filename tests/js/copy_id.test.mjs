// Behavioural tests for web/static/copy-id.js: the real script in a node vm
// against a minimal fake document and clipboard.
// Usage: node tests/js/copy_id.test.mjs
import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../../web/static/copy-id.js", import.meta.url), "utf8");

function button(copy, label = "Copy") {
  const b = { dataset: {}, textContent: label, tag: "button.copy-id" };
  if (copy !== undefined) b.dataset.copy = copy;
  b.closest = (sel) => (sel === "button.copy-id" ? b : null);
  return b;
}

function setup({ clipboard, execOk = true } = {}) {
  const listeners = {};
  const timers = [];
  const copied = [];
  const appended = [];
  const document = {
    addEventListener(t, f) { listeners[t] = f; },
    createElement() { return { style: {}, setAttribute() {}, select() {}, value: "" }; },
    body: {
      appendChild(el) { appended.push(el); },
      removeChild(el) { appended.splice(appended.indexOf(el), 1); },
    },
    execCommand(cmd) {
      if (!execOk) return false;
      copied.push({ via: "execCommand", text: appended[appended.length - 1].value });
      return cmd === "copy";
    },
  };
  const navigator = clipboard === undefined ? {} : { clipboard };
  const ctx = {
    document, navigator,
    setTimeout(f) { timers.push(f); return timers.length; },
    clearTimeout() {},
    Promise,
  };
  vm.runInNewContext(src, ctx);
  return { click: (target) => listeners.click({ target }), timers, copied, appended };
}

// 1. Clicking the button copies its data-copy value and says so, then resets.
{
  const written = [];
  const env = setup({ clipboard: { writeText: (t) => { written.push(t); return Promise.resolve(); } } });
  const b = button("K7QF-2MXD");
  assert.equal(await env.click(b), true);
  assert.deepEqual(written, ["K7QF-2MXD"]);
  assert.equal(b.textContent, "Copied");
  env.timers.at(-1)();
  assert.equal(b.textContent, "Copy");
}

// 2. A second click while it still says "Copied" keeps the original label.
{
  const env = setup({ clipboard: { writeText: () => Promise.resolve() } });
  const b = button("AAAA-BBBB");
  await env.click(b);
  await env.click(b);
  env.timers.at(-1)();
  assert.equal(b.textContent, "Copy");
}

// 3. No clipboard API (dashboard opened by LAN IP): the textarea fallback.
{
  const env = setup({});
  const b = button("K7QF-2MXD");
  assert.equal(await env.click(b), true);
  assert.deepEqual(env.copied, [{ via: "execCommand", text: "K7QF-2MXD" }]);
  assert.equal(env.appended.length, 0); // the helper element is removed
  assert.equal(b.textContent, "Copied");
}

// 4. The clipboard API refusing falls back too; if that fails, it says so.
{
  const env = setup({ clipboard: { writeText: () => Promise.reject(new Error("denied")) }, execOk: false });
  const b = button("K7QF-2MXD");
  assert.equal(await env.click(b), false);
  assert.equal(b.textContent, "Copy failed");
}

// 5. Clicks elsewhere, and a button with nothing to copy, do nothing.
{
  const written = [];
  const env = setup({ clipboard: { writeText: (t) => { written.push(t); return Promise.resolve(); } } });
  assert.equal(env.click({ closest: () => null }), undefined);
  const empty = button("");
  assert.equal(env.click(empty), undefined);
  assert.deepEqual(written, []);
  assert.equal(empty.textContent, "Copy");
}

console.log("copy_id: all passed");
