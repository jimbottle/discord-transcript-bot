// Behavioural tests for the Transcripts list's in-row tag dropdowns
// (web/static/tag-editor.js, form.tag-inline): the real script in a node vm
// against a minimal fake document and fetch.
// Usage: node tests/js/tag_inline.test.mjs
import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../../web/static/tag-editor.js", import.meta.url), "utf8");

function select(name, value, options) {
  const listeners = {};
  const sel = {
    name, value, disabled: false, options,
    addEventListener(t, f) { listeners[t] = f; },
    fire() { return listeners.change(); },
    get selectedOptions() { return options.filter((o) => o.value === sel.value); },
  };
  return sel;
}

function setup(response) {
  const status = { textContent: "" };
  const campaign = select("campaign", "rtnc", [
    { value: "", dataset: {} },
    { value: "rtnc", dataset: { game: "cyberpunk-red", group: "noah-gm" } },
    { value: "lmop", dataset: { game: "dnd5e", group: "fri" } },
  ]);
  campaign.parentNode = { querySelector: (s) => (s === ".tag-status" ? status : null) };
  const game = select("game", "cyberpunk-red", []);
  const group = select("group", "noah-gm", []);
  const form = { dataset: { url: "/transcripts/a.txt/tag" }, elements: { campaign, game, group } };
  const posts = [];
  const timers = [];
  let reloaded = false;
  const ctx = {
    document: {
      readyState: "complete",
      querySelectorAll: (s) => (s === "form.tag-inline" ? [form] : []),
    },
    window: { location: { reload() { reloaded = true; } } },
    setTimeout(f) { timers.push(f); },
    fetch: async (url, init) => {
      posts.push({ url, body: JSON.parse(init.body), headers: init.headers });
      if (response instanceof Error) throw response;
      return { ok: response.ok, status: response.status, json: async () => response.body };
    },
    TypeError, Error, JSON, Promise,
  };
  vm.runInNewContext(src, ctx);
  return { campaign, game, group, status, posts, timers, reloaded: () => reloaded };
}

// Choosing a campaign fills its game and group and saves the whole session.
{
  const t = setup({ ok: true, status: 200, body: { updated: ["a.txt", "b.txt"] } });
  t.campaign.value = "lmop";
  await t.campaign.fire();
  assert.equal(t.posts.length, 1);
  assert.equal(t.posts[0].url, "/transcripts/a.txt/tag");
  assert.deepEqual(t.posts[0].body, { campaign: "lmop", game: "dnd5e", group: "fri", whole_session: true });
  assert.equal(t.posts[0].headers["X-Requested-With"], "XMLHttpRequest");
  assert.equal(t.status.textContent, "Saved 2 recordings");
  assert.ok(t.campaign.disabled && t.game.disabled && t.group.disabled);
  t.timers.forEach((f) => f());
  assert.ok(t.reloaded());
}

// Changing only the group saves just that, keeping campaign and game.
{
  const t = setup({ ok: true, status: 200, body: { updated: ["a.txt"] } });
  t.group.value = "";
  await t.group.fire();
  assert.deepEqual(t.posts[0].body, { campaign: "rtnc", game: "cyberpunk-red", group: "", whole_session: true });
  assert.equal(t.status.textContent, "Saved 1 recording");
}

// A refused save puts the row back and says why; nothing reloads.
{
  const t = setup({ ok: false, status: 400, body: { error: "unknown campaign" } });
  t.campaign.value = "lmop";
  await t.campaign.fire();
  assert.equal(t.campaign.value, "rtnc");
  assert.equal(t.game.value, "cyberpunk-red");
  assert.equal(t.group.value, "noah-gm");
  assert.ok(!t.campaign.disabled && !t.game.disabled && !t.group.disabled);
  assert.equal(t.status.textContent, "unknown campaign");
  assert.equal(t.timers.length, 0);
}

// An unreachable dashboard is reported as such.
{
  const t = setup(new TypeError("Failed to fetch"));
  t.game.value = "dnd5e";
  await t.game.fire();
  assert.equal(t.game.value, "cyberpunk-red");
  assert.match(t.status.textContent, /^Couldn't reach the dashboard/);
}

console.log("tag_inline: ok");
