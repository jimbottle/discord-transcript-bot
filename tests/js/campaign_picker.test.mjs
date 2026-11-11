// Behavioural tests for web/static/campaign-picker.js: the real script in a
// node vm against a minimal fake form and a recording fetch.
// Usage: node tests/js/campaign_picker.test.mjs
import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../../web/static/campaign-picker.js", import.meta.url), "utf8");

class El {
  constructor(id = "") {
    this.id = id; this.value = ""; this.hidden = false; this.textContent = "";
    this.listeners = {}; this.options = []; this.focused = false;
  }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  fire(t, ev = { preventDefault() {} }) { return Promise.all((this.listeners[t] || []).map((f) => f(ev))); }
  focus() { this.focused = true; }
  querySelector(sel) {
    const m = sel.match(/option\[value="(.*)"\]/);
    return m ? this.options.find((o) => o.value === m[1]) || null : null;
  }
  insertBefore(o, ref) { this.options.splice(this.options.indexOf(ref), 0, o); }
}

function setup(responses) {
  const parts = {};
  for (const id of ["campaign-select", "new-campaign", "new-campaign-name",
    "new-campaign-game", "new-campaign-cancel", "campaign-status"]) parts[id] = new El(id);
  parts["new-campaign"].hidden = true;
  parts["campaign-select"].options = [{ value: "" }, { value: "rtnc" }, { value: "__new__" }];
  const form = new El("form");
  form.querySelector = (sel) => parts[sel.slice(1)];
  const calls = [];
  const ctx = {
    document: { activeElement: null, createElement: () => ({ value: "", textContent: "" }) },
    fetch: async (url, init) => {
      calls.push({ url, body: JSON.parse(init.body), headers: init.headers });
      const r = responses.shift();
      return { ok: r.ok !== false, status: r.status || 200, json: async () => r.body };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  const picker = ctx.createCampaignPicker(form, {
    tagUrl: (f) => "/transcripts/" + f + "/tag", createUrl: "/library/campaigns",
  });
  return { picker, form, parts, calls, ctx };
}
const tick = () => new Promise((r) => setTimeout(r, 0));
const FILE = "2026-11-10_20-04-24.txt";

// hidden with no session; shows the session's campaign once there is one
{
  const { picker, form, parts } = setup([]);
  picker.update(null, {});
  assert.equal(form.hidden, true);
  picker.update(FILE, { campaign_id: "rtnc" });
  assert.equal(form.hidden, false);
  assert.equal(parts["campaign-select"].value, "rtnc");
}

// choosing a campaign retags the whole session with the CHOSEN id
{
  const { picker, parts, calls } = setup([{ body: { session: { campaign_id: "other" } } }]);
  picker.update(FILE, { campaign_id: "rtnc" });
  const select = parts["campaign-select"];
  select.value = "other";
  await select.fire("change"); await tick();
  assert.deepEqual(calls.map((c) => [c.url, c.body]), [
    ["/transcripts/" + FILE + "/tag", { campaign: "other", whole_session: true }],
  ]);
  assert.equal(calls[0].headers["X-Requested-With"], "XMLHttpRequest");
  assert.equal(select.value, "other");
  assert.match(parts["campaign-status"].textContent, /^Saved/);
}

// a failed save puts the select back and says why
{
  const { picker, parts } = setup([{ ok: false, status: 400, body: { error: "unknown campaign 'x'" } }]);
  picker.update(FILE, { campaign_id: "rtnc" });
  const select = parts["campaign-select"];
  select.value = "x";
  await select.fire("change"); await tick();
  assert.equal(select.value, "rtnc");
  assert.equal(parts["campaign-status"].textContent, "Not saved: unknown campaign 'x'");
}

// the poll does not move the select while the person has it open
{
  const { picker, parts, ctx } = setup([]);
  picker.update(FILE, { campaign_id: "rtnc" });
  const select = parts["campaign-select"];
  ctx.document.activeElement = select;
  select.value = "other";
  picker.update(FILE, { campaign_id: "rtnc" });
  assert.equal(select.value, "other");
}

// "New campaign…" creates it for the session's group, then uses it
{
  const { picker, form, parts, calls } = setup([
    { status: 201, body: { id: "new-heist", name: "New Heist", game: "cyberpunk-red", group: "noah-gm" } },
    { body: { session: { campaign_id: "new-heist" } } },
  ]);
  picker.update(FILE, { campaign_id: "rtnc", game_id: "cyberpunk-red", group_id: "noah-gm" });
  const select = parts["campaign-select"];
  select.value = "__new__";
  await select.fire("change");
  assert.equal(parts["new-campaign"].hidden, false);
  assert.equal(parts["new-campaign-game"].value, "cyberpunk-red");
  assert.equal(calls.length, 0);
  // polling while the form is open must not close or reset it
  picker.update(FILE, { campaign_id: "rtnc" });
  assert.equal(select.value, "__new__");
  parts["new-campaign-name"].value = "  New Heist ";
  await form.fire("submit"); await tick();
  assert.deepEqual(calls.map((c) => [c.url, c.body]), [
    ["/library/campaigns", { name: "New Heist", game: "cyberpunk-red", group: "noah-gm" }],
    ["/transcripts/" + FILE + "/tag", { campaign: "new-heist", whole_session: true }],
  ]);
  assert.deepEqual(select.options.map((o) => o.value), ["", "rtnc", "new-heist", "__new__"]);
  assert.equal(select.value, "new-heist");
  assert.equal(parts["new-campaign"].hidden, true);
}

// an empty name is refused without a request; cancel goes back to the tag
{
  const { picker, form, parts, calls } = setup([]);
  picker.update(FILE, { campaign_id: "rtnc" });
  const select = parts["campaign-select"];
  select.value = "__new__";
  await select.fire("change");
  await form.fire("submit");
  assert.equal(calls.length, 0);
  assert.equal(parts["campaign-status"].textContent, "Give the campaign a name.");
  await parts["new-campaign-cancel"].fire("click");
  assert.equal(parts["new-campaign"].hidden, true);
  assert.equal(select.value, "rtnc");
}

console.log("campaign_picker: ok");
