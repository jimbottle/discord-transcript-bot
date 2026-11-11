// Session-tag editors. Both post the chosen campaign/game/group to
// /transcripts/<file>/tag, then reload so every label on the page reflects
// the new tag (a new campaign can move the recording to another session).
//  - form.tag-form (templates/_tag_editor.html, transcript page): Save button,
//    optional "apply to all recordings" checkbox.
//  - form.tag-inline (templates/_tag_inline.html, Transcripts list): the row's
//    Campaign / Game / Group cells are the dropdowns; a change saves at once
//    and retags the whole session.
(function () {
    async function save(form, status, wholeSession) {
        const el = form.elements;
        const res = await fetch(form.dataset.url, {
            method: "POST",
            headers: {
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                campaign: el.campaign.value,
                game: el.game.value,
                group: el.group.value,
                whole_session: wholeSession,
            }),
        });
        const body = await res.json().catch(function () { return {}; });
        if (!res.ok) throw new Error(body.error || ("Save failed (" + res.status + ")"));
        const n = (body.updated || []).length;
        status.textContent = "Saved " + n + " recording" + (n === 1 ? "" : "s");
        setTimeout(function () { window.location.reload(); }, 400);
    }

    function failure(e) {
        return e instanceof TypeError ? "Couldn't reach the dashboard: " + e : e.message;
    }

    // Choosing a campaign fills in its game and group.
    function fillFromCampaign(form) {
        const el = form.elements;
        const opt = el.campaign.selectedOptions[0];
        if (opt && opt.value) {
            el.game.value = opt.dataset.game || "";
            el.group.value = opt.dataset.group || "";
        }
    }

    function wireEditor(form) {
        const status = form.querySelector(".tag-status");
        form.elements.campaign.addEventListener("change", function () { fillFromCampaign(form); });
        form.addEventListener("submit", async function (ev) {
            ev.preventDefault();
            status.textContent = "Saving…";
            const whole = form.elements.whole_session;
            try {
                await save(form, status, whole ? whole.checked : true);
            } catch (e) {
                status.textContent = failure(e);
            }
        });
    }

    function wireInline(form) {
        const el = form.elements;
        const selects = [el.campaign, el.game, el.group];
        const status = el.campaign.parentNode.querySelector(".tag-status");
        const saved = selects.map(function (s) { return s.value; });
        selects.forEach(function (sel) {
            sel.addEventListener("change", async function () {
                if (sel === el.campaign) fillFromCampaign(form);
                selects.forEach(function (s) { s.disabled = true; });
                status.textContent = "Saving…";
                try {
                    await save(form, status, true);
                } catch (e) {
                    // Put the row back the way it is saved.
                    selects.forEach(function (s, i) { s.value = saved[i]; s.disabled = false; });
                    status.textContent = failure(e);
                }
            });
        });
    }

    function init() {
        document.querySelectorAll("form.tag-form").forEach(wireEditor);
        document.querySelectorAll("form.tag-inline").forEach(wireInline);
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
