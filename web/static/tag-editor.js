// Session-tag editor (templates/_tag_editor.html). Posts the chosen
// campaign/game/group to /transcripts/<file>/tag, then reloads so every
// label on the page reflects the new tag.
(function () {
    function wire(form) {
        const campaign = form.elements.campaign;
        const game = form.elements.game;
        const group = form.elements.group;
        const status = form.querySelector(".tag-status");
        campaign.addEventListener("change", function () {
            const opt = campaign.selectedOptions[0];
            if (opt && opt.value) {
                game.value = opt.dataset.game || "";
                group.value = opt.dataset.group || "";
            }
        });
        form.addEventListener("submit", async function (ev) {
            ev.preventDefault();
            status.textContent = "Saving…";
            const whole = form.elements.whole_session;
            try {
                const res = await fetch(form.dataset.url, {
                    method: "POST",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        campaign: campaign.value,
                        game: game.value,
                        group: group.value,
                        whole_session: whole ? whole.checked : true,
                    }),
                });
                const body = await res.json().catch(function () { return {}; });
                if (!res.ok) {
                    status.textContent = body.error || ("Save failed (" + res.status + ")");
                    return;
                }
                status.textContent = "Saved " + body.updated.length + " recording"
                    + (body.updated.length === 1 ? "" : "s");
                setTimeout(function () { window.location.reload(); }, 400);
            } catch (e) {
                status.textContent = "Couldn't reach the dashboard: " + e;
            }
        });
    }
    function init() {
        document.querySelectorAll("form.tag-form").forEach(wire);
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
