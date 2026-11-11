// Campaign picker on the Live page (discord-transcript-bot-5bd). Choosing a
// campaign retags the whole current session through /transcripts/<file>/tag;
// "New campaign…" creates one first (POST /library/campaigns). The page
// polls every few seconds, so update() only moves the select when the
// person is not using it.
function createCampaignPicker(form, urls) {
    const NEW = "__new__";
    const select = form.querySelector("#campaign-select");
    const newBox = form.querySelector("#new-campaign");
    const nameInput = form.querySelector("#new-campaign-name");
    const gameSelect = form.querySelector("#new-campaign-game");
    const cancel = form.querySelector("#new-campaign-cancel");
    const status = form.querySelector("#campaign-status");
    let filename = null;
    let session = {};
    let busy = false;

    async function post(url, body) {
        const res = await fetch(url, {
            method: "POST",
            headers: {
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });
        const data = await res.json().catch(function () { return {}; });
        if (!res.ok) throw new Error(data.error || ("request failed (" + res.status + ")"));
        return data;
    }

    function creating() { return !newBox.hidden; }

    function showNew(on) {
        newBox.hidden = !on;
        if (on) {
            gameSelect.value = session.game_id || "";
            nameInput.focus();
        } else {
            nameInput.value = "";
            select.value = session.campaign_id || "";
        }
    }

    async function tag(campaignId) {
        if (!filename) return;
        busy = true;
        status.textContent = "Saving…";
        try {
            const data = await post(urls.tagUrl(filename), {
                campaign: campaignId,
                whole_session: true,
            });
            session = data.session || session;
            select.value = session.campaign_id || "";
            status.textContent = campaignId
                ? "Saved. Questions now use this campaign."
                : "Saved. This session has no campaign.";
        } catch (e) {
            select.value = session.campaign_id || "";
            status.textContent = "Not saved: " + e.message;
        } finally {
            busy = false;
        }
    }

    select.addEventListener("change", function () {
        const choice = select.value;
        status.textContent = "";
        if (choice === NEW) {
            showNew(true);
            return;
        }
        newBox.hidden = true;
        nameInput.value = "";
        tag(choice);
    });

    cancel.addEventListener("click", function () { showNew(false); status.textContent = ""; });

    form.addEventListener("submit", async function (ev) {
        ev.preventDefault();
        if (!creating() || busy) return;
        const name = nameInput.value.trim();
        if (!name) { status.textContent = "Give the campaign a name."; return; }
        busy = true;
        status.textContent = "Creating…";
        try {
            const c = await post(urls.createUrl, {
                name: name,
                game: gameSelect.value,
                group: session.group_id || "",
            });
            const opt = document.createElement("option");
            opt.value = c.id;
            opt.textContent = c.name;
            select.insertBefore(opt, select.querySelector('option[value="' + NEW + '"]'));
            newBox.hidden = true;
            nameInput.value = "";
            busy = false;
            await tag(c.id);
        } catch (e) {
            status.textContent = "Not created: " + e.message;
            busy = false;
        }
    });

    return {
        update: function (file, s) {
            filename = file || null;
            form.hidden = !filename;
            if (!filename || busy || creating()) return;
            session = s || {};
            if (document.activeElement !== select) {
                select.value = session.campaign_id || "";
            }
        },
    };
}
if (typeof module !== "undefined") module.exports = { createCampaignPicker };
