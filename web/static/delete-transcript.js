// Delete control (templates/_delete_transcript.html): "Delete…" reveals an
// inline confirmation; only "Yes, delete" sends the request, with
// confirm: true (the server refuses without it). No browser dialogs.
(function () {
    function wire(box) {
        const start = box.querySelector(".delete-start");
        if (!start) return;  // live recording: no control
        const confirmBox = box.querySelector(".delete-confirm");
        const yes = box.querySelector(".delete-yes");
        const cancel = box.querySelector(".delete-cancel");
        const audio = box.querySelector(".delete-audio");
        const status = box.querySelector(".delete-status");
        start.addEventListener("click", function () {
            confirmBox.hidden = false;
            start.hidden = true;
            cancel.focus();
        });
        cancel.addEventListener("click", function () {
            confirmBox.hidden = true;
            start.hidden = false;
            status.textContent = "";
        });
        yes.addEventListener("click", async function () {
            yes.disabled = true;
            status.textContent = "Deleting…";
            try {
                const res = await fetch(box.dataset.url, {
                    method: "POST",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ confirm: true, include_audio: audio ? audio.checked : false }),
                });
                const body = await res.json().catch(function () { return {}; });
                if (!res.ok) {
                    status.textContent = body.error || ("Delete failed (" + res.status + ")");
                    yes.disabled = false;
                    return;
                }
                status.textContent = "Deleted.";
                const after = box.dataset.after;
                setTimeout(function () {
                    if (after && after !== "reload") window.location.href = after;
                    else window.location.reload();
                }, 300);
            } catch (e) {
                status.textContent = "Couldn't reach the dashboard: " + e;
                yes.disabled = false;
            }
        });
    }
    function init() { document.querySelectorAll(".delete-transcript").forEach(wire); }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
