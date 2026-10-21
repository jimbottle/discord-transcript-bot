// "Ask about this session" box shared by the Live page and the transcript
// viewer (see templates/_ask.html). Posts to /ask, which runs the same
// Claude module as Discord's /ask. One question in flight at a time;
// answers stack newest-first so a long session's Q&A stays readable.
//
// window.AskBox.setAvailability(reason) enables the box when `reason` is
// empty and disables it (showing the reason) otherwise. A page with a
// live poll calls it on every poll; a static page gets its state from
// the form's data-* attributes once, here.
(function () {
    const NO_KEY_REASON =
        "ANTHROPIC_API_KEY is not set for the dashboard — start it with `make web` (reads op.env).";

    function init() {
        const form = document.getElementById("ask-form");
        if (!form) return;
        const url = form.dataset.askUrl;
        const filename = form.dataset.filename || null;
        const input = document.getElementById("ask-question");
        const submit = document.getElementById("ask-submit");
        const status = document.getElementById("ask-status");
        const meta = document.getElementById("ask-meta");
        const answers = document.getElementById("ask-answers");
        let busy = false;
        let ready = false;

        function setAvailability(reason) {
            ready = !reason;
            meta.textContent = reason || "";
            input.disabled = !ready || busy;
            submit.disabled = !ready || busy;
        }

        function addQA(question, result, errorText) {
            // textContent only: the question is user input and the answer
            // is model output; neither may be parsed as HTML.
            const block = document.createElement("div");
            block.className = "qa" + (errorText ? " qa-failed" : "");

            const q = document.createElement("div");
            q.className = "qa-q";
            q.textContent = question;
            block.appendChild(q);

            const a = document.createElement("div");
            a.className = errorText ? "qa-err" : "qa-a";
            a.textContent = errorText || result.answer;
            block.appendChild(a);

            if (result && result.model) {
                const m = document.createElement("div");
                m.className = "qa-meta";
                m.textContent = result.model
                    + " · in " + result.input_tokens
                    + (result.cached_tokens ? " (" + result.cached_tokens + " cached)" : "")
                    + " · out " + result.output_tokens
                    + (result.notes_used && result.notes_used.length
                        ? " · notes: " + result.notes_used.join(", ") : "")
                    + " · " + new Date().toLocaleTimeString();
                block.appendChild(m);
            }
            answers.prepend(block);
        }

        async function submitAsk(ev) {
            ev.preventDefault();
            const question = input.value.trim();
            if (!question || busy || !ready) return;

            busy = true;
            input.disabled = true;
            submit.disabled = true;
            submit.setAttribute("aria-busy", "true");
            status.textContent = "Asking…";
            const started = Date.now();
            const payload = { question: question };
            if (filename) payload.filename = filename;

            try {
                const res = await fetch(url, {
                    method: "POST",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest",
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(payload),
                });
                let body = null;
                try { body = await res.json(); } catch (e) { body = null; }
                if (res.ok && body && body.answer) {
                    addQA(question, body, null);
                    input.value = "";
                    status.textContent =
                        "Answered in " + Math.round((Date.now() - started) / 1000) + "s";
                } else {
                    const msg = (body && body.error)
                        ? body.error
                        : ("Request failed (" + res.status + ")");
                    addQA(question, null, msg);
                    status.textContent = "";
                }
            } catch (e) {
                addQA(question, null, "Couldn't reach the dashboard: " + e);
                status.textContent = "";
            } finally {
                busy = false;
                submit.removeAttribute("aria-busy");
                input.disabled = !ready;
                submit.disabled = !ready;
                if (ready) input.focus();
            }
        }

        form.addEventListener("submit", submitAsk);
        // Cmd/Ctrl+Enter submits from the textarea; plain Enter stays a newline.
        input.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submitAsk(e);
        });

        window.AskBox = { setAvailability: setAvailability, NO_KEY_REASON: NO_KEY_REASON };

        if (form.dataset.static === "true") {
            if (form.dataset.askConfigured !== "true") setAvailability(NO_KEY_REASON);
            else if (form.dataset.empty === "true") setAvailability("Nothing transcribed in this file.");
            else setAvailability("");
        } else {
            setAvailability("Loading…");
        }
    }

    init();
})();
