// Newest-first / oldest-first preference and its toggle (Live feed and
// transcript pages). One remembered choice per page key, per browser;
// storage can be unavailable (private window, blocked site data), so every
// read and write is guarded and the page still works with the default.
(function () {
    const ORDERS = ["newest", "oldest"];
    const LABELS = { newest: "Newest first", oldest: "Oldest first" };

    function get(key, fallback) {
        try {
            const v = window.localStorage.getItem(key);
            if (ORDERS.indexOf(v) !== -1) return v;
        } catch (e) { /* storage unavailable: use the default */ }
        return fallback;
    }

    function set(key, value) {
        try { window.localStorage.setItem(key, value); } catch (e) { /* ignore */ }
    }

    // Two buttons in `container`; aria-pressed marks the current order.
    // Calls onChange(order) when the user picks the other one.
    function mountToggle(container, current, onChange) {
        if (!container) return;
        // No role="group": Pico styles that as a full-width button bar.
        // The buttons carry aria-pressed, which is what a reader needs.
        container.setAttribute("aria-label", "Sort order");
        const buttons = {};
        function paint(order) {
            ORDERS.forEach(function (o) {
                buttons[o].setAttribute("aria-pressed", o === order ? "true" : "false");
                buttons[o].className = "order-btn" + (o === order ? " active" : "");
            });
        }
        ORDERS.forEach(function (o) {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = LABELS[o];
            b.dataset.order = o;
            b.addEventListener("click", function () {
                if (o === current) return;
                current = o;
                paint(o);
                onChange(o);
            });
            buttons[o] = b;
            container.appendChild(b);
        });
        paint(current);
    }

    window.FeedOrder = { get: get, set: set, mountToggle: mountToggle, ORDERS: ORDERS };
})();
