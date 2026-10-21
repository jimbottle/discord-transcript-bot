// Live page transcript feed (templates/live.html), in either order.
//
// newest-first: new lines are prepended. A reader at the top stays pinned
//   to the newest line; a reader who scrolled down keeps their place (the
//   scroll offset grows by the inserted height).
// oldest-first: new lines are appended. A reader at the bottom follows the
//   newest line; a reader who scrolled up stays where they are.
//
// Existing nodes are never torn down while a session grows, so selection is
// kept and aria-live only announces new lines. A different session file,
// a shrunk file, or an order change rebuilds the feed.
(function () {
    const FOLLOW_PX = 80;

    // "20:26:09" -> "8:26 PM" (same rule as src/utils/timefmt.clock).
    function clock12(hms) {
        const m = /^(\d{1,2}):(\d{2})/.exec(hms || "");
        if (!m) return hms;
        const h = +m[1];
        return (h % 12 || 12) + ":" + m[2] + (h < 12 ? " AM" : " PM");
    }

    function buildEntry(e) {
        // createElement + textContent: transcript text is raw user speech
        // and must never be parsed as HTML.
        const entry = document.createElement("div");
        entry.className = "entry";
        let who = "";
        if (e.player) {
            who = e.player;
            if (e.character && e.character !== e.player) who += " (" + e.character + ")";
        }
        if (who || e.time) {
            const head = document.createElement("div");
            head.className = "entry-meta";
            const whoSpan = document.createElement("strong");
            whoSpan.textContent = who || "Unknown";
            head.appendChild(whoSpan);
            if (e.time) head.appendChild(document.createTextNode(" · " + clock12(e.time)));
            entry.appendChild(head);
        }
        const body = document.createElement("div");
        body.className = "entry-text";
        body.textContent = e.text;
        entry.appendChild(body);
        return entry;
    }

    function createLiveFeed(feed, options) {
        let order = (options && options.order) || "newest";
        let renderedCount = 0;
        let renderedFile = null;
        let lastEntries = [];
        let emptyNode = null;

        function clear() {
            while (feed.firstChild) feed.removeChild(feed.firstChild);
            emptyNode = null;
            renderedCount = 0;
        }

        function showEmpty(text) {
            if (emptyNode) { emptyNode.textContent = text; return; }
            emptyNode = document.createElement("p");
            emptyNode.id = "empty-state";
            emptyNode.textContent = text;
            feed.appendChild(emptyNode);
        }

        function addNew(entries, fresh) {
            if (entries.length === renderedCount) return;
            const frag = document.createDocumentFragment();
            if (order === "newest") {
                const atTop = fresh || feed.scrollTop < FOLLOW_PX;
                const prevHeight = feed.scrollHeight;
                // The file is oldest-first: walk the new tail backwards so
                // the newest line ends up on top.
                for (let i = entries.length - 1; i >= renderedCount; i--) {
                    frag.appendChild(buildEntry(entries[i]));
                }
                feed.insertBefore(frag, feed.firstChild);
                if (atTop) feed.scrollTop = 0;
                else feed.scrollTop += feed.scrollHeight - prevHeight;
            } else {
                const atBottom = fresh
                    || feed.scrollHeight - feed.scrollTop - feed.clientHeight < FOLLOW_PX;
                for (let i = renderedCount; i < entries.length; i++) {
                    frag.appendChild(buildEntry(entries[i]));
                }
                feed.appendChild(frag);
                if (atBottom) feed.scrollTop = feed.scrollHeight;
            }
            renderedCount = entries.length;
        }

        // entries: the whole session file, oldest first (as /live/data sends it).
        function update(entries, filename, emptyText) {
            entries = entries || [];
            lastEntries = entries;
            const fresh = filename !== renderedFile || entries.length < renderedCount;
            if (fresh) {
                clear();
                renderedFile = filename;
            }
            if (entries.length === 0) {
                showEmpty(emptyText || "No session transcript yet.");
                return;
            }
            if (emptyNode) { emptyNode.remove(); emptyNode = null; }
            addNew(entries, fresh);
        }

        function setOrder(next) {
            if (next === order) return;
            order = next;
            clear();
            if (lastEntries.length) addNew(lastEntries, true);
        }

        return { update: update, setOrder: setOrder, order: function () { return order; } };
    }

    window.createLiveFeed = createLiveFeed;
    window.clock12 = clock12;
})();
