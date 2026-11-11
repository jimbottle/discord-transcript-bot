// Copy-to-clipboard for session ids (discord-transcript-bot-qef).
//
// Any <button class="copy-id" data-copy="K7QF-2MXD"> copies its data-copy
// value when clicked and says so on the button for a moment. One delegated
// listener on the document, so it also covers the Live page, where the box
// is filled in by the poll after this script has loaded.
(function () {
  "use strict";

  var RESET_MS = 1500;

  function fallbackCopy(text) {
    // navigator.clipboard needs a secure context (https or localhost). The
    // dashboard is normally on 127.0.0.1, but it can be reached by LAN IP.
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }
    document.body.removeChild(ta);
    return ok;
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(
        function () { return true; },
        function () { return fallbackCopy(text); }
      );
    }
    return Promise.resolve(fallbackCopy(text));
  }

  function show(button, label) {
    if (button.dataset.label === undefined) {
      button.dataset.label = button.textContent;
    }
    button.textContent = label;
    clearTimeout(button._copyTimer);
    button._copyTimer = setTimeout(function () {
      button.textContent = button.dataset.label;
    }, RESET_MS);
  }

  function onClick(event) {
    var button = event.target && event.target.closest
      ? event.target.closest("button.copy-id")
      : null;
    if (!button) return undefined;
    var text = button.dataset.copy || "";
    if (!text) return undefined;
    return copyText(text).then(function (ok) {
      show(button, ok ? "Copied" : "Copy failed");
      return ok;
    });
  }

  document.addEventListener("click", onClick);
})();
