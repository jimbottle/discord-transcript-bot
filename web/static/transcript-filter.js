// Shared client-side filter for the transcript and JSON-log viewers.
// Hides any .entry inside #parsed-view whose data-search attribute
// doesn't contain the (lowercased) query. No-op if the elements aren't
// present, so it can be loaded site-wide from base.html.
(function () {
  function init() {
    var input = document.getElementById("filter");
    var view = document.getElementById("parsed-view");
    if (!input || !view) return;
    input.addEventListener("input", function () {
      var q = input.value.trim().toLowerCase();
      var rows = view.querySelectorAll(".entry");
      for (var i = 0; i < rows.length; i++) {
        var hay = rows[i].getAttribute("data-search") || "";
        rows[i].style.display = !q || hay.indexOf(q) !== -1 ? "" : "none";
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
