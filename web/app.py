import os

from flask import Flask, abort, jsonify, render_template, request

from bot_manager import BotManager
from transcript_service import (
    format_uptime,
    get_bot_state,
    get_live_session,
    get_log_entries,
    get_transcript,
    list_logs,
    list_transcripts,
    parse_transcript_text,
    search_logs,
)

app = Flask(__name__)
bot = BotManager()


def _require_csrf_header():
    # CSRF defense for state-changing endpoints. Cross-origin <form> submissions
    # cannot set custom headers, so requiring X-Requested-With blocks the classic
    # form-CSRF attack. Additionally reject any request whose Origin is set and
    # doesn't match this host — defense in depth against a browser extension or
    # other same-machine actor that can set arbitrary headers.
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        abort(403)
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        abort(403)


@app.route("/")
def index():
    transcripts = list_transcripts()[:5]
    state = get_bot_state()
    return render_template(
        "index.html",
        transcripts=transcripts,
        bot_status=bot.status(),
        live=get_live_session(),
        bot_state=state,
        uptime=format_uptime(state.get("started_at")) if state else None,
    )


@app.route("/transcripts")
def transcripts():
    txt_files = list_transcripts()
    log_files = list_logs()
    return render_template("transcripts.html", txt_files=txt_files, log_files=log_files)


@app.route("/transcripts/<filename>")
def view_transcript(filename):
    # Read the file once: get raw, then parse locally (raw is also kept
    # for the client-side Raw toggle, so no second read).
    raw = get_transcript(filename)
    if raw is None:
        return "Not found", 404
    entries = parse_transcript_text(raw)
    return render_template(
        "transcript.html", filename=filename, entries=entries, raw=raw
    )


@app.route("/logs/<filename>")
def view_log(filename):
    entries = get_log_entries(filename)
    if entries is None:
        return "Not found", 404
    return render_template("log.html", filename=filename, entries=entries)


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/live/data")
def live_data():
    return jsonify(get_live_session())


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = search_logs(query) if query else []
    return render_template("search.html", query=query, results=results)


@app.route("/bot/start", methods=["POST"])
def bot_start():
    _require_csrf_header()
    bot.start()
    return jsonify(bot.status())


@app.route("/bot/stop", methods=["POST"])
def bot_stop():
    _require_csrf_header()
    bot.stop()
    return jsonify(bot.status())


@app.route("/bot/status")
def bot_status():
    return jsonify(bot.status())


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="127.0.0.1", port=5001, debug=debug)
