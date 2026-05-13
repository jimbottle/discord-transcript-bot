import os

from flask import Flask, abort, jsonify, render_template, request

from bot_manager import BotManager
from transcript_service import (
    get_transcript,
    list_logs,
    list_transcripts,
    search_logs,
)

app = Flask(__name__)
bot = BotManager()


def _require_same_origin():
    # CSRF defense: cross-origin <form> submissions cannot set custom headers,
    # so require X-Requested-With on every state-changing request. The dashboard
    # invokes these via fetch() with the header set.
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        abort(403)


@app.route("/")
def index():
    transcripts = list_transcripts()[:5]
    return render_template("index.html", transcripts=transcripts, bot_status=bot.status())


@app.route("/transcripts")
def transcripts():
    txt_files = list_transcripts()
    log_files = list_logs()
    return render_template("transcripts.html", txt_files=txt_files, log_files=log_files)


@app.route("/transcripts/<filename>")
def view_transcript(filename):
    content = get_transcript(filename)
    if content is None:
        return "Not found", 404
    return render_template("transcript.html", filename=filename, content=content)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = search_logs(query) if query else []
    return render_template("search.html", query=query, results=results)


@app.route("/bot/start", methods=["POST"])
def bot_start():
    _require_same_origin()
    bot.start()
    return jsonify(bot.status())


@app.route("/bot/stop", methods=["POST"])
def bot_stop():
    _require_same_origin()
    bot.stop()
    return jsonify(bot.status())


@app.route("/bot/status")
def bot_status():
    return jsonify(bot.status())


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="127.0.0.1", port=5001, debug=debug)
