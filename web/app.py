import os

from flask import Flask, abort, jsonify, render_template, request

from bot_manager import BotManager
from roster_service import get_roster, get_unmapped_speakers, roster_path
from transcript_service import (
    STATE_STALE_SECONDS,
    format_duration,
    format_uptime,
    get_bot_state,
    get_live_session,
    get_log_entries,
    get_transcript,
    heartbeat_age,
    list_logs,
    list_transcripts,
    parse_transcript_text,
    search_logs,
)

# roster_service put the project root on sys.path so the shared store
# (the bot's own roster writer) is importable from web/.
from src import player_map_store  # noqa: E402

# Load .env so a custom PLAYER_MAP_FILE_PATH (or other config) is honoured
# by the dashboard process too — the bot already loads it. Best-effort:
# a missing python-dotenv or .env just falls back to the defaults.
try:
    from dotenv import load_dotenv

    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    )
except ImportError:
    pass

app = Flask(__name__)
bot = BotManager()
# Format a raw seconds count in templates, e.g. a guild's stalled output age.
app.jinja_env.filters["duration"] = format_duration


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
    hb = heartbeat_age(state)
    return render_template(
        "index.html",
        transcripts=transcripts,
        bot_status=bot.status(),
        live=get_live_session(),
        bot_state=state,
        uptime=format_uptime(state.get("started_at")) if state else None,
        heartbeat_label=format_duration(hb),
        # True when the bot hasn't refreshed its state within the heartbeat
        # window — the flags below may be stale (e.g. a dead-but-"recording"
        # bot). The card warns instead of trusting them.
        state_stale=(hb is not None and hb > STATE_STALE_SECONDS),
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


@app.route("/roster")
def roster():
    return render_template(
        "roster.html",
        roster=get_roster(),
        unmapped=get_unmapped_speakers(),
    )


def _parse_user_id(raw):
    """Coerce a posted user_id to a positive int, or None if invalid.

    Discord snowflakes are positive ints; reject anything else so a bad
    value never becomes a roster key.
    """
    try:
        uid = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return uid if uid > 0 else None


@app.route("/roster/entry", methods=["POST"])
def roster_upsert():
    _require_csrf_header()
    data = request.get_json(silent=True) or request.form
    uid = _parse_user_id(data.get("user_id"))
    if uid is None:
        return jsonify({"error": "user_id must be a positive integer"}), 400
    player = (data.get("player") or "").strip()
    character = (data.get("character") or "").strip()
    if not player or not character:
        return jsonify({"error": "player and character are required"}), 400
    try:
        player_map_store.upsert(roster_path(), uid, player, character)
    except ValueError as e:
        # File is valid YAML but not a mapping — refuse rather than clobber.
        return jsonify({"error": str(e)}), 400
    # user_id as a string: Discord snowflakes exceed JS's safe-integer range,
    # so a JSON number would lose precision on the client (Discord's own API
    # returns snowflakes as strings for the same reason).
    return jsonify({"user_id": str(uid), "player": player, "character": character})


@app.route("/roster/entry/delete", methods=["POST"])
def roster_delete():
    _require_csrf_header()
    data = request.get_json(silent=True) or request.form
    uid = _parse_user_id(data.get("user_id"))
    if uid is None:
        return jsonify({"error": "user_id must be a positive integer"}), 400
    try:
        deleted = player_map_store.delete(roster_path(), uid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"user_id": str(uid), "deleted": deleted})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="127.0.0.1", port=5001, debug=debug)
