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
    session_info,
    transcript_path,
)

# roster_service put the project root on sys.path so the shared store
# (the bot's own roster writer) is importable from web/.
from src import player_map_store  # noqa: E402

# The Live page's ask box goes through the very module the bot's /ask uses,
# so the prompt, the transcript cache and the error messages are identical
# whether the question comes from Discord or from the dashboard.
from src.llm import claude  # noqa: E402
from src import question_log, sessions  # noqa: E402
import time  # noqa: E402

# Upper bound on a posted question; the transcript is the real payload.
MAX_QUESTION_CHARS = 2000

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

# The dashboard only ever listens on loopback. Requests must also *address*
# it as loopback: a DNS-rebinding page (attacker.example resolving to
# 127.0.0.1 after first load) arrives with Host/Origin set to the attacker's
# name, and that is what the Host check and the Origin check below reject.
# Both compare against these fixed values, never against the request's own
# Host header, which is attacker-controlled in exactly that scenario.
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5001
_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


def _allowed_hosts():
    port = app.config.get("DASHBOARD_PORT", DASHBOARD_PORT)
    bare = set(_LOOPBACK_NAMES)
    return bare | {f"{name}:{port}" for name in bare}


def _allowed_origins():
    return {f"http://{host}" for host in _allowed_hosts()}


@app.before_request
def _reject_foreign_host():
    if request.host not in _allowed_hosts():
        abort(403)


def _require_csrf_header():
    # CSRF defense for state-changing endpoints. Cross-origin <form> submissions
    # cannot set custom headers, so requiring X-Requested-With blocks the classic
    # form-CSRF attack. Additionally reject any request whose Origin is set and
    # isn't one of the loopback origins — defense in depth against a browser
    # extension or other same-machine actor that can set arbitrary headers.
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        abort(403)
    origin = request.headers.get("Origin")
    if origin and origin not in _allowed_origins():
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
        "transcript.html",
        filename=filename,
        entries=entries,
        raw=raw,
        ask_configured=claude.configured(),
        session=session_info(filename),
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
    data = get_live_session()
    # Lets the Live page disable its ask box, with the reason, instead of
    # finding out on the first question.
    data["ask_configured"] = claude.configured()
    data["session"] = session_info(data["filename"])
    return jsonify(data)


@app.route("/ask", methods=["POST"])
def ask_session():
    """Answer a question about a session transcript with Claude.

    JSON body: ``{"question": str, "filename": optional str}``. Without
    ``filename`` the newest session file (the live one) is used.
    Blocking for the length of the API call (tens of seconds at most);
    Flask's threaded dev server keeps serving the live poll meanwhile.
    Errors come back as ``{"error": msg}`` with a status that says why:
    400 bad question, 404 unknown transcript, 409 nothing transcribed
    yet, 502 Anthropic said no (``AskFailed.user_message``), 503 no key.
    """
    _require_csrf_header()
    data = request.get_json(silent=True) or {}
    question = str(data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return (
            jsonify(
                {"error": f"question is too long (max {MAX_QUESTION_CHARS} characters)"}
            ),
            400,
        )
    filename = data.get("filename")
    if filename is not None and not isinstance(filename, str):
        # _safe_path does substring tests on it; anything but a string is
        # an unknown transcript, not a 500.
        return jsonify({"error": "transcript not found"}), 404
    filename = filename or get_live_session()["filename"]
    path = transcript_path(filename) if filename else None
    if filename and path is None:
        return jsonify({"error": "transcript not found"}), 404

    def _log(status, started=None, **kw):
        question_log.record(
            source="dashboard",
            question=question,
            status=status,
            asker="dashboard",
            transcript_path=path,
            latency_s=(time.perf_counter() - started) if started else None,
            **kw,
        )

    if not claude.configured():
        _log(question_log.REJECTED, error=claude.NO_KEY_MSG)
        return jsonify({"error": claude.NO_KEY_MSG}), 503

    # The whole session the file belongs to (every recording of that
    # campaign night), its header line and its note folders.
    transcript, header, note_dirs, files = (
        sessions.ask_context(path) if path else ("", "", [], [])
    )
    if not transcript or not transcript.strip():
        no_data = "No transcript data in the current session yet."
        _log(question_log.REJECTED, error=no_data)
        return jsonify({"error": no_data}), 409

    started = time.perf_counter()
    try:
        result = claude.ask(transcript, question, header=header, note_dirs=note_dirs)
    except claude.AskFailed as e:
        app.logger.error("ask failed: %s (%r)", e.user_message, e.__cause__)
        _log(question_log.FAILED, started, error=e.user_message)
        return jsonify({"error": e.user_message}), 502
    answer = result.answer.strip()
    if not answer:
        _log(question_log.FAILED, started, error=claude.EMPTY_MSG, result=result)
        return jsonify({"error": claude.EMPTY_MSG}), 502
    _log(question_log.ANSWERED, started, answer=answer, result=result)
    app.logger.info(
        "ask answered by %s (input=%s, cached=%s, output=%s, notes=%s)",
        result.model,
        result.input_tokens,
        result.cached_tokens,
        result.output_tokens,
        result.notes_used,
    )
    return jsonify(
        {
            "filename": filename,
            "question": question,
            "answer": answer,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "cached_tokens": result.cached_tokens,
            "output_tokens": result.output_tokens,
            "notes_used": list(result.notes_used),
            "recordings": len(files),
        }
    )


def _history_filters():
    def arg(name):
        v = request.args.get(name, "").strip()
        return v or None

    return {
        "session_key": arg("session"),
        "campaign": arg("campaign"),
        "source": arg("source"),
        "status": arg("status"),
    }


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


@app.route("/questions")
def questions():
    filters = _history_filters()
    entries = question_log.read(**filters, limit=500)
    everything = question_log.read()
    lib = sessions_library()
    return render_template(
        "questions.html",
        entries=entries,
        filters=filters,
        campaigns=sorted({e["campaign"] for e in everything if e.get("campaign")}),
        session_keys=sorted(
            {e["session_key"] for e in everything if e.get("session_key")}, reverse=True
        ),
        campaign_name=lambda cid: lib.name("campaign", cid),
        sources=question_log.SOURCES,
        total=len(everything),
    )


@app.route("/questions/data")
def questions_data():
    try:
        limit = max(1, min(500, int(request.args.get("limit", "100"))))
    except ValueError:
        limit = 100
    return jsonify(question_log.read(**_history_filters(), limit=limit))


def sessions_library():
    from src import library

    return library.load()


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
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=debug)
