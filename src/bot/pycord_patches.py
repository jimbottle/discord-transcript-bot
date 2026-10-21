"""Guarded runtime patches for the pinned Pycord voice-receive branch
(``fix/voice-rec-2``, PR #3159) — see the DAVE note in CLAUDE.md.

Each patch (a) checks that the upstream code still has the shape it
expects and skips itself with a WARNING otherwise, so a branch update
can't turn a fix into a crash, and (b) is idempotent. Applied once from
``main.py`` at startup. Everything here is a workaround for something we
have reported (or drafted a report for) upstream; delete a patch the
moment the upstream fix lands in the pinned version.
"""

import inspect
import logging

logger = logging.getLogger(__name__)

# How long the keep-alive thread waits after a failed send before trying
# again. Upstream `continue`s immediately, which is a 100%-core spin for as
# long as the failure persists.
KEEPALIVE_FAILURE_BACKOFF_S = 1.0
# Log every Nth consecutive failure after the first, so a persistent fault
# stays visible without flooding.
_LOG_EVERY = 60

# Source fragments the patched method depends on. If any is missing the
# upstream loop changed and our replacement can no longer claim to mirror
# it, so we don't install it.
_EXPECTED_FRAGMENTS = (
    "self.client.wait_until_connected()",
    'self.counter.to_bytes(8, "big")',
    "vc._connection.socket.sendto(",
    "vc._connection.endpoint_ip",
    "vc._connection.voice_port",
    "vc.wait_until_connected()",
    "if vc.is_connected():",
    "time.sleep(self.delay)",
)


def _keepalive_run(self):  # mirrors discord.voice.receive.reader.UDPKeepAlive.run
    self.client.wait_until_connected()
    failures = 0

    while not self._end_thread.is_set():
        vc = self.client

        try:
            packet = self.counter.to_bytes(8, "big")
        except OverflowError:
            self.counter = 0
            continue

        try:
            vc._connection.socket.sendto(
                packet, (vc._connection.endpoint_ip, vc._connection.voice_port)
            )
        except Exception as exc:  # noqa: BLE001 - upstream catches Exception too
            failures += 1
            if failures == 1 or failures % _LOG_EVERY == 0:
                # Upstream logs this at DEBUG, i.e. invisibly at the bot's
                # log level — the whole reason discord-transcript-bot-309
                # could not be root-caused from the logs.
                logger.warning(
                    "Voice UDP keep-alive send failed (%s: %s) — %d consecutive; "
                    "backing off %.1fs instead of retrying immediately",
                    type(exc).__name__,
                    exc,
                    failures,
                    KEEPALIVE_FAILURE_BACKOFF_S,
                )
            vc.wait_until_connected()
            if not vc.is_connected():
                break
            # The one behavioural change: upstream `continue`s here with no
            # delay. Wait on the end event so /stop still exits promptly.
            if self._end_thread.wait(KEEPALIVE_FAILURE_BACKOFF_S):
                break
        else:
            if failures:
                logger.info(
                    "Voice UDP keep-alive recovered after %d failure(s)", failures
                )
                failures = 0
            self.counter += 1
            # Upstream: time.sleep(self.delay). Same duration, but
            # interruptible by stop() so a finished recording doesn't leave
            # the thread parked for the rest of the delay.
            if self._end_thread.wait(self.delay):
                break


def patch_udp_keepalive_backoff() -> bool:
    """Install the backoff on ``UDPKeepAlive.run``. Returns True when the
    patch is (already) in place, False when skipped."""
    try:
        from discord.voice.receive import reader as _reader
    except Exception as e:  # noqa: BLE001 - not this Pycord
        logger.debug("keep-alive backoff patch skipped: %s", e)
        return False

    cls = getattr(_reader, "UDPKeepAlive", None)
    if cls is None:
        logger.debug("keep-alive backoff patch skipped: no UDPKeepAlive class")
        return False
    if getattr(cls, "_volo_backoff_patched", False):
        return True

    try:
        src = inspect.getsource(cls.run)
    except (OSError, TypeError):
        src = ""
    missing = [frag for frag in _EXPECTED_FRAGMENTS if frag not in src]
    if missing:
        logger.warning(
            "Pycord UDPKeepAlive.run no longer matches the expected shape "
            "(missing %s); keep-alive backoff patch NOT applied — re-check "
            "discord-transcript-bot-309 against this Pycord version",
            ", ".join(repr(m) for m in missing),
        )
        return False

    cls._volo_original_run = cls.run
    cls.run = _keepalive_run
    cls._volo_backoff_patched = True
    logger.info(
        "Patched Pycord UDPKeepAlive.run: %.1fs backoff on send failure "
        "(discord-transcript-bot-309)",
        KEEPALIVE_FAILURE_BACKOFF_S,
    )
    return True


def apply_all() -> dict:
    """Apply every patch; returns {name: applied}. Never raises."""
    results = {}
    for name, fn in (("udp_keepalive_backoff", patch_udp_keepalive_backoff),):
        try:
            results[name] = bool(fn())
        except Exception as e:  # noqa: BLE001 - a patch must never block startup
            logger.warning("Pycord patch %s failed to apply: %s", name, e)
            results[name] = False
    return results
