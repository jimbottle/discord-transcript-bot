# Draft upstream report — Pycord `fix/voice-rec-2` (PR #3159)

Status: DRAFT, not yet posted. Written by the agent for discord-transcript-bot-309;
a human reviews and posts it (under their own account) at
https://github.com/Pycord-Development/pycord/issues/new — or as a review comment on
PR #3159, since the code only exists on that branch. Adjust the tone before posting.

Verified against branch head `cb46392` (2026-08-29); local pin was `a61fdc9`. Both
have the same code in `discord/voice/receive/reader.py`.

---

**Title:** voice receive: `UDPKeepAlive` busy-loops at 100% CPU when a send fails, and its `delay` looks like ms used as seconds

### Summary

Two small problems in `discord/voice/receive/reader.py::UDPKeepAlive.run` on
`fix/voice-rec-2`:

1. **Unbounded retry with no delay.** On any exception from `sendto`, the loop logs at
   DEBUG, calls `vc.wait_until_connected()` (which returns immediately while the voice
   client is connected) and `continue`s. If the send keeps failing while the client is
   still "connected", that is a tight loop: I measured ~6,500 iterations/s and a full
   core pegged, with nothing visible at INFO/WARNING.

   A persistent `sendto` failure on a connected client is realistic: on macOS a UDP
   socket whose local address went away (Wi-Fi roam, DHCP renewal, VPN toggle) raises
   `EADDRNOTAVAIL` on every send until the socket is recreated, while the voice
   websocket carries on, so `is_connected()` stays true. `ENETUNREACH`/`EHOSTUNREACH`
   during a network blip behave the same.

   We hit what looks like exactly this in production: a bot recording a silent channel
   burned 100% of one core for 40+ minutes; it dropped to 0% on `stop_recording()` while
   still connected, which is what scopes it to the threads `AudioReader.start()` starts.

2. **`delay: int = 5000` is passed to `time.sleep`**, i.e. 5,000 *seconds* between
   keep-alives. Presumably intended as milliseconds (5 s). As written the keep-alive is
   sent once at start and then every 83 minutes. `time.sleep` is also uninterruptible,
   so `stop()` leaves the thread parked for the remainder of the delay.

### Reproduction (no Discord needed)

```python
import time
from unittest.mock import MagicMock
from discord.voice.receive.reader import UDPKeepAlive

client = MagicMock()
client.wait_until_connected.return_value = True
client.is_connected.return_value = True
client._connection.socket.sendto.side_effect = OSError(49, "Can't assign requested address")

ka = UDPKeepAlive(client); ka.start()
time.sleep(2); ka.stop()
print(client._connection.socket.sendto.call_count)   # ~13,000 on an M-series Mac
```

### Suggested fix

```python
        except Exception as exc:
            _log.warning("Error while sending udp keep alive ...", exc_info=exc)  # visible
            vc.wait_until_connected()
            if not vc.is_connected():
                break
            if self._end_thread.wait(1.0):   # back off instead of spinning
                break
            continue
        else:
            self.counter += 1
            if self._end_thread.wait(self.delay):   # interruptible; and delay = 5 (seconds)
                break
```

Happy to open a PR if that shape is acceptable.

### Side observation (separate, not a CPU issue)

`JitterBuffer` with `pref_size=1` never releases the final packet of a burst until the
next packet for that SSRC arrives or the recording ends (`_pop_if_ready` requires
`len > pref_size`). For a transcription sink that means the last 20 ms of every
utterance is delivered at the *start* of the speaker's next utterance. Cosmetic for
audio, mildly confusing for segmenting sinks.

---

## What was checked and ruled out before writing this (for the reviewer)

Offline, with the real classes and synthetic RTP packets
(`PacketRouter` + `PacketDecoder` + `JitterBuffer` + `MultiDataEvent`): a clean
burst, a mid-burst packet loss, resumed speech, a duplicate packet, a destroyed
decoder, and loss right at the tail all measured **0.00 CPU-s over 2 s** of silence
afterwards — the router parks in `MultiDataEvent.wait()`. `SpeakingTimer`,
`SinkEventRouter` (0.5 s timed `get`) and `SocketReader` (30 s timed `select`) all
block properly on the happy path. `SocketReader._do_run` does share the same
retry-with-no-delay shape on its `select` error path (`continue` after
`ValueError/TypeError/OSError`), which would spin if `state.socket` were ever closed
while a listener is registered; not reproduced, mentioned for completeness.
