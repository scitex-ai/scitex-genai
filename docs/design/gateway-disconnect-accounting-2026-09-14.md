# Gateway disconnect accounting incident (2026-09-14)

## Observed state

At 18:41 UTC on `scitex-compute-04`, the gateway reported one admitted request
(99,692 input tokens) and one queued request (99,976 input tokens). SGLang
simultaneously reported zero running requests, zero queued requests, and zero
token usage. The admitted request began at 18:23:24, but no matching completion
line exists. Its gateway-to-SGLang socket was in `FIN-WAIT-1` while the peer was
in `CLOSE-WAIT`. The last queued-client journal event at 18:38:33 was
`client_disconnected_before_admission`.

The release operation was after unbounded `response.aclose()` and
`client.aclose()` awaits. A transport stuck while closing therefore retained
an already-settled admission indefinitely. A duplicate late release could also
decrement global counters a second time because `max(0, ...)` hid the error.

## Permanent behavior

Confirmed ownership is now released before transport cleanup. Ambiguous engine
ownership still transfers to the existing abort reaper; this change does not
guess that an active engine request is safe to release. For explicit sessions,
the active-session receipt makes release idempotent, so a late duplicate cannot
debit a different active request. Queued cancellation removes counters only
when its ticket is still owned by that queue.

Tests cover queued ASGI disconnect token counters, duplicate admitted release,
and a transport close that never returns. In the latter case admission health
converges to zero before socket cleanup completes.

## Separate Hermes defect

The Hub TUI remained in `compacting` after interruption. Its Hermes process held
a client socket in `CLOSE-WAIT` and did not unwind the in-progress provider
call. Gateway accounting can be repaired independently, but Hermes/SAC must
also cancel and join the provider task when an interactive turn is interrupted;
otherwise the TUI can remain busy even after gateway and SGLang are idle.
