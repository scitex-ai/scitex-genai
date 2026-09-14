# Gateway request lifecycle observability

## Purpose

Gateway admission and SGLang scheduling describe different ownership layers.
On 2026-09-14 the live TP=2 Qwen3.8-27B gateway was observed first with three
admitted requests and two queued requests (1.57M estimated input tokens
admitted and 1.0M queued), and later with two admitted and four queued (875,426
admitted and 1,969,719 queued). At the same time SGLang sometimes exported
`num_running_reqs=1`. This is not a counting contradiction: gateway admission
begins before upstream dispatch and remains owned through response
streaming/confirmed abort, while SGLang's gauge describes only its current
scheduler state. Operators need a joinable request record before changing
either limit.

This change adds observations only. It does not alter admission order, token
capacity, SGLang flags, or a running service.

## Context ceiling is not resident capacity

The model's configured 1,000,000-token context ceiling is a per-request
protocol/model limit. It is not evidence that an engine can keep one such
request, much less several, resident in KV cache. A measured TP=1 H100 canary
started with that context configuration but reported
`max_total_num_tokens=563215` and `max_running=11`. Therefore a 1M
request cannot be treated as safely resident on that TP=1 engine merely because
the configured context accepts its shape. The running-request ceiling is also
not a promise that eleven maximum-context requests fit simultaneously.

For the live TP=2 service, the gateway's 1.6M token capacity is an admission
budget, not a discovery of engine KV capacity and not a substitute for the
engine generation's `max_total_num_tokens`. The lifecycle fields deliberately
report both the per-request admission charge and the total charged immediately
after admission. Deployment/canary evidence must keep these distinct:

1. configured context ceiling (request validity);
2. gateway token capacity (front-door safety policy);
3. SGLang `max_total_num_tokens` (engine resident-token capacity); and
4. SGLang `running`/`queued` (instantaneous scheduler state).

Do not route 1M-context agents to TP=1 until the request's actual token count is
below its measured resident capacity with an explicit safety margin. Do not
raise the TP=2 gateway budget from queue depth alone.

## Identity contract

SAC should attach these headers to every inference request:

- `X-SciTeX-Agent-ID`: the stable SAC agent identity (for example, the spec's
  agent name), unchanged across turns and process restarts.
- `X-SciTeX-Session-ID`: the stable logical conversation identity, unchanged
  across turns in one conversation.

The gateway immediately converts both to domain-separated, 12-hex-character
labels. Raw values are not retained, logged, returned, or forwarded upstream.
The agent header is a gateway hop-by-hop header; the opaque session digest is
also the cache-affinity key and, on supported SGLang routes, the engine
`session_id`. Callers that cannot supply an agent ID appear as `unknown`;
requests without an explicit/derived session appear as `anonymous`.

The response includes `X-SciTeX-Request-Label`, `X-SciTeX-Agent-Label`, and
`X-SciTeX-Session-Label`, allowing SAC to correlate its own event with the
gateway record without learning or reproducing the gateway's hash domains.

## Lifecycle contract

Each request has exactly one current `phase`:

- `admission_queued`: accepted by the gateway and awaiting routing/admission;
- `upstream_inflight`: owns gateway capacity and has been dispatched;
- `completed`: reached a clean terminal outcome or terminal relay error;
- `disconnected`: the client disconnected/cancelled before completion.

`queue_elapsed_s` is measured from gateway acceptance to the latest upstream
admission. `estimated_input_tokens` is the request estimate used for admission;
`admitted_input_tokens` is the amount charged for that request, and
`gateway_input_tokens_admitted` is the upstream's total immediately after the
admission. The record also carries the predicted uncached tokens, cache class,
sanitized upstream URL, and `gateway_capacity_owned`. The last field stays true
on a terminal client outcome while an abort cleanup reaper still owns the
admission slot, then flips to false only after confirmed release. When an
upstream response reports usage/cache details, the terminal record carries
reported input/output tokens, total and per-tier cached tokens, tier, and
storage backend.

Active records are never evicted. The authenticated `GET /admin/status`
returns active records plus the newest 256 terminal records under
`request_lifecycle`. Unauthenticated `GET /health` exposes aggregate
phase/terminal counters only. Stable labels and request-level token counts stay
behind authenticated `GET /admin/status` and the trusted local journal.
Journal lines use the same labels and write one `[request]` entry for every
phase transition. No prompt, completion, raw identity, authorization value, or
credential-bearing upstream URL enters these records.

## Admission correctness and fairness audit

The existing admission controller was reviewed against the measured state.

| Property | Finding |
|---|---|
| Request ownership | Admission increments before dispatch and releases only at clean EOF, confirmed abort, or a tracked cleanup handoff. Therefore gateway `admitted` may legitimately exceed SGLang `running`. |
| Token safety | The configured token capacity is checked before admission, and queued backfill must fit the remaining capacity. A single request above capacity is refused. |
| Capacity meaning | The gateway budget bounds estimated admitted input; it does not prove engine KV residency. Configured context, gateway budget, and SGLang resident capacity remain separate. |
| Session correctness | A stable session cannot have two active gateway tickets; later turns wait rather than losing cache affinity. |
| Queue bound | The queue count is checked under the same condition lock used to append/remove tickets. |
| Backfill fairness | A token-blocked head may be bypassed only a configured number of times (default four). After that, admission drains until the head fits. |
| Continuation fairness | Priority work may pass ordinary fitting work, but ordinary work ages into priority after the configured interval. |
| Known limitation | The gateway cannot preempt SGLang work already scheduled, and a gateway ticket does not imply an SGLang `running` slot. |

The smallest deterministic next change is operational, not a scheduler rewrite:
propagate both SAC labels, deploy this observation-only schema, and capture one
full contention window. Join `[request]` transitions with SGLang generation,
`running`, `queued`, and token-usage observations by time/upstream. Only then
adjust one admission knob. If the bounded-backfill audit shows unacceptable
head wait, the smallest follow-up is to expose the pool's existing
`max_admission_bypasses` constructor value in gateway configuration and run a
strict-FIFO canary with zero bypasses; do not infer that change from
`num_running_reqs` alone. Strict FIFO is deterministic and starvation-free but
can leave otherwise usable token capacity idle, so it is a canary proposal
rather than part of this PR.

## Deployment plan (not executed)

1. Merge and install the reviewed commit into a new immutable deployment
   directory. Run the gateway unit tests there; do not replace the live code.
2. Update SAC's inference client to send the two stable headers. Verify in a
   disposable gateway that the raw values are absent from upstream captures,
   `/admin/status`, `/health`, and journal output.
3. During an announced maintenance window, call authenticated
   `POST /admin/drain?timeout_s=1800` on the live gateway. Proceed only when it
   returns `draining=true`, `in_flight=0`, and `queued=0`. If it times out,
   leave admission closed for investigation or explicitly call
   `POST /admin/resume`; do not kill owned requests.
4. Point the systemd service at the immutable deployment and restart only the
   gateway. Do not restart SGLang. Verify `/health` is ready and the upstream
   engine generation is unchanged.
5. Submit one short canary request. Confirm its observable transitions,
   response labels, terminal cache fields when returned, zero raw identity,
   balanced admission counters, and the unchanged SGLang
   `max_total_num_tokens`. Resume the agent fleet gradually.
6. Roll back by draining the gateway again and restoring the prior immutable
   deployment. The API addition is backward compatible and does not require an
   SGLang rollback.
