Inference gateway
=================

The gateway can relay Anthropic and OpenAI-compatible requests to one or more
local inference servers. Deployment settings belong in
``~/.scitex/genai/config.yaml``; a one-member pool is supported and should
list only the upstream that is actually reachable.

The request-phase and SAC identity contract is documented in
``docs/design/gateway-request-lifecycle-observability.md`` in the source tree.

.. code-block:: yaml

   gateway:
     host: 0.0.0.0
     port: 18772
     inference_upstreams:
       - label: qwen-tp2
         url: http://127.0.0.1:18773
         token_capacity: 1600000
       - label: qwen-tp1
         url: http://127.0.0.1:18774
         token_capacity: 500000
     inference_timeout_s: 1800
     inference_capacity_per_upstream: 8
     inference_max_queue_size: 128
     # Opt in only for a verified SGLang OpenAI endpoint with /abort_request.
     inference_continuation_qos_enabled: false
     inference_continuation_qos_max_retries: 1
     inference_continuation_qos_min_preempt_tokens: 400000
     # Opt in only when the OpenAI upstream implements SGLang's extension.
     inference_cache_report_enabled: true

``inference_timeout_s`` must be a finite number greater than zero. It
defaults to 600 seconds for backward compatibility. The legacy
``HOIST_TIMEOUT_S`` environment variable remains the first fallback when
the configuration field is absent. The automatically namespaced
``SCITEX_GATEWAY_INFERENCE_TIMEOUT_S`` remains supported after it. A
systemd drop-in is no longer needed:
``scitex-genai-gateway install-unit`` starts the gateway with this settings
file.

Command-line settings
---------------------

An operator may override the file for either a foreground run or a generated
unit:

.. code-block:: console

   $ scitex-genai-gateway --inference-timeout-s 1800
   $ scitex-genai-gateway install-unit --inference-timeout-s 1800

For ``install-unit``, shared settings may appear before or after the
subcommand; both forms produce the same unit.

The generated unit carries the latter value in its ``ExecStart`` command.
Direct values take precedence over the configuration file, then
``HOIST_TIMEOUT_S``, then ``SCITEX_GATEWAY_INFERENCE_TIMEOUT_S``, then the
600-second default.

Capacity planning
-----------------

Each local upstream is a mapping with exactly ``label``, ``url``, and
``token_capacity``. Labels and URLs must be unique, and token capacity must be
a positive integer measured for that engine topology. Bare URL lists and comma
strings are rejected: a uniform global token limit cannot safely describe a
mixed TP=1/TP=2 pool.

Migrate deterministically before installing the new gateway. Replace:

.. code-block:: yaml

   inference_upstreams:
     - http://127.0.0.1:18773
     - http://127.0.0.1:18774
   inference_token_capacity_per_upstream: 1600000

with explicit members:

.. code-block:: yaml

   inference_upstreams:
     - label: qwen-tp2
       url: http://127.0.0.1:18773
       token_capacity: 1600000
     - label: qwen-tp1
       url: http://127.0.0.1:18774
       token_capacity: 500000

The measured TP=1 engine reports ``max_total_num_tokens=563215``, already lower
than its one-million-token configured context ceiling; the example uses a
500,000-token gateway budget to retain headroom. Context validity is not
resident KV capacity. New sessions are placed
only on members that can ever fit their current estimated input. If a pinned
session grows beyond its member's hard capacity, it is deliberately repinned
to a capable member and pays one cold-cache turn; if no member fits, admission
fails before dispatch. Health reports each member's label, URL, and capacity.

The gateway admits at most ``inference_capacity_per_upstream`` concurrent
requests to each member (default 8). Additional requests wait in a bounded
pool-wide queue of ``inference_max_queue_size`` entries (default 128); once
that queue is full the gateway returns 503. Set the per-member value no higher
than the inference engine's own running-request limit.

Capacity eligibility happens before a new session's sticky placement; feasible
members are compared by admitted-plus-queued tokens divided by their own
``token_capacity``. A warm conversation still waits for its existing home
instead of moving to an idle replica and losing its prefix cache, unless its
grown request can never fit there. ``/health`` retains the ``upstreams`` URL list and adds a
``members`` list with each member's ``active``, ``in_flight``, ``queued``, and
``capacity`` state, plus pool-wide totals. Cancelled waiters release their
queue entries, and shutdown wakes all waiters while admitted streams drain.

Both admission settings may also be passed before or after ``install-unit``::

   scitex-genai-gateway --inference-capacity-per-upstream 8 \
     install-unit --inference-max-queue-size 128

Configure only active, reachable upstreams. A stale port is not spare
capacity: requests pinned to it can wait and then fail while a real member is
busy.

Tensor parallelism is capacity allocation, not replication. In the current
two-H100 deployment, one ``TP=2`` Qwen instance occupies both GPUs and is
therefore one gateway member. A second replica requires a second two-H100
allocation; do not configure a second member until that independently
allocated server is reachable.

The engine's request limit (for example SGLang
``--max-running-requests``) remains authoritative.

Drained deployment restart
--------------------------

Restart through the package command, not ``systemctl restart`` directly::

   scitex-genai-gateway restart-unit --drain-timeout-s 1800

The command enters a server-side admission barrier. Under the same lock that
owns admission counters, the gateway first marks itself draining and wakes
queued callers with a 503, then waits until both ``in_flight`` and ``queued``
are zero. A successful response leaves admission closed, so no request can
enter between the empty observation and the systemd restart. ``/health``
reports ``status: draining`` and ``ready: false`` with HTTP 503 throughout.

The wait is bounded by ``--drain-timeout-s``. An unreachable gateway,
malformed response, or expired deadline refuses the restart and leaves
admission closed (fail-closed) with exact remaining ownership counts. Use the
authenticated ``POST /admin/resume`` only after the failed release has been
deliberately abandoned. This prevents a deployment restart from cutting
through a long agent turn and forcing a cold prefix replay.

This guard is grounded in the 2026-09-12 deployment observation: a direct
gateway restart overlapped an active approximately 672,000-token Hub turn,
and its retry arrived without the prior prefix cache benefit. The drain
command turns that timing-dependent failure into a checked precondition.

Continuation handoff (opt in)
-----------------------------

``inference_continuation_qos_enabled`` defaults to false. When enabled, only
the canonical ``X-SciTeX-Session-ID`` header participates (legacy session
headers remain routing-only). A session is a ``continuation`` only after the
entire body of a 2xx upstream response reaches clean EOF;
body-derived affinity keys remain unclassified. This is conversation-history
QoS, not a claim that an engine prefix is currently resident.

For supported OpenAI routes, the gateway replaces any caller ``rid`` with a
fresh opaque SGLang request ID. If a proven continuation targets an upstream
where replay-safe cold work has not produced its first response-body byte, the
gateway posts that ID to the same upstream's ``/abort_request``, closes the old
transport, waits for its capacity release, runs the continuation, and then
replays the fully buffered body. Upstream headers do not end eligibility:
SGLang can send them before prefill produces a token. The first body byte does,
so work whose output may have reached the caller is never replayed. Retry count
is bounded by
``inference_continuation_qos_max_retries`` (default 1), and an unconfirmed
abort refuses the continuation instead of dispatching both requests together.
Eligible work is a first turn, a request classified as a large uncached
prefill, or a continuation whose last cache report came from host or storage.
It must also meet ``inference_continuation_qos_min_preempt_tokens``. A
predicted-cold or host/storage-restored continuation joins that replay-safe
victim class; only a predicted-hot continuation initiates a handoff or receives
continuation admission priority. The token threshold defaults to 0 for simple
semantics; deployments should set it from
measured harmful cold-prefill sizes (400,000 estimated tokens in the current
measured fleet), rather than making small requests pay an abort/replay cycle.
Predicted-cold work also waits while any non-cold request is already in flight
on its upstream. This protects decoding that has passed the replay boundary;
the existing bounded-bypass admission rule eventually drains new work so the
older cold ticket cannot starve.

The corresponding CLI/unit flags are ``--inference-continuation-qos`` (or
``--no-inference-continuation-qos``) and
``--inference-continuation-qos-max-retries`` plus
``--inference-continuation-qos-min-preempt-tokens``. ``/health`` reports only bounded,
content-free counters under ``continuation_qos``; all classification state is
ephemeral and disappears on gateway restart.

Client cancellation and stream errors use the same explicit abort before
capacity is released. If an abort cannot be confirmed and the original stream
cannot reach EOF, the reservation transfers to a visible background cleanup
reaper. It retries abort with bounded backoff until confirmation, then releases
the slot exactly once. Health counters expose pending reapers, attempts, and
recoveries; shutdown cancels and awaits those tasks without pretending the
engine state was reclaimed.

Request count is not enough when agents have very different context lengths.
For each member, ``token_capacity`` limits
the sum of estimated input tokens in flight on each upstream.  It estimates
one token per four UTF-8 request-body bytes, without loading a model tokenizer
into the gateway.  This is a planning approximation, so set the budget below
the engine's measured usable token capacity with enough margin for output and
estimation error.  A request larger than the configured budget is refused;
otherwise it waits in the same bounded queue as count-limited work.

Capacity eligibility happens before first sticky placement, and feasible new
sessions prefer the lowest normalized token pressure. A warm conversation
waits for room on its cache-owning upstream rather than moving merely because
another member is idle. It is repinned only when its request has grown beyond
that home's hard capacity. The total-token guard continues to account the full
estimated input because active sequence KV is not free.

When the cold-prefill limit and threshold are configured, that separate guard
uses predicted **uncached prefill tokens**.  Compatible lineage starts from the
previous response's actual cached-token report and adds request growth.  Missing
reports, changed lineage, and changed engine generation are unknown and budget
the full prompt.  Evidence and queued-hot classifications expire after 300
seconds so a delayed request cannot assume cache residency indefinitely.  The
existing request-count and total-token limits still apply independently.

For SGLang processes launched by ``scitex-genai serve``, the supervisor mints
a new opaque ``scitex_engine_generation`` label for every process start and
passes it through SGLang's ``--extra-metric-labels`` support.  Before making a
history-based prediction, the gateway reads that label atomically with
``sglang:num_running_reqs``, ``sglang:num_queue_reqs``, and
``sglang:token_usage`` from ``/metrics``.  A missing, timed-out, partial, or
mixed-generation metrics response makes the engine generation unavailable;
in that state actual cache reports remain observable but are never retained as
reusable prediction history.  Existing engines acquire the label at their next
normal supervised start; no engine restart is required during gateway rollout.

With the token guard enabled, ``/health`` adds
``input_tokens_in_flight``, ``input_tokens_queued``, and per-member
``token_capacity`` fields.  Each ``[relay] ... ->`` journal line also records
the request estimate and the admitted token total; payloads remain absent.
Stream completion lines distinguish ``outcome=complete``,
``outcome=client_disconnected``, and ``outcome=stream_error``.

Request-level cache observations
--------------------------------

Every inference journal entry records the estimated input tokens, queue wait,
time to the first upstream body byte (``ttft_s``), upstream and total latency,
and a domain-separated SHA-256 fingerprint of the first 16 KiB of the relayed
request.  The fingerprint exposes no prompt text and makes early-prefix drift
between otherwise related requests visible.

SGLang's Anthropic Messages stream reports ``cache_read_input_tokens`` without
request extensions; the completion journal records it as ``cached_tokens``.
For SGLang OpenAI Chat/Completions, enable
``inference_cache_report_enabled`` only after verifying that the upstream was
started with ``--enable-cache-report``.  The gateway then requests
``return_cached_tokens_details`` and ``stream_options.include_usage`` and logs
the upstream-reported ``device``, ``host``, and ``storage`` token counts.  The
option defaults to false because arbitrary OpenAI-compatible providers may
reject the SGLang-only request field.  Prompts, generated text, credentials,
and raw session identifiers are never journaled.

The gateway keeps the bounded admission predictor across a gateway-only
restart in ``~/.scitex/genai/runtime/admission-history.json``.  The file is an
atomic, mode-0600 handoff artifact, not conversation storage: it contains only
hashed session and upstream labels, lineage digests, token counts, cache tier,
the hashed engine generation, and timestamps.  Entries expire after 300
seconds.  The gateway restores an entry only after its engine probe supplies
the same authoritative generation; an absent or changed generation is a cold,
unknown prediction and never reuses the file.  This is deliberately local
rather than Postgres state: the hint describes one node-local KV-cache
incarnation and must not acquire cross-host availability or durability.

The gateway releases its reservation after closing a disconnected upstream
HTTP stream.  The inference engine must actually abort that request too.
SGLang regressions `#36333 <https://github.com/sgl-project/sglang/issues/36333>`_
and `#36876 <https://github.com/sgl-project/sglang/issues/36876>`_ describe
versions where tokenizer state is deleted but the scheduler keeps a zombie
request running.  Gateway admission cannot observe or safely reclaim that
engine-side state; deploy an SGLang build containing the upstream abort fix.

Heterogeneous rollout
---------------------

Do not add a TP=1 URL to a running gateway under the old uniform-capacity
configuration. First write the structured TP=2 member only, validate the file
without binding a port:

.. code-block:: console

   $ python -c 'from scitex_genai.gateway._settings import load_settings; print(load_settings())'

Then drain the gateway with
``scitex-genai-gateway restart-unit``, and confirm ``/health`` reports the TP=2
label, URL, and 1,600,000-token capacity. This migration changes the gateway
process only; it does not restart SGLang.

Start and probe the TP=1 canary separately. After its readiness and token-limit
measurements pass, append it with a conservative initial ``token_capacity`` of
500,000, drain/restart the gateway again, and confirm both member records before
sending canary traffic. Exercise requests immediately below and above 500,000:
the former may select TP=1 and the latter must select TP=2. A request above
1,600,000 must fail before dispatch.

Rollback is deterministic: drain, restore the last known-good structured file
containing only the TP=2 member, restart the gateway, and verify that health has
one configured member. Never roll back to the retired bare-URL/global-capacity
schema. Preserve the prior file beside the deployment as the rollback artifact.
