Inference gateway
=================

The gateway can relay Anthropic and OpenAI-compatible requests to one or more
local inference servers. Deployment settings belong in
``~/.scitex/genai/config.yaml``; a one-member pool is supported and should
list only the upstream that is actually reachable.

.. code-block:: yaml

   gateway:
     host: 0.0.0.0
     port: 18772
     inference_upstreams:
       - http://127.0.0.1:18773
     inference_timeout_s: 1800
     inference_capacity_per_upstream: 8
     inference_max_queue_size: 128
     # Optional weighted guard; choose from measured engine KV capacity.
     inference_token_capacity_per_upstream: 1600000
     # Opt in only for a verified SGLang OpenAI endpoint with /abort_request.
     inference_continuation_qos_enabled: false
     inference_continuation_qos_max_retries: 1
     inference_continuation_qos_min_preempt_tokens: 400000

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

The gateway admits at most ``inference_capacity_per_upstream`` concurrent
requests to each member (default 8). Additional requests wait in a bounded
pool-wide queue of ``inference_max_queue_size`` entries (default 128); once
that queue is full the gateway returns 503. Set the per-member value no higher
than the inference engine's own running-request limit.

Admission happens after sticky placement. A conversation therefore waits for
its existing home member instead of moving to an idle replica and losing its
prefix cache. ``/health`` retains the ``upstreams`` URL list and adds a
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
where a first turn is still waiting for response headers, the gateway posts
that ID to the same upstream's ``/abort_request``, closes the old transport,
waits for its capacity release, runs the continuation, and then replays the
fully buffered first-turn body. It never preempts after response headers have
been exposed. Retry count is bounded by
``inference_continuation_qos_max_retries`` (default 1), and an unconfirmed
abort refuses the continuation instead of dispatching both requests together.
Only first turns at or above
``inference_continuation_qos_min_preempt_tokens`` are eligible. That threshold
defaults to 0 for simple semantics; deployments should set it from measured
harmful cold-prefill sizes (400,000 estimated tokens in the current measured
fleet), rather than making small requests pay an abort/replay cycle.

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
When ``inference_token_capacity_per_upstream`` is set, the gateway also limits
the sum of estimated input tokens in flight on each upstream.  It estimates
one token per four UTF-8 request-body bytes, without loading a model tokenizer
into the gateway.  This is a planning approximation, so set the budget below
the engine's measured usable token capacity with enough margin for output and
estimation error.  A request larger than the configured budget is refused;
otherwise it waits in the same bounded queue as count-limited work.

Weighted admission happens *after* sticky placement.  A warm conversation
therefore waits for room on its cache-owning upstream rather than moving to an
idle replica and paying a cold prefill.  Cache hits reduce prefill work but do
not make active sequence KV free, so the guard accounts the full estimated
input instead of discounting a presumed cached prefix.

With the token guard enabled, ``/health`` adds
``input_tokens_in_flight``, ``input_tokens_queued``, and per-member
``token_capacity`` fields.  Each ``[relay] ... ->`` journal line also records
the request estimate and the admitted token total; payloads remain absent.
Stream completion lines distinguish ``outcome=complete``,
``outcome=client_disconnected``, and ``outcome=stream_error``.

The gateway releases its reservation after closing a disconnected upstream
HTTP stream.  The inference engine must actually abort that request too.
SGLang regressions `#36333 <https://github.com/sgl-project/sglang/issues/36333>`_
and `#36876 <https://github.com/sgl-project/sglang/issues/36876>`_ describe
versions where tokenizer state is deleted but the scheduler keeps a zombie
request running.  Gateway admission cannot observe or safely reclaim that
engine-side state; deploy an SGLang build containing the upstream abort fix.
