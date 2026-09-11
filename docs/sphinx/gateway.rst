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
