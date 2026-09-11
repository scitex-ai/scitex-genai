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

Configure only active, reachable upstreams. A stale port is not spare
capacity: requests pinned to it can wait and then fail while a real member is
busy.

Tensor parallelism is capacity allocation, not replication. In the current
two-H100 deployment, one ``TP=2`` Qwen instance occupies both GPUs and is
therefore one gateway member. A second replica requires a second two-H100
allocation; do not configure a second member until that independently
allocated server is reachable.

The engine's request limit (for example SGLang
``--max-running-requests``) remains authoritative. This release does not
claim gateway admission-control semantics: adding a queue must preserve
conversation stickiness and prefix-cache locality, propagate cancellation
while waiting and streaming, and drain safely during shutdown.
