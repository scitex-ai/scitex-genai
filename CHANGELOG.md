# Changelog

All notable changes to `scitex-genai` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-05

### Added

- Gateway: relay of Anthropic `/v1/messages` to a pool of local inference
  upstreams (vLLM / LiteLLM) with sticky per-conversation selection and
  health-aware reselection around a dead member (`--inference-upstream`,
  `HOIST_UPSTREAM`). Replaces the hand-run hoist proxy.
- Gateway: settings from `~/.scitex/genai/config.yaml` through scitex-config
  (direct -> file -> environment -> default), and
  `scitex-genai-gateway install-unit`, which writes, reloads and enables the
  systemd user unit from the package (login-shell ExecStart so the profile
  supplies `SCITEX_GENAI_GATEWAY_API_KEY`).
- Serve surface for local model engines (`scitex_genai.serve`): engine confs
  from `~/.scitex/genai/models.d/<key>.conf` (bash-style, `${NAME:-default}`
  expanded), site settings from the `serve:` section, a pure launch renderer
  (engine env with a per-engine node-local cache, vLLM argv, generated LiteLLM
  sidecar config, un-multiplexed reverse tunnel), progress-bounded readiness
  (no fixed timeout while the JIT cache is still growing), a supervising
  runner, the node-side `scitex-genai-serve <key>` console script, and the
  fleet-side `scitex-genai-serve launch` that renders a lease hold body and
  books a persistent scitex-hpc lease with it (`[serve]` extra).

### Changed

- The gateway's upstream pool is backend-neutral (`StickyPool`), shared by the
  Codex-account and inference-upstream backends.
- `scitex-config` and, for the `serve` extra, `scitex-hpc` are declared
  dependencies.

## [0.1.4] - 2026-07-19

### Added

- Provider-qualified Codex account discovery from the SAC account store at
  `~/.scitex/agent-container/accounts/openai/*/auth.json`.

### Changed

- New Codex sessions always pass through quota-aware, least-loaded random
  account selection, including a one-account candidate pool.
- A present but empty or invalid provider account store now fails gateway
  startup loudly instead of silently skipping credentials or changing source.

## [0.1.3] - 2026-07-19

### Fixed

- Claude Code streamed requests now preserve full internal session stickiness
  while satisfying Codex's 64-character cache-key and session-header limits.
- Streamed Codex HTTP errors now surface their actual upstream status and
  message instead of raising `httpx.ResponseNotRead` and leaving the harness
  retrying an opaque broken stream.

## [0.1.2] - 2026-07-19

### Added

- Opt-in LiteLLM dispatch backend for unified provider and self-hosted model
  routing.
- Authenticated Anthropic Messages gateway that keeps Claude Code as the agent
  harness while using Codex subscription accounts as the model backend.
- Multi-account Codex OAuth refresh, quota-aware scheduling, sticky sessions,
  concurrent load spreading, and 401/429/5xx failover.
- Streaming text, image, and tool-call translation between Anthropic Messages
  and Codex Responses protocols.

- Modality-organised package layout: `llm` (implemented), `agent`, `image`,
  `audio`, `video`, `embed`, `multimodal` (reserved namespaces).
- `llm/`: provider factory (`GenAI`) over OpenAI, Anthropic, Google, Groq,
  DeepSeek, Perplexity, Llama. Lifted from `scitex-ai/_gen_ai`.
- Optional extras: `[agent]` (claude-agent-sdk), `[litellm]`, `[ollama]`.
- Smoke tests covering top-level import, lazy `GenAI`, modality submodule
  import, and the reserved-stub `NotImplementedError` contract.
