# Changelog

All notable changes to `scitex-genai` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
