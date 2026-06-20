# Changelog

All notable changes to `scitex-genai` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Modality-organised package layout: `llm` (implemented), `agent`, `image`,
  `audio`, `video`, `embed`, `multimodal` (reserved namespaces).
- `llm/`: provider factory (`GenAI`) over OpenAI, Anthropic, Google, Groq,
  DeepSeek, Perplexity, Llama. Lifted from `scitex-ai/_gen_ai`.
- Optional extras: `[agent]` (claude-agent-sdk), `[litellm]`, `[ollama]`.
- Smoke tests covering top-level import, lazy `GenAI`, modality submodule
  import, and the reserved-stub `NotImplementedError` contract.
