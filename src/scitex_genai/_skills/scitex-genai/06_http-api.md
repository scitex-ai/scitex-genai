---
description: |
  [TOPIC] HTTP API
  [DETAILS] Authenticated Anthropic Messages API backed by one or more Codex subscription accounts, including OAuth refresh, tool round trips, streaming, quota-aware routing, and Claude Code configuration.
tags: [scitex-genai-http-api]
---

# Claude Code model gateway

`scitex-genai-gateway` exposes the Anthropic Messages API while using the Codex
subscription Responses transport upstream. Claude Code remains the harness:
the gateway does not interpret hooks, load skills, run tools, edit files, or
manage a second agent loop.

## Install and configure

```bash
pip install 'scitex-genai[gateway]'

export SCITEX_GENAI_CODEX_HOMES="$HOME/.codex-alpha:$HOME/.codex-beta"
export SCITEX_GENAI_GATEWAY_API_KEY="<random local relay key>"
scitex-genai-gateway --host 127.0.0.1 --port 8765
```

Each directory in `SCITEX_GENAI_CODEX_HOMES` must contain an `auth.json`
created by `codex login`. Do not put tokens in a spec, `.env.example`, test, or
package file. The directory basename is used as the account alias; public
identity is provider-qualified as `openai:<alias>`.

Configure Claude Code:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8765"
export ANTHROPIC_API_KEY="$SCITEX_GENAI_GATEWAY_API_KEY"
export ANTHROPIC_MODEL="gpt-5.4"
claude
```

## Implemented behavior

- `POST /v1/messages`, streaming and non-streaming
- text, base64/URL images, system blocks, and model parameters
- tool declarations, `tool_use`, and `tool_result` round trips
- Codex OAuth refresh with atomic `auth.json` replacement
- sticky sessions and prompt-cache keys
- quota-window polling, least-used selection, concurrent spreading
- 429 cooldown and retry through another configured account
- authenticated inbound requests; health response exposes no identities

`POST /v1/messages/count_tokens` currently returns a conservative byte-based
estimate. It is not a billing or exact context-window measurement.

## Boundary

The ChatGPT Codex transport is distinct from the public OpenAI API. This module
uses the transport contract exercised by Codex clients; it must be treated as a
versioned adapter and covered by live smoke tests before production rollout.
