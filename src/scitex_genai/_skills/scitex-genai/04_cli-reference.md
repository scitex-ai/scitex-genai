---
description: |
  [TOPIC] CLI Reference
  [DETAILS] Start the authenticated Anthropic-compatible Codex subscription gateway with scitex-genai-gateway or python -m scitex_genai.
tags: [scitex-genai-cli-reference]
---

# CLI Reference

## Gateway

```bash
scitex-genai-gateway \
  --host 127.0.0.1 \
  --port 8765 \
  --log-level info
```

`python -m scitex_genai` accepts the same arguments.

For a deployed inference gateway, drain before restarting:

```bash
scitex-genai-gateway restart-unit --drain-timeout-s 1800
```

`restart-unit` crosses the live gateway's atomic admission barrier before it
invokes systemd. During the bounded wait `/health` returns HTTP 503 with
`status: draining` and `ready: false`. A timeout refuses the restart and leaves
admission closed; explicitly `POST /admin/resume` only after abandoning the
release.

The barrier returns success only after both `in_flight` and `queued` ownership
are zero; the command then restarts the systemd user unit while admission is
still closed.

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | HTTP bind address. |
| `--port` | `8765` | HTTP bind port. |
| `--codex-base-url` | `https://chatgpt.com/backend-api` | Codex-compatible upstream base URL. |
| `--log-level` | `info` | Uvicorn log level. |

The command requires `SCITEX_GENAI_GATEWAY_API_KEY`. Account locations come
from `~/.scitex/agent-container/accounts/openai/*/auth.json` when that store
exists. The path-separated `SCITEX_GENAI_CODEX_HOMES` variable is an explicit
override. Without a provider store or override, discovery uses `CODEX_HOME`
and then `~/.codex`.

See [06_http-api.md](06_http-api.md) for Claude Code configuration, endpoint
coverage, account rotation, and security boundaries.
