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
