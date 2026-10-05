# Provider key availability

`scitex_genai.availability.probe_provider_key` checks a key against the
provider's inference API with a tiny synthetic request. It never sends an
agent conversation or changes data-use settings. A models-list request alone
cannot establish whether a key has credits or the selected model's entitlement.

```python
import os
from scitex_genai.availability import probe_provider_key

result = probe_provider_key("opencode-go", "kimi-k3", os.environ["OPENCODE_GO_API_KEY_3"])
print(result.to_dict())
```

The result uses `scitex_dev.status.Check` with a `StatusCode` cause. HTTP
codes are retained verbatim, including 400, 401, 402, 403 and 429. A timeout
reports `errno/ETIMEDOUT`, with an unknown availability verdict. It does not
become an invented HTTP 504. `available` is True, False or None; callers
should use only True as evidence of current inference access. `reset_at`
preserves a provider reset timestamp or Retry-After value when supplied.

OpenCode Go/Zen and Command Code have built-in official routes. Other
providers and local relays supply `endpoint_url` and `protocol` explicitly.
OpenCode's per-model protocols follow its [official endpoint table](https://opencode.ai/docs/go/#endpoints).
Requests carry a client user agent and stable session header. Redirects are
refused so a bearer is not forwarded to another endpoint. Results contain no
API key or arbitrary upstream response body.

```bash
python -m scitex_genai.availability --provider opencode-go --model kimi-k3 --key-env OPENCODE_GO_API_KEY_3
```

Key ordering, quarantine, cooldowns and engine failover belong to the caller.
SAC keeps rejected credentials in its spec, skips them in persistent runtime
state, and retries replaced keys or keys whose quota window has reopened.
