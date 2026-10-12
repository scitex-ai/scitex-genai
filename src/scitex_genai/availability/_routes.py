"""Official API surfaces, or an explicitly configured compatible endpoint."""

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProviderRoute:
    endpoint_url: str
    protocol: str


def provider_route(
    provider: str,
    model: str,
    *,
    endpoint_url: str | None = None,
    protocol: str | None = None,
) -> ProviderRoute:
    """Resolve the inference surface used by a small availability request.

    OpenCode's per-model protocols are published at
    https://opencode.ai/docs/go/#endpoints. Explicit endpoint/protocol pairs
    cover local relays and other OpenAI/Anthropic compatible providers.
    """
    if endpoint_url is None:
        if provider in ("opencode-go", "opencode-zen", "opencode"):
            if model.startswith(("muse-spark-", "gpt-", "grok-")):
                protocol, suffix = "openai-responses", "responses"
            elif model.startswith(("minimax-", "qwen3.")):
                protocol, suffix = "anthropic-messages", "messages"
            else:
                protocol, suffix = "openai-chat-completions", "chat/completions"
            root = "https://opencode.ai/zen" + (
                "/go" if provider == "opencode-go" else ""
            )
            endpoint_url = root + "/v1/" + suffix
        elif provider == "command-code":
            endpoint_url = "https://api.commandcode.ai/provider/v1/chat/completions"
            protocol = "openai-chat-completions"
        else:
            raise ValueError("Unknown provider; supply its endpoint_url and protocol")
    parsed = urlsplit(endpoint_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Provider endpoint must be an HTTP URL without embedded credentials or query"
        )
    if protocol not in (
        "openai-chat-completions",
        "openai-responses",
        "anthropic-messages",
    ):
        raise ValueError("Declare the provider's inference protocol")
    return ProviderRoute(endpoint_url, protocol)
