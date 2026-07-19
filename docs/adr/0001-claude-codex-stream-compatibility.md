# ADR 0001: Claude Code to Codex streaming compatibility

Status: accepted

## Context

Claude Code uses Anthropic streaming requests and may supply a session identity
longer than 64 characters. The Codex Responses transport accepts at most 64
characters for both `prompt_cache_key` and its `session_id` header. Forwarding
the harness value unchanged causes an explicit HTTP 400 response.

An upstream failure received through `httpx.AsyncClient.stream()` also has an
unread response body. Calling `response.json()` before consuming that body
raises `httpx.ResponseNotRead`, hides the upstream diagnostic, and leaves the
Claude harness retrying an opaque failed stream.

## Decision

- Preserve the original Claude session identity inside the account pool so
  sticky account selection retains its full input.
- At the Codex protocol boundary only, keep session identities of 64 characters
  or fewer unchanged and replace longer values with their deterministic
  64-character SHA-256 hexadecimal digest.
- Apply the same normalization to the Responses `prompt_cache_key` and Codex
  `session_id` header.
- Explicitly consume streamed error bodies before decoding them, then raise the
  typed gateway error with the upstream status and message.

## Consequences

Claude Code sessions remain stable for load balancing while every forwarded
cache/session key satisfies the Codex contract. Invalid requests fail quickly
with the actual upstream explanation. There is no silent retry, account
fallback, or payload-field omission for non-transient protocol errors.
