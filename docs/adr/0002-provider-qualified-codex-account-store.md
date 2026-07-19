# ADR 0002: Provider-qualified Codex account store

## Status

Accepted on 2026-07-19.

## Context

SAC may hold Claude Code and OpenAI Codex subscriptions whose human-derived
account slugs are identical. A flat credential namespace cannot represent
those as distinct identities. A single configured Codex account must also use
the same selection behavior as a multi-account pool so adding accounts does
not change the execution path.

## Decision

The default Codex gateway store is
`~/.scitex/agent-container/accounts/openai/<alias>/auth.json`, and public
identity is `openai:<alias>`. `SCITEX_GENAI_CODEX_HOMES` remains an explicit
override.

After quota and concurrent-load filtering, every new non-sticky session calls
the random choice function with the eligible account list. A list containing
one account is not special-cased. Invalid configured credentials and a present
but empty provider store are startup errors.

## Consequences

Provider identities cannot collide, one-account and multi-account deployments
exercise the same scheduler, and broken account configuration is visible at
startup. Codex OAuth tokens remain only in `auth.json` files and are never
placed in specs, examples, or package metadata.
