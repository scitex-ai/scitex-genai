"""Payload-free reservation and settlement for external-provider requests."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ._errors import BudgetExceededError


class ExternalUsagePolicy(Protocol):
    provider: str
    canonical_model: str
    max_requests_per_run: int | None
    max_input_tokens_per_run: int | None
    max_output_tokens_per_run: int | None
    max_total_tokens_per_run: int | None
    max_estimated_usd_per_run: float | None
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


@dataclass
class UsageCounters:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    responses_with_usage: int = 0
    responses_without_usage: int = 0
    reported_model_mismatches: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ExternalUsageLedger:
    """Concurrency-safe, in-memory ceilings for one gateway incarnation."""

    def __init__(self, policy: ExternalUsagePolicy) -> None:
        self.policy = policy
        self.total = UsageCounters()
        self.runs: dict[str, UsageCounters] = {}
        self.last_reported_model = ""
        self._lock = asyncio.Lock()

    async def reserve(self, run_key: str, *, estimated_input: int, output: int) -> None:
        async with self._lock:
            counters = self.runs.setdefault(run_key, UsageCounters())
            projected = UsageCounters(
                requests=counters.requests + 1,
                input_tokens=counters.input_tokens,
                output_tokens=counters.output_tokens,
                reserved_input_tokens=counters.reserved_input_tokens + estimated_input,
                reserved_output_tokens=counters.reserved_output_tokens + output,
            )
            checks = (
                (self.policy.max_requests_per_run, projected.requests, "requests"),
                (
                    self.policy.max_input_tokens_per_run,
                    projected.input_tokens + projected.reserved_input_tokens,
                    "input tokens",
                ),
                (
                    self.policy.max_output_tokens_per_run,
                    projected.output_tokens + projected.reserved_output_tokens,
                    "output tokens",
                ),
                (
                    self.policy.max_total_tokens_per_run,
                    projected.input_tokens
                    + projected.reserved_input_tokens
                    + projected.output_tokens
                    + projected.reserved_output_tokens,
                    "total tokens",
                ),
            )
            for limit, value, label in checks:
                if limit is not None and value > limit:
                    raise BudgetExceededError(
                        f"External-provider run budget would exceed {label}: "
                        f"{value}/{limit}"
                    )
            if self.policy.max_estimated_usd_per_run is not None:
                projected_cost = (
                    (projected.input_tokens + projected.reserved_input_tokens)
                    * self.policy.input_usd_per_million_tokens
                    + (projected.output_tokens + projected.reserved_output_tokens)
                    * self.policy.output_usd_per_million_tokens
                ) / 1_000_000
                if projected_cost > self.policy.max_estimated_usd_per_run:
                    raise BudgetExceededError(
                        "External-provider run budget would exceed estimated cost: "
                        f"${projected_cost:.6f}/${self.policy.max_estimated_usd_per_run:.6f}"
                    )
            counters.requests += 1
            counters.reserved_input_tokens += estimated_input
            counters.reserved_output_tokens += output
            self.total.requests += 1
            self.total.reserved_input_tokens += estimated_input
            self.total.reserved_output_tokens += output

    async def settle(
        self,
        run_key: str,
        *,
        reserved_input: int,
        reserved_output: int,
        input_tokens: int | None,
        output_tokens: int | None,
        reported_model: str,
    ) -> None:
        async with self._lock:
            counters = self.runs[run_key]
            counters.reserved_input_tokens = max(
                0, counters.reserved_input_tokens - reserved_input
            )
            self.total.reserved_input_tokens = max(
                0, self.total.reserved_input_tokens - reserved_input
            )
            counters.reserved_output_tokens = max(
                0, counters.reserved_output_tokens - reserved_output
            )
            self.total.reserved_output_tokens = max(
                0, self.total.reserved_output_tokens - reserved_output
            )
            if input_tokens is None or output_tokens is None:
                input_value = (
                    reserved_input
                    if input_tokens is None
                    else max(0, int(input_tokens))
                )
                output_value = (
                    reserved_output
                    if output_tokens is None
                    else max(0, int(output_tokens))
                )
                counters.responses_without_usage += 1
                self.total.responses_without_usage += 1
            else:
                input_value = max(0, int(input_tokens))
                output_value = max(0, int(output_tokens))
                counters.responses_with_usage += 1
                self.total.responses_with_usage += 1
            counters.input_tokens += input_value
            counters.output_tokens += output_value
            self.total.input_tokens += input_value
            self.total.output_tokens += output_value
            if reported_model:
                self.last_reported_model = reported_model
                if reported_model != self.policy.canonical_model:
                    counters.reported_model_mismatches += 1
                    self.total.reported_model_mismatches += 1

    def snapshot(self) -> dict[str, Any]:
        usage = asdict(self.total)
        usage["total_tokens"] = self.total.total_tokens
        usage["estimated_cost_usd"] = round(
            (
                self.total.input_tokens * self.policy.input_usd_per_million_tokens
                + self.total.output_tokens * self.policy.output_usd_per_million_tokens
            )
            / 1_000_000,
            8,
        )
        return {
            "provider": self.policy.provider,
            "canonical_model": self.policy.canonical_model,
            "last_reported_model": self.last_reported_model or None,
            "usage": usage,
            "runs_observed": len(self.runs),
        }


__all__ = ["ExternalUsageLedger", "UsageCounters"]
