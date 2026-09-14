"""State and failures for one inference member's zero-loss cutover."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceMemberQuiesceState:
    """Admission-locked state for one member's zero-loss cutover."""

    alias: str
    quiesced: bool
    in_flight: int
    queued: int
    held: int

    @property
    def empty(self) -> bool:
        """Whether all pre-cutoff work is gone; held work is deliberately ignored."""
        return self.in_flight == 0 and self.queued == 0

    def as_dict(self) -> dict[str, str | bool | int]:
        """Return the stable operator response without transport addresses."""
        return {
            "member": self.alias,
            "quiesced": self.quiesced,
            "in_flight": self.in_flight,
            "queued": self.queued,
            "held": self.held,
        }


class InferenceMemberQuiesceTimeout(TimeoutError):
    """The member stays quiesced when its pre-cutoff barrier times out."""

    def __init__(self, state: InferenceMemberQuiesceState) -> None:
        self.state = state
        super().__init__(
            f"member {state.alias} quiesce deadline expired with "
            f"in_flight={state.in_flight} queued={state.queued}; member remains quiesced"
        )


class InferenceMemberResumeError(RuntimeError):
    """A quiesced member has not proven a safe new engine incarnation."""
