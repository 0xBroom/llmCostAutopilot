"""Token counting without a tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from autopilot.domain.models import Message, ModelConfig


@dataclass
class HeuristicTokenCounter:
    """chars / 4, plus a small per-message overhead.

    Wrong by roughly 10-20% against a real tokenizer, and that is fine: no test
    in the core suite should care about the exact number. Tests that do care —
    context-window boundary behaviour — belong in the contract tier against the
    real adapter.

    The value here is that the core suite needs no vendor package installed and
    runs in milliseconds.
    """

    chars_per_token: int = 4
    per_message_overhead: int = 4
    calls: list[tuple[int, str]] = field(default_factory=list)

    def count(self, messages: Sequence[Message], model: ModelConfig) -> int:
        total = sum(
            len(m.content) // self.chars_per_token + self.per_message_overhead for m in messages
        )
        self.calls.append((total, model.key))
        return total
