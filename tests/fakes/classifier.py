"""A classifier with no model file and no randomness."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from autopilot.domain.models import ClassificationResult, ComplexityTier, PromptFeatures


def _one_hot(tier: ComplexityTier, confidence: float) -> dict[ComplexityTier, float]:
    """Spread the remaining mass evenly over the other tiers.

    The domain type refuses probabilities that do not sum to one, which is
    exactly the sort of thing a lazily-built fixture gets wrong.
    """
    others = [t for t in ComplexityTier if t is not tier]
    share = (1.0 - confidence) / len(others)
    dist = dict.fromkeys(others, share)
    dist[tier] = confidence
    return dist


@dataclass
class FakeClassifier:
    """Returns whatever the test tells it to.

    Either a fixed tier, or a rule over the feature vector when the test needs
    behaviour rather than a constant.
    """

    tier: ComplexityTier = ComplexityTier.SIMPLE
    confidence: float = 0.9
    version: str = "fake-1"
    rule: Callable[[PromptFeatures], tuple[ComplexityTier, float]] | None = None
    seen: list[PromptFeatures] = field(default_factory=list)

    def classify(self, features: PromptFeatures) -> ClassificationResult:
        self.seen.append(features)
        tier, confidence = self.rule(features) if self.rule else (self.tier, self.confidence)
        return ClassificationResult(
            tier=tier,
            confidence=confidence,
            probabilities=_one_hot(tier, confidence),
            classifier_version=self.version,
        )
