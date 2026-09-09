"""Score an answer by what its siblings also found, not by a flat count.

`fpg`'s comparison counts every element of the true graph alike. That is not
what separates a good answer from a lucky one. The service the incident text
already names is in the graph too — an episode that filters on the endpoint it
was handed scores a hit for restating the prompt, while the injected service it
had to dig for counts exactly the same. Measured over the fifty collected
episodes, the part of a naive process signal that tracked the outcome best was
precisely that free part.

`n_samples` rollouts of one prompt are enough to tell the two apart without any
external label. Build the matrix of which sample found which element and read
the columns: an element every sibling found was free, an element one sibling
found was the hard part of that case. Weighting recall by how few found it is
the difficulty normalization from the coding-agent worked example, applied to
graph elements instead of test cases.

This is not process supervision. Nothing here judges a step; the whole thing is
a better-calibrated outcome, and every turn of a trajectory still carries the
same value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# Elements an answer is compared on, and the weight the flat score gives each.
AXES: tuple[tuple[str, float], ...] = (("roots", 0.4), ("subjects", 0.3), ("edges", 0.3))


@dataclass(frozen=True)
class Answer:
    """What one sample claimed, as sets of elements, against what was true."""

    found: dict[str, frozenset[str]]
    truth: dict[str, frozenset[str]]
    claimed: dict[str, frozenset[str]] = field(default_factory=dict)


def element_weights(answers: Sequence[Answer], axis: str) -> dict[str, float]:
    """How much each true element is worth, from how few samples found it.

    `1 - found/k`, so an element every sample found is worth nothing and one
    that a single sample found is worth almost everything. An element no sample
    found keeps full weight: it is the hardest, not the least relevant.
    """
    truths: set[str] = set()
    for answer in answers:
        truths |= answer.truth.get(axis, frozenset())
    if not answers:
        return dict.fromkeys(truths, 1.0)
    return {
        element: 1.0 - sum(1 for a in answers if element in a.found.get(axis, ())) / len(answers)
        for element in truths
    }


def weighted_score(answer: Answer, weights: dict[str, dict[str, float]]) -> float:
    """One answer's score, with recall weighted by difficulty.

    Recall is weighted and precision is not. A true element carries a
    difficulty because the siblings measured one; a claim that is not in the
    graph carries none — it is simply wrong, and wrong at the same price
    wherever it lands.
    """
    total = 0.0
    for axis, share in AXES:
        truth = answer.truth.get(axis, frozenset())
        found = answer.found.get(axis, frozenset())
        claimed = answer.claimed.get(axis, found)
        if not truth:
            continue
        axis_weights = weights.get(axis, {})
        mass = sum(axis_weights.get(e, 1.0) for e in truth)
        hit = sum(axis_weights.get(e, 1.0) for e in found)
        # Every element of this axis was free for every sibling: it separates
        # nothing, so it neither rewards nor punishes.
        recall = hit / mass if mass > 0 else 1.0
        precision = len(found) / len(claimed) if claimed else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        total += share * f1
    return total


__all__ = ["AXES", "Answer", "element_weights", "weighted_score"]
