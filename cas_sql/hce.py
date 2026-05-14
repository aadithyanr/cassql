"""
Heuristic Complexity Estimator (HCE).

Implements Algorithm 1 from the CAS-SQL paper. Pure-CPU, no learned parameters.
Each feature class contributes its weight at most once to the total score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


# Feature classes from Table II of the paper.
# Each pattern is a list of substrings; matching is whole-word where the
# pattern contains only letters (we use \b boundaries), otherwise raw substring.
FEATURE_PATTERNS: dict[str, tuple[list[str], int]] = {
    "aggregation": (
        ["total", "sum", "average", "count", "mean", "how many"],
        1,
    ),
    "superlative": (
        ["most", "least", "highest", "lowest", "best", "worst", "top", "bottom"],
        1,
    ),
    "multi_entity": (
        ["both", "either", "and ", "compare", "versus", " vs "],
        1,
    ),
    "negation": (
        ["not ", "except", "exclude", "without", "other than"],
        1,
    ),
    "temporal": (
        ["before", "after", "between", "since", "during", "prior"],
        1,
    ),
    "nested_intent": (
        ["among those", "from the ones", "of the ", " that have ", " that are "],
        2,
    ),
    "quantitative_threshold": (
        ["more than", "less than", "at least", "fewer than", "greater than", "at most"],
        1,
    ),
}


@dataclass
class HCEResult:
    tier: Tier
    score: int
    fired: dict[str, bool]  # which feature classes fired

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "score": self.score,
            "fired": self.fired,
        }


def _class_fires(text_lower: str, patterns: list[str]) -> bool:
    """Return True if any pattern in the list occurs in the text."""
    for p in patterns:
        if p in text_lower:
            return True
    return False


def estimate(question: str) -> HCEResult:
    """
    Estimate complexity tier for a natural-language question.

    Args:
        question: The user's natural-language question.

    Returns:
        HCEResult with tier, score, and a dict of which classes fired.
    """
    q = question.lower()
    # Normalize multiple whitespace and add trailing space so patterns like
    # "and " match end-of-string occurrences too.
    q = re.sub(r"\s+", " ", q).strip() + " "

    score = 0
    fired: dict[str, bool] = {}

    for class_name, (patterns, weight) in FEATURE_PATTERNS.items():
        if _class_fires(q, patterns):
            score += weight
            fired[class_name] = True
        else:
            fired[class_name] = False

    if score <= 1:
        tier = Tier.SIMPLE
    elif score == 2:
        tier = Tier.MEDIUM
    else:
        tier = Tier.COMPLEX

    return HCEResult(tier=tier, score=score, fired=fired)


# Spider's gold hardness uses 4 levels: easy / medium / hard / extra.
# For sanity-check evaluation we map our 3-tier output to Spider's 4-level scale.
# Conservative mapping: SIMPLE→easy, MEDIUM→medium, COMPLEX→hard-or-extra.
def tier_to_spider_hardness(tier: Tier) -> str:
    return {
        Tier.SIMPLE: "easy",
        Tier.MEDIUM: "medium",
        Tier.COMPLEX: "hard",  # also covers "extra"
    }[tier]


if __name__ == "__main__":
    # Smoke tests
    examples = [
        ("list all students", Tier.SIMPLE),
        ("how many students are there", Tier.SIMPLE),
        ("what is the average salary of employees", Tier.SIMPLE),
        ("show the top 5 departments by average salary", Tier.MEDIUM),
        (
            "find the top three departments by average salary excluding "
            "interns hired before 2020",
            Tier.COMPLEX,
        ),
        (
            "among those students who scored more than 80, which are the top 3?",
            Tier.COMPLEX,
        ),
    ]
    print(f"{'Question':<70} {'Expected':<8} {'Got':<8} {'Score':<5} Fired")
    print("-" * 110)
    for q, expected in examples:
        r = estimate(q)
        fired = ",".join(k for k, v in r.fired.items() if v)
        ok = "✓" if r.tier == expected else "✗"
        print(f"{q[:69]:<70} {expected.value:<8} {r.tier.value:<8} {r.score:<5} {fired} {ok}")
