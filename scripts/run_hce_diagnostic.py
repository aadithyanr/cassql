"""
HCE diagnostic: how well does the heuristic estimator agree with Spider's
gold hardness labels?

Maps:
  Spider gold {easy} -> SIMPLE
  Spider gold {medium} -> MEDIUM
  Spider gold {hard, extra} -> COMPLEX

Outputs:
  - Confusion matrix (3x3)
  - Per-tier accuracy and Cohen's kappa
  - Distribution of predicted tiers
  - JSONL with per-example predictions

Runs on CPU in seconds. NO LLM needed.

Usage:
  python scripts/run_hce_diagnostic.py --spider-root /path/to/spider --out results/hce_diag.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from cas_sql.hce import Tier, estimate
from eval.datasets import load_spider_dev, spider_hardness_from_sql


def gold_to_tier(gold_hardness: str) -> Tier:
    if gold_hardness == "easy":
        return Tier.SIMPLE
    if gold_hardness == "medium":
        return Tier.MEDIUM
    return Tier.COMPLEX  # hard, extra


def cohens_kappa(pred: list[str], gold: list[str], labels: list[str]) -> float:
    """Cohen's kappa for two raters with the same label set."""
    n = len(pred)
    if n == 0:
        return 0.0
    # observed agreement
    po = sum(1 for p, g in zip(pred, gold) if p == g) / n
    # expected agreement
    pe = 0.0
    for lab in labels:
        p_marg = pred.count(lab) / n
        g_marg = gold.count(lab) / n
        pe += p_marg * g_marg
    return 0.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/hce_diagnostic.json"))
    ap.add_argument("--per-example-jsonl", type=Path,
                    default=Path("results/hce_per_example.jsonl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    examples = load_spider_dev(args.spider_root)
    print(f"Loaded {len(examples)} Spider dev examples")

    pred_tiers: list[str] = []
    gold_tiers: list[str] = []
    per_example = []

    for ex in examples:
        hce_res = estimate(ex.question)
        pred = hce_res.tier

        gold_hardness = ex.hardness
        if not gold_hardness:
            gold_hardness = spider_hardness_from_sql(ex.gold_sql)
        gold = gold_to_tier(gold_hardness)

        pred_tiers.append(pred.value)
        gold_tiers.append(gold.value)

        per_example.append({
            "question": ex.question,
            "db_id": ex.db_id,
            "gold_hardness": gold_hardness,
            "gold_tier": gold.value,
            "pred_tier": pred.value,
            "score": hce_res.score,
            "fired": hce_res.fired,
        })

    labels = ["SIMPLE", "MEDIUM", "COMPLEX"]

    # Confusion matrix: rows = gold, cols = predicted
    cm = {g: {p: 0 for p in labels} for g in labels}
    for p, g in zip(pred_tiers, gold_tiers):
        cm[g][p] += 1

    # Per-tier metrics
    per_tier = {}
    for tier in labels:
        gold_count = sum(cm[tier].values())
        correct = cm[tier][tier]
        per_tier[tier] = {
            "gold_count": gold_count,
            "correct": correct,
            "recall": correct / gold_count if gold_count else 0.0,
        }

    accuracy = sum(1 for p, g in zip(pred_tiers, gold_tiers) if p == g) / len(pred_tiers)
    kappa = cohens_kappa(pred_tiers, gold_tiers, labels)

    summary = {
        "n_examples": len(examples),
        "agreement_accuracy": accuracy,
        "cohens_kappa": kappa,
        "confusion_matrix_gold_x_pred": cm,
        "per_tier": per_tier,
        "pred_distribution": dict(Counter(pred_tiers)),
        "gold_distribution": dict(Counter(gold_tiers)),
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {args.out}")

    with open(args.per_example_jsonl, "w") as f:
        for r in per_example:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote per-example to {args.per_example_jsonl}")

    # Pretty print
    print()
    print(f"N examples         : {summary['n_examples']}")
    print(f"Agreement accuracy : {accuracy:.3f}")
    print(f"Cohen's kappa      : {kappa:.3f}")
    print()
    print("Confusion matrix (rows=gold, cols=pred):")
    header = f"{'':<10}" + "".join(f"{l:<10}" for l in labels)
    print(header)
    for g in labels:
        row = f"{g:<10}" + "".join(f"{cm[g][p]:<10}" for p in labels)
        print(row)
    print()
    print("Per-tier (gold) recall:")
    for tier, m in per_tier.items():
        print(f"  {tier:<8}: {m['correct']}/{m['gold_count']} = {m['recall']:.3f}")
    print()
    print(f"Predicted distribution: {summary['pred_distribution']}")
    print(f"Gold distribution    : {summary['gold_distribution']}")


if __name__ == "__main__":
    main()
