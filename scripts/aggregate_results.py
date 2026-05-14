"""
Aggregate summary.json files from multiple model runs into a comparison table
matching Table IV / Table V in the paper.

Usage:
  python scripts/aggregate_results.py results/mistral_spider results/qwen_spider results/llama_spider --out paper_tables.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Each should contain summary.json")
    ap.add_argument("--out", type=Path, default=Path("results/paper_tables.txt"))
    args = ap.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        spath = run_dir / "summary.json"
        if not spath.exists():
            print(f"WARN: no summary.json in {run_dir}")
            continue
        with open(spath) as f:
            s = json.load(f)
        rows.append({
            "name": run_dir.name,
            "n": s["n_examples"],
            "ex": s["execution_accuracy"],
            "avg_in": s["tokens_per_query"]["avg_in"],
            "avg_out": s["tokens_per_query"]["avg_out"],
            "avg_tot": s["tokens_per_query"]["avg_total"],
            "avg_lat": s["avg_latency_s"],
            "by_tier": s.get("by_tier", {}),
            "by_hardness": s.get("by_gold_hardness", {}),
        })

    if not rows:
        print("No results found.")
        return

    out_lines = []

    # Table IV-style: EX
    out_lines.append("=" * 80)
    out_lines.append("Table IV (filled): Execution Accuracy")
    out_lines.append("=" * 80)
    out_lines.append(f"{'Run':<40} {'N':>5} {'EX':>7}")
    out_lines.append("-" * 80)
    for r in rows:
        out_lines.append(f"{r['name']:<40} {r['n']:>5} {r['ex']*100:>6.1f}%")

    # Table V-style: Token cost
    out_lines.append("")
    out_lines.append("=" * 80)
    out_lines.append("Table V (filled): Measured token cost per query")
    out_lines.append("=" * 80)
    out_lines.append(f"{'Run':<40} {'in':>7} {'out':>7} {'total':>8} {'lat(s)':>7}")
    out_lines.append("-" * 80)
    for r in rows:
        out_lines.append(
            f"{r['name']:<40} {r['avg_in']:>7.0f} {r['avg_out']:>7.0f} "
            f"{r['avg_tot']:>8.0f} {r['avg_lat']:>7.2f}"
        )

    # Per-tier breakdown
    out_lines.append("")
    out_lines.append("=" * 80)
    out_lines.append("Per-tier accuracy (predicted tier)")
    out_lines.append("=" * 80)
    out_lines.append(f"{'Run':<40} {'SIMPLE':>14} {'MEDIUM':>14} {'COMPLEX':>14}")
    out_lines.append("-" * 80)
    for r in rows:
        cells = []
        for t in ["SIMPLE", "MEDIUM", "COMPLEX"]:
            tinfo = r["by_tier"].get(t, {})
            if tinfo:
                cells.append(f"{tinfo['correct']}/{tinfo['n']} ({tinfo['accuracy']*100:.1f}%)")
            else:
                cells.append("-")
        out_lines.append(f"{r['name']:<40} " + " ".join(f"{c:>14}" for c in cells))

    # Per-hardness breakdown
    out_lines.append("")
    out_lines.append("=" * 80)
    out_lines.append("Accuracy by Spider gold hardness")
    out_lines.append("=" * 80)
    out_lines.append(
        f"{'Run':<40} {'easy':>14} {'medium':>14} {'hard':>14} {'extra':>14}"
    )
    out_lines.append("-" * 80)
    for r in rows:
        cells = []
        for h in ["easy", "medium", "hard", "extra"]:
            hinfo = r["by_hardness"].get(h, {})
            if hinfo and hinfo["n"]:
                cells.append(f"{hinfo['correct']}/{hinfo['n']} ({hinfo['accuracy']*100:.1f}%)")
            else:
                cells.append("-")
        out_lines.append(f"{r['name']:<40} " + " ".join(f"{c:>14}" for c in cells))

    text = "\n".join(out_lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
