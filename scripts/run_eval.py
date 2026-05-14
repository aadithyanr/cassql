"""
Main evaluation runner for CAS-SQL.

Runs the full pipeline (HCE -> CGSP -> CCSG) on Spider dev or BIRD dev,
executes predicted SQL against the DB, compares to gold, and writes
per-example results + an aggregate summary.

Features:
  - Resume from existing predictions file (skips completed indices)
  - Per-tier accuracy breakdown
  - Token usage logging
  - Schema-pruning recall (did the gold tables survive pruning?)

Usage:
  python scripts/run_eval.py \
      --dataset spider \
      --data-root /data/spider \
      --model qwen2.5-coder:14b \
      --out-dir results/qwen14b_spider \
      --limit 1034
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# repo-root path setup
sys.path.insert(0, str(Path(__file__).parent.parent))

from cas_sql.ccsg import CCSG
from cas_sql.cgsp import CGSP, load_schema_from_sqlite
from cas_sql.pipeline import CASSQLPipeline
from eval.datasets import (
    Example,
    load_bird_dev,
    load_spider_dev,
    spider_hardness_from_sql,
)
from eval.executor import execution_match


def gold_tables_from_sql(sql: str) -> set[str]:
    """Heuristic extraction of table names referenced in gold SQL."""
    # FROM <table> [AS alias], JOIN <table>, UPDATE <table>, INTO <table>
    # Very rough; good enough for schema-recall analysis.
    s = sql
    pat = re.compile(
        r"(?i)\b(?:from|join|update|into)\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    return {m.group(1).lower() for m in pat.finditer(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["spider", "bird"], required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. qwen2.5-coder:14b")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="Max examples to run")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--exec-timeout", type=int, default=30)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--resume", action="store_true",
                    help="Skip examples already in predictions.jsonl")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.dataset == "spider":
        examples = load_spider_dev(args.data_root)
    else:
        examples = load_bird_dev(args.data_root)

    if args.start:
        examples = examples[args.start :]
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} {args.dataset} examples")

    # Init pipeline
    cgsp = CGSP()
    ccsg = CCSG(
        model=args.model,
        host=args.host,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        timeout=args.timeout,
    )
    if not ccsg.ping():
        print(f"WARNING: model '{args.model}' not found on Ollama at {args.host}")
        print(f"  Available models can be listed with: curl {args.host}/api/tags")
        print(f"  Pull with: ollama pull {args.model}")
        sys.exit(2)

    pipeline = CASSQLPipeline(cgsp=cgsp, ccsg=ccsg)

    # Resume support
    pred_path = args.out_dir / "predictions.jsonl"
    completed_keys: set[str] = set()
    if args.resume and pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_keys.add(f"{rec['db_id']}|||{rec['question']}")
                except Exception:
                    continue
        print(f"Resume: {len(completed_keys)} examples already done")

    # Cache schemas across examples (same db_id reused many times)
    schema_cache: dict[str, object] = {}

    f_pred = open(pred_path, "a")
    t_global = time.time()

    for i, ex in enumerate(examples):
        key = f"{ex.db_id}|||{ex.question}"
        if key in completed_keys:
            continue
        if not ex.db_path.exists():
            print(f"[{i}] SKIP missing DB: {ex.db_path}")
            continue

        # Load schema (cached)
        if ex.db_id not in schema_cache:
            try:
                schema_cache[ex.db_id] = load_schema_from_sqlite(ex.db_path, ex.db_id)
            except Exception as e:
                print(f"[{i}] SKIP schema-load error for {ex.db_id}: {e}")
                continue
        schema = schema_cache[ex.db_id]

        t0 = time.time()
        try:
            result = pipeline.run(ex.question, schema)
        except Exception as e:
            print(f"[{i}] PIPELINE ERROR: {e}")
            continue

        # Evaluate execution match
        match, exec_err = execution_match(
            ex.db_path, result.sql, ex.gold_sql, timeout_s=args.exec_timeout
        )

        # Schema-pruning recall
        gold_tables = gold_tables_from_sql(ex.gold_sql)
        selected_lc = {t.lower() for t in result.selected_tables}
        recall = (
            len(gold_tables & selected_lc) / len(gold_tables) if gold_tables else 1.0
        )

        # Gold hardness (compute if absent)
        gold_hardness = ex.hardness or spider_hardness_from_sql(ex.gold_sql)

        rec = result.to_dict()
        rec.update({
            "gold_sql": ex.gold_sql,
            "gold_hardness": gold_hardness,
            "exec_match": bool(match),
            "exec_err": exec_err,
            "gold_tables": sorted(gold_tables),
            "schema_recall": recall,
            "wall_s": time.time() - t0,
        })
        f_pred.write(json.dumps(rec) + "\n")
        f_pred.flush()

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_global
            eta = elapsed / (i + 1) * (len(examples) - i - 1)
            print(
                f"[{i+1}/{len(examples)}] tier={result.tier} "
                f"match={match} lat={result.latency_s:.1f}s "
                f"in_tok={result.prompt_tokens} out_tok={result.output_tokens} "
                f"ETA={eta/60:.1f}min"
            )

    f_pred.close()

    # Aggregate
    print("\nAggregating...")
    summarize(pred_path, args.out_dir / "summary.json")


def summarize(pred_path: Path, out_path: Path):
    """Compute aggregate metrics from predictions.jsonl."""
    by_tier_correct = Counter()
    by_tier_total = Counter()
    by_hardness_correct = Counter()
    by_hardness_total = Counter()
    tokens_by_tier: dict[str, list[int]] = defaultdict(list)
    recall_by_tier: dict[str, list[float]] = defaultdict(list)
    n = 0
    total_correct = 0
    total_in_tokens = 0
    total_out_tokens = 0
    total_latency = 0.0
    errors = 0

    with open(pred_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            t = rec.get("tier", "UNKNOWN")
            h = rec.get("gold_hardness", "unknown")
            ok = bool(rec.get("exec_match"))

            by_tier_total[t] += 1
            by_hardness_total[h] += 1
            if ok:
                by_tier_correct[t] += 1
                by_hardness_correct[h] += 1
                total_correct += 1
            if rec.get("error") or rec.get("exec_err"):
                errors += 1

            in_t = int(rec.get("prompt_tokens", 0))
            out_t = int(rec.get("output_tokens", 0))
            total_in_tokens += in_t
            total_out_tokens += out_t
            tokens_by_tier[t].append(in_t + out_t)
            recall_by_tier[t].append(float(rec.get("schema_recall", 0.0)))
            total_latency += float(rec.get("latency_s", 0.0))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "n_examples": n,
        "execution_accuracy": total_correct / n if n else 0.0,
        "n_errors": errors,
        "tokens_per_query": {
            "avg_in": total_in_tokens / n if n else 0.0,
            "avg_out": total_out_tokens / n if n else 0.0,
            "avg_total": (total_in_tokens + total_out_tokens) / n if n else 0.0,
        },
        "avg_latency_s": total_latency / n if n else 0.0,
        "by_tier": {
            t: {
                "n": by_tier_total[t],
                "correct": by_tier_correct[t],
                "accuracy": by_tier_correct[t] / by_tier_total[t]
                if by_tier_total[t]
                else 0.0,
                "avg_tokens": mean(tokens_by_tier[t]),
                "avg_schema_recall": mean(recall_by_tier[t]),
            }
            for t in by_tier_total
        },
        "by_gold_hardness": {
            h: {
                "n": by_hardness_total[h],
                "correct": by_hardness_correct[h],
                "accuracy": by_hardness_correct[h] / by_hardness_total[h]
                if by_hardness_total[h]
                else 0.0,
            }
            for h in by_hardness_total
        },
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
