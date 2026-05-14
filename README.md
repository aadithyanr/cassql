# CAS-SQL: Empirical evaluation harness

Training-free implementation of the CAS-SQL pipeline (HCE → CGSP → CCSG)
for Spider / BIRD dev. Designed to be runnable end-to-end on one GPU.

## Hardware expectations

- GPU with ≥ 16GB VRAM (24GB recommended for 14B models)
- 32GB system RAM is the tight spec — close other apps
- ~10GB disk for Spider, ~25GB for BIRD, ~30GB for Ollama model cache
- Ollama installed and running (`ollama serve`)

## Today's runbook (in order, do not skip)

```bash
# 0. Setup
bash setup.sh
bash scripts/download_data.sh

# 1. Smoke test (no LLM, < 10s) — confirms plumbing works
python scripts/smoke_test.py

# 2. HCE diagnostic (no LLM, < 60s) — fills a gap in your paper
python scripts/run_hce_diagnostic.py --spider-root data/spider

# --- now the GPU runs ---

# 3. Spider eval, Mistral 7B (your DTS-SQL-comparable model first)
python scripts/run_eval.py \
    --dataset spider --data-root data/spider \
    --model mistral:7b-instruct-v0.3-q4_K_M \
    --out-dir results/mistral_spider \
    --resume

# 4. Spider eval, Qwen2.5-Coder 14B
python scripts/run_eval.py \
    --dataset spider --data-root data/spider \
    --model qwen2.5-coder:14b \
    --out-dir results/qwen_spider \
    --resume

# 5. Spider eval, Llama-3.1 8B
python scripts/run_eval.py \
    --dataset spider --data-root data/spider \
    --model llama3.1:8b-instruct-q4_K_M \
    --out-dir results/llama_spider \
    --resume

# 6. Spider eval, Phi-4 14B  (skip if running out of time)
python scripts/run_eval.py \
    --dataset spider --data-root data/spider \
    --model phi4:14b \
    --out-dir results/phi4_spider \
    --resume

# 7. Aggregate into paper tables
python scripts/aggregate_results.py \
    results/mistral_spider results/qwen_spider \
    results/llama_spider results/phi4_spider \
    --out results/paper_tables.txt

# 8. (Stretch) BIRD on best-performing model only
python scripts/run_eval.py \
    --dataset bird --data-root data/bird \
    --model qwen2.5-coder:14b \
    --out-dir results/qwen_bird \
    --resume
```

## Expected timings (Ada 4500, 24GB)

| Step | Wall time |
|---|---|
| setup.sh (mostly Ollama pulls) | 20–40 min |
| download_data.sh (Spider + BIRD) | 15–30 min |
| smoke_test.py | < 10s |
| run_hce_diagnostic.py | < 60s |
| run_eval.py Spider 7B | 60–90 min |
| run_eval.py Spider 14B | 90–120 min |
| run_eval.py BIRD 14B | 180–240 min |

If a run dies mid-way, re-run the same command with `--resume`; it picks up
from where the predictions.jsonl left off.

## Outputs per run

```
results/<run-name>/
  predictions.jsonl   # one row per example: tier, sql, gold, match, tokens...
  summary.json        # aggregate metrics
```

Aggregated comparison goes to `results/paper_tables.txt`.

## Filling the paper

After step 7, you have everything for:

- **Table IV (Execution Accuracy)**: from summary.json `execution_accuracy`
- **Table V (Token cost — measured, not projected)**: from `tokens_per_query`
- **New per-tier breakdown table**: from `by_tier` in each summary
- **HCE diagnostic table**: from `results/hce_diagnostic.json`
- **Schema-pruning recall (new figure)**: from `by_tier[*].avg_schema_recall`

## Troubleshooting

**Ollama OOM mid-run**: drop `--num-ctx` from 8192 to 4096 in run_eval.py args.
Spider schemas mostly fit in 4k; only a few extreme cases need 8k.

**Embedding model download fails**: HuggingFace mirror issue. Set
`HF_ENDPOINT=https://hf-mirror.com` before running setup.sh.

**Phi-4 produces garbage**: known issue with some Ollama Phi-4 ports — stop
tokens behave oddly. Drop it and report 3 models, not 4.

**BIRD download fails**: the OSS-Beijing mirror is sometimes blocked outside
mainland China. Manual download from https://bird-bench.github.io/ and unzip
to `data/bird/`.

## What's in this repo

```
cas_sql/
  hce.py        # Algorithm 1 — heuristic complexity estimator
  cgsp.py       # Algorithm 2 — complexity-gated schema pruner
  ccsg.py       # Algorithm 3 — Ollama-backed generator + SQL extraction
  prompts.py    # Tier-specific prompt templates
  pipeline.py   # Orchestrator: HCE → CGSP → CCSG
eval/
  datasets.py   # Spider/BIRD loaders + hardness scorer
  executor.py   # Execution Accuracy evaluator (SQLite execution + comparison)
scripts/
  smoke_test.py             # End-to-end plumbing check (no LLM)
  run_hce_diagnostic.py     # HCE vs Spider gold hardness (paper diagnostic)
  run_eval.py               # Main eval runner
  aggregate_results.py      # Multi-model comparison tables
  download_data.sh          # Fetch Spider + BIRD
setup.sh                    # One-shot env setup
requirements.txt
```

## What this code does NOT do

- No fine-tuning. Paper is training-free; this code respects that.
- No multi-sample / self-consistency. τ=0, one LLM call per query.
- No retrieval-augmented exemplar selection. Few-shot examples are fixed in prompts.
- No SQL repair loop. If the model outputs syntactically broken SQL, it counts as wrong.
