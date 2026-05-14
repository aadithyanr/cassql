#!/usr/bin/env bash
# setup.sh — one-shot environment setup for the college PC.
# Run from the repo root.
set -euo pipefail

echo "=== CAS-SQL setup ==="

# 1. Python venv
if [ ! -d ".venv" ]; then
  echo "[1/4] Creating Python venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo "    venv ready"

# 2. Pre-download the sentence embedding model
echo "[2/4] Caching sentence-transformer model..."
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('    embed model cached')
"

# 3. Check Ollama
echo "[3/4] Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: Ollama not found. Install from https://ollama.com"
  exit 1
fi
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "ERROR: Ollama daemon not running. Start with: ollama serve"
  exit 1
fi
echo "    Ollama OK"

# 4. Pull models
echo "[4/4] Pulling Ollama models (this may take a while)..."
echo "    Pulling mistral:7b-instruct-v0.3-q4_K_M ..."
ollama pull mistral:7b-instruct-v0.3-q4_K_M
echo "    Pulling qwen2.5-coder:14b ..."
ollama pull qwen2.5-coder:14b
echo "    Pulling llama3.1:8b-instruct-q4_K_M ..."
ollama pull llama3.1:8b-instruct-q4_K_M
echo "    Pulling phi4:14b ..."
ollama pull phi4:14b || echo "    (phi4 pull failed — OK to skip)"

echo
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Download Spider dev:  bash scripts/download_data.sh"
echo "  2. Run HCE diagnostic:   python scripts/run_hce_diagnostic.py --spider-root data/spider"
echo "  3. Run eval:             python scripts/run_eval.py --dataset spider --data-root data/spider --model mistral:7b-instruct-v0.3-q4_K_M --out-dir results/mistral_spider"
