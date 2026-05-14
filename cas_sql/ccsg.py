"""
Complexity-Conditioned SQL Generator (CCSG).

Implements Algorithm 3 from the CAS-SQL paper using a frozen LLM accessed
via the local Ollama HTTP API. Tier-conditioned prompt selection and
robust post-processing of the model output.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from cas_sql.hce import Tier
from cas_sql.prompts import PROMPTS


@dataclass
class GenerationResult:
    sql: str
    raw_output: str
    prompt_tokens: int  # estimated via tokenizer when available, else len(prompt)//4
    output_tokens: int
    latency_s: float
    model: str
    error: str | None = None


def _approx_token_count(s: str) -> int:
    """Cheap token estimator: 1 token ≈ 4 chars. Good enough for cost reporting."""
    return max(1, len(s) // 4)


_CODE_FENCE = re.compile(r"```(?:sql|sqlite)?\s*([\s\S]*?)```", re.IGNORECASE)
_SQL_LABEL = re.compile(r"(?i)\bSQL\s*[:\-]?\s*", re.MULTILINE)


def extract_sql(raw: str) -> str:
    """
    Extract a single SQL statement from a model output.

    Strategy:
      1. If a fenced ```sql block exists, take the first one.
      2. Else if 'SQL:' label exists, take everything after the last occurrence.
      3. Else assume the whole output is SQL.
    Then trim to the first statement (up to and including the first ';' if any),
    strip whitespace, and collapse internal newlines.
    """
    if not raw:
        return ""

    # 1. fenced block
    m = _CODE_FENCE.search(raw)
    if m:
        candidate = m.group(1)
    else:
        # 2. SQL: label — take after the last occurrence
        matches = list(_SQL_LABEL.finditer(raw))
        if matches:
            candidate = raw[matches[-1].end() :]
        else:
            candidate = raw

    candidate = candidate.strip()

    # Take first statement only
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0]

    # Collapse internal whitespace
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # Sanity: must start with a SQL keyword
    if not re.match(r"(?i)^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", candidate):
        # Try to find a SELECT/WITH inside
        m = re.search(r"(?i)\b(WITH|SELECT)\b", candidate)
        if m:
            candidate = candidate[m.start() :]

    return candidate.strip()


class CCSG:
    """
    Complexity-Conditioned SQL Generator backed by Ollama.

    Ollama HTTP API: POST http://localhost:11434/api/generate
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        num_predict: int = 512,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout

    def generate(self, question: str, schema: str, tier: Tier) -> GenerationResult:
        prompt = PROMPTS[tier.value].format(schema=schema, question=question)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return GenerationResult(
                sql="",
                raw_output="",
                prompt_tokens=_approx_token_count(prompt),
                output_tokens=0,
                latency_s=time.time() - t0,
                model=self.model,
                error=f"{type(e).__name__}: {e}",
            )

        latency = time.time() - t0
        raw = data.get("response", "")
        sql = extract_sql(raw)

        # Ollama returns token counts; fall back to estimate if missing
        prompt_tokens = data.get("prompt_eval_count") or _approx_token_count(prompt)
        output_tokens = data.get("eval_count") or _approx_token_count(raw)

        return GenerationResult(
            sql=sql,
            raw_output=raw,
            prompt_tokens=int(prompt_tokens),
            output_tokens=int(output_tokens),
            latency_s=latency,
            model=self.model,
        )

    def ping(self) -> bool:
        """Check that Ollama is reachable and the model is pulled."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = [m["name"] for m in tags]
            return any(self.model in n or n in self.model for n in names)
        except Exception:
            return False


if __name__ == "__main__":
    # Test SQL extraction logic without needing Ollama
    cases = [
        ("```sql\nSELECT * FROM students;\n```", "SELECT * FROM students"),
        ("Reasoning: blah.\nSQL: SELECT name FROM students WHERE age > 18;",
         "SELECT name FROM students WHERE age > 18"),
        ("SELECT 1;", "SELECT 1"),
        ("Sure, here you go:\n```\nSELECT count(*) FROM t;\n```\nHope this helps!",
         "SELECT count(*) FROM t"),
        ("Decomposition:\n1. step\nSQL: WITH a AS (SELECT * FROM t) SELECT * FROM a LIMIT 3;",
         "WITH a AS (SELECT * FROM t) SELECT * FROM a LIMIT 3"),
    ]
    print(f"{'Input':<70} {'Expected':<50} OK")
    print("-" * 130)
    for inp, expected in cases:
        got = extract_sql(inp)
        ok = "✓" if got == expected else f"✗ got: {got!r}"
        print(f"{inp[:69]!r:<70} {expected[:49]!r:<50} {ok}")
