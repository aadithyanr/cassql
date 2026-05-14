"""
Spider and BIRD dataset loaders.

Spider dev: dev.json + database/<db_id>/<db_id>.sqlite
BIRD dev: dev.json + dev_databases/<db_id>/<db_id>.sqlite (and optional dev_tables.json)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    question: str
    gold_sql: str
    db_id: str
    db_path: Path
    hardness: str | None = None  # Spider provides this; BIRD has "difficulty"


def load_spider_dev(spider_root: Path) -> list[Example]:
    """
    spider_root layout:
      spider_root/
        dev.json
        database/<db_id>/<db_id>.sqlite
    """
    dev_path = spider_root / "dev.json"
    db_root = spider_root / "database"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing {dev_path}")
    if not db_root.exists():
        raise FileNotFoundError(f"Missing {db_root}")

    with open(dev_path) as f:
        data = json.load(f)

    examples = []
    for row in data:
        db_id = row["db_id"]
        db_path = db_root / db_id / f"{db_id}.sqlite"
        examples.append(
            Example(
                question=row["question"],
                gold_sql=row["query"],
                db_id=db_id,
                db_path=db_path,
                hardness=row.get("hardness"),  # Spider 1.0 has this
            )
        )
    return examples


def load_bird_dev(bird_root: Path) -> list[Example]:
    """
    bird_root layout (the dev split):
      bird_root/
        dev.json
        dev_databases/<db_id>/<db_id>.sqlite
    """
    dev_path = bird_root / "dev.json"
    db_root = bird_root / "dev_databases"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing {dev_path}")
    if not db_root.exists():
        raise FileNotFoundError(f"Missing {db_root}")

    with open(dev_path) as f:
        data = json.load(f)

    examples = []
    for row in data:
        db_id = row["db_id"]
        db_path = db_root / db_id / f"{db_id}.sqlite"
        # BIRD uses "SQL" for gold and "question" for NL.
        gold = row.get("SQL") or row.get("query")
        # Some BIRD examples carry "evidence" — domain hint to append to question.
        question = row["question"]
        if row.get("evidence"):
            question = f"{question} (Hint: {row['evidence']})"
        examples.append(
            Example(
                question=question,
                gold_sql=gold,
                db_id=db_id,
                db_path=db_path,
                hardness=row.get("difficulty"),
            )
        )
    return examples


# Spider provides gold "hardness" labels that fall into 4 buckets:
# easy, medium, hard, extra. Some Spider dev releases don't include this
# in dev.json; in that case we can fall back to computing it from the SQL
# using the official component-counting heuristic. Implement a simple version.

_AGG_FUNCS = {"max", "min", "count", "sum", "avg"}
_KEYWORDS = ["where", "group by", "order by", "limit", "join", "or ", "like", "having"]


def spider_hardness_from_sql(sql: str) -> str:
    """
    Simplified version of Spider's official hardness scorer.
    Counts components and SQL keywords to bucket into easy/medium/hard/extra.
    """
    s = sql.lower()
    # count keywords
    kw_count = sum(s.count(kw) for kw in _KEYWORDS)
    # count aggregations
    agg_count = sum(s.count(f"{a}(") for a in _AGG_FUNCS)
    # count nested SELECTs
    nested = max(0, s.count("select") - 1)

    score = kw_count + agg_count + 2 * nested

    if nested >= 1 and (kw_count + agg_count) >= 4:
        return "extra"
    if score >= 5:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"


if __name__ == "__main__":
    # Hardness scorer smoke test
    cases = [
        ("SELECT name FROM students", "easy"),
        ("SELECT name FROM students WHERE age > 18", "easy"),
        ("SELECT dept, COUNT(*) FROM students GROUP BY dept ORDER BY COUNT(*) DESC", "medium"),
        (
            "SELECT name FROM students WHERE id IN (SELECT student_id FROM "
            "enrollments WHERE grade > 90) AND dept_id IN (SELECT id FROM "
            "departments WHERE name LIKE '%CS%') ORDER BY name",
            "extra",
        ),
    ]
    for sql, expected in cases:
        got = spider_hardness_from_sql(sql)
        ok = "✓" if got == expected else f"≠ expected {expected}"
        print(f"{ok} {got:<10} {sql[:80]}")
