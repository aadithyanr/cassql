"""
Execution Accuracy (EX) evaluator.

Runs predicted and gold SQL against the SQLite database and compares
result sets. Handles:
  - timeouts (kills query after N seconds)
  - errors in predicted SQL (counted as wrong, not crash)
  - ordering: ORDER BY in gold means order matters; else compare as multisets
  - case-insensitive string comparison for VARCHAR equality
"""

from __future__ import annotations

import re
import signal
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class QueryTimeout(Exception):
    pass


@contextmanager
def _timeout(seconds: int):
    """POSIX-only timeout via SIGALRM. On Windows, falls through (no timeout)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise QueryTimeout(f"Query exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _execute(db_path: Path, sql: str, timeout_s: int = 30) -> tuple[list, str | None]:
    """
    Execute SQL against the SQLite DB. Returns (rows, error_message).
    rows is empty list on error.
    """
    if not sql or not sql.strip():
        return [], "empty SQL"
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda x: x.decode("utf-8", errors="replace")
        cur = conn.cursor()
        with _timeout(timeout_s):
            cur.execute(sql)
            rows = cur.fetchall()
        conn.close()
        return rows, None
    except QueryTimeout as e:
        return [], str(e)
    except sqlite3.Error as e:
        return [], f"SQLiteError: {e}"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _normalize_row(row: tuple) -> tuple:
    """Normalize values for comparison: lowercase strings, round floats."""
    out = []
    for v in row:
        if isinstance(v, str):
            out.append(v.strip().lower())
        elif isinstance(v, float):
            out.append(round(v, 4))
        else:
            out.append(v)
    return tuple(out)


def _gold_has_order_by(gold_sql: str) -> bool:
    """Heuristic: ORDER BY in the gold => order matters in comparison."""
    return bool(re.search(r"(?is)\border\s+by\b", gold_sql))


def compare_results(
    pred_rows: list, gold_rows: list, order_matters: bool
) -> bool:
    """Compare two result sets. Returns True if they match."""
    if pred_rows is None or gold_rows is None:
        return False
    if len(pred_rows) != len(gold_rows):
        return False

    p_norm = [_normalize_row(r) for r in pred_rows]
    g_norm = [_normalize_row(r) for r in gold_rows]

    if order_matters:
        return p_norm == g_norm
    return sorted(p_norm) == sorted(g_norm)


def execution_match(
    db_path: Path, pred_sql: str, gold_sql: str, timeout_s: int = 30
) -> tuple[bool, str | None]:
    """
    Execute both queries; return (match, error_string).
    error_string is non-None when the predicted query errored out.
    """
    gold_rows, gold_err = _execute(db_path, gold_sql, timeout_s)
    if gold_err is not None:
        # Gold itself errored — skip (rare; data issue)
        return False, f"GOLD_ERR: {gold_err}"

    pred_rows, pred_err = _execute(db_path, pred_sql, timeout_s)
    if pred_err is not None:
        return False, pred_err

    order_matters = _gold_has_order_by(gold_sql)
    return compare_results(pred_rows, gold_rows, order_matters), None


if __name__ == "__main__":
    # Smoke test
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = Path(f.name)

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE t (id INTEGER, name TEXT, score REAL);
        INSERT INTO t VALUES (1,'Alice',90.0),(2,'Bob',85.0),(3,'Charlie',95.0);
        """
    )
    conn.commit()
    conn.close()

    cases = [
        ("SELECT name FROM t WHERE score > 80",
         "SELECT name FROM t WHERE score > 80", True),
        ("SELECT name FROM t ORDER BY score DESC",
         "SELECT name FROM t ORDER BY score DESC", True),
        # different order, gold has ORDER BY -> mismatch
        ("SELECT name FROM t",
         "SELECT name FROM t ORDER BY score DESC", False),
        # same set, no ORDER BY in gold -> match
        ("SELECT name FROM t ORDER BY name",
         "SELECT name FROM t", True),
        # bad SQL
        ("SELECT bogus FROM t", "SELECT name FROM t", False),
    ]
    for pred, gold, expected in cases:
        match, err = execution_match(path, pred, gold)
        ok = "✓" if match == expected else "✗"
        print(f"{ok} expected={expected} got={match} err={err}  | pred={pred!r}")
    path.unlink()
