"""
End-to-end smoke test: HCE + CGSP + (mocked CCSG) + executor.
Verifies that all the code paths work before you spend GPU time.

Run: python scripts/smoke_test.py
Expected: all checks pass in under 10 seconds.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cas_sql.ccsg import extract_sql
from cas_sql.cgsp import CGSP, load_schema_from_sqlite
from cas_sql.hce import Tier, estimate
from eval.executor import execution_match


def setup_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    path = Path(f.name)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, building TEXT);
        CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER,
            FOREIGN KEY(dept_id) REFERENCES departments(id));
        CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, credits INTEGER);
        CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade REAL);
        INSERT INTO departments VALUES (10,'CS','HallA'),(20,'Math','HallB');
        INSERT INTO students VALUES (1,'Alice',10),(2,'Bob',20),(3,'Carol',10);
        INSERT INTO courses VALUES (100,'Algebra',3),(200,'Algorithms',4);
        INSERT INTO enrollments VALUES (1,100,90.0),(1,200,85.0),(2,100,78.0),(3,200,92.0);
    """)
    conn.commit()
    conn.close()
    return path


def main():
    print("=== CAS-SQL smoke test ===\n")
    failures = 0

    # Test 1: HCE
    print("[1] HCE tier assignment")
    r = estimate("list all students")
    assert r.tier == Tier.SIMPLE, f"expected SIMPLE, got {r.tier}"
    r = estimate("show top 5 departments by average salary")
    assert r.tier == Tier.MEDIUM, f"expected MEDIUM, got {r.tier}"
    r = estimate("find the top 3 departments by average salary excluding interns hired before 2020")
    assert r.tier == Tier.COMPLEX, f"expected COMPLEX, got {r.tier}"
    print("    OK")

    # Test 2: Schema loading + CGSP
    print("[2] CGSP schema pruning")
    db = setup_db()
    try:
        schema = load_schema_from_sqlite(db, "test_db")
        assert len(schema.tables) == 4
        assert len(schema.foreign_keys) >= 1
        cgsp = CGSP()
        s_simple, sel_s = cgsp.prune("list all students", schema, Tier.SIMPLE)
        assert len(sel_s) == 2, f"SIMPLE should give 2 tables, got {len(sel_s)}"
        s_medium, sel_m = cgsp.prune("show top departments by student count", schema, Tier.MEDIUM)
        assert len(sel_m) >= 2
        s_complex, sel_c = cgsp.prune("complex query", schema, Tier.COMPLEX)
        assert len(sel_c) == 4, f"COMPLEX should give all 4 tables, got {len(sel_c)}"
        assert "sample rows" in s_complex.lower(), "COMPLEX should include sample rows"
        print(f"    OK (SIMPLE={sel_s}, MEDIUM={sel_m}, COMPLEX={sel_c})")
    except Exception as e:
        print(f"    FAIL: {e}")
        failures += 1

    # Test 3: SQL extraction
    print("[3] SQL extraction post-processing")
    raw = "Reasoning blah.\nSQL: SELECT name FROM students WHERE id = 1;"
    assert extract_sql(raw) == "SELECT name FROM students WHERE id = 1"
    raw = "```sql\nSELECT 1\n```"
    assert extract_sql(raw) == "SELECT 1"
    print("    OK")

    # Test 4: Executor
    print("[4] Execution accuracy comparator")
    match, err = execution_match(
        db,
        "SELECT name FROM students WHERE dept_id = 10",
        "SELECT name FROM students WHERE dept_id = 10",
    )
    assert match and err is None, f"expected match, got match={match} err={err}"
    match, err = execution_match(
        db,
        "SELECT name FROM students WHERE dept_id = 20",
        "SELECT name FROM students WHERE dept_id = 10",
    )
    assert not match, "expected mismatch (different results)"
    match, err = execution_match(db, "SELECT bogus FROM students", "SELECT name FROM students")
    assert not match and err is not None, "expected pred SQL error"
    print("    OK")

    db.unlink()

    if failures == 0:
        print("\n✅ All smoke tests passed.")
        return 0
    print(f"\n❌ {failures} test(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
