"""
Complexity-Gated Schema Pruner (CGSP).

Implements Algorithm 2 from the CAS-SQL paper. Uses sentence-transformers
(all-MiniLM-L6-v2) to embed the question and each table, ranks by cosine
similarity, and retains a tier-appropriate subset.

Schema retention policy:
  SIMPLE  : top-2 tables, no sample rows
  MEDIUM  : top-4 tables expanded with FK neighbors, no sample rows
  COMPLEX : full schema with 3 sample rows per table
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from cas_sql.hce import Tier


@dataclass
class Column:
    name: str
    dtype: str
    is_pk: bool = False


@dataclass
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)

    def textual(self) -> str:
        """Flat textual rep for embedding: table name + column names."""
        cols = " ".join(c.name for c in self.columns)
        return f"{self.name} {cols}"


@dataclass
class Schema:
    db_id: str
    tables: dict[str, Table]
    foreign_keys: list[ForeignKey]
    db_path: Path  # path to the SQLite file (for sample rows)

    def all_table_names(self) -> list[str]:
        return list(self.tables.keys())


def load_schema_from_sqlite(db_path: Path, db_id: str) -> Schema:
    """Load a schema by introspecting the SQLite file directly."""
    conn = sqlite3.connect(str(db_path))
    conn.text_factory = lambda x: x.decode("utf-8", errors="replace")
    cur = conn.cursor()

    tables: dict[str, Table] = {}
    fks: list[ForeignKey] = []

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    for (tname,) in cur.fetchall():
        cols: list[Column] = []
        cur.execute(f'PRAGMA table_info("{tname}")')
        for cid, cname, ctype, notnull, dflt, pk in cur.fetchall():
            cols.append(Column(name=cname, dtype=(ctype or "TEXT"), is_pk=bool(pk)))
        tables[tname] = Table(name=tname, columns=cols)

        cur.execute(f'PRAGMA foreign_key_list("{tname}")')
        for row in cur.fetchall():
            # row schema: (id, seq, table, from, to, on_update, on_delete, match)
            _, _, to_table, from_col, to_col, *_ = row
            fks.append(
                ForeignKey(
                    from_table=tname,
                    from_column=from_col,
                    to_table=to_table,
                    to_column=to_col,
                )
            )

    conn.close()
    return Schema(db_id=db_id, tables=tables, foreign_keys=fks, db_path=db_path)


class CGSP:
    """Complexity-Gated Schema Pruner."""

    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs)

    def _rank_tables(self, question: str, schema: Schema) -> list[tuple[str, float]]:
        names = schema.all_table_names()
        if not names:
            return []
        table_texts = [schema.tables[n].textual() for n in names]
        q_emb = self._embed([question])[0]
        t_embs = self._embed(table_texts)
        sims = t_embs @ q_emb  # cosine sim since embeddings are normalized
        ranked = sorted(zip(names, sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked

    def _fk_neighbors(self, schema: Schema, selected: list[str]) -> list[str]:
        sel_set = set(selected)
        neighbors: list[str] = []
        for fk in schema.foreign_keys:
            if fk.from_table in sel_set and fk.to_table not in sel_set:
                neighbors.append(fk.to_table)
            if fk.to_table in sel_set and fk.from_table not in sel_set:
                neighbors.append(fk.from_table)
        # dedupe, preserve order
        seen, out = set(), []
        for n in neighbors:
            if n not in seen and n in schema.tables:
                seen.add(n)
                out.append(n)
        return out

    def _sample_rows(self, schema: Schema, table_name: str, n: int = 3) -> list[tuple]:
        try:
            conn = sqlite3.connect(str(schema.db_path))
            conn.text_factory = lambda x: x.decode("utf-8", errors="replace")
            cur = conn.cursor()
            cur.execute(f'SELECT * FROM "{table_name}" LIMIT {n}')
            rows = cur.fetchall()
            conn.close()
            return rows
        except sqlite3.Error:
            return []

    def _format_create_table(self, table: Table) -> str:
        cols = []
        for c in table.columns:
            pk = " PRIMARY KEY" if c.is_pk else ""
            cols.append(f'    "{c.name}" {c.dtype}{pk}')
        return f'CREATE TABLE "{table.name}" (\n' + ",\n".join(cols) + "\n);"

    def _format_fks_for_tables(self, schema: Schema, table_names: set[str]) -> str:
        lines = []
        for fk in schema.foreign_keys:
            if fk.from_table in table_names and fk.to_table in table_names:
                lines.append(
                    f"-- {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}"
                )
        return "\n".join(lines)

    def prune(
        self, question: str, schema: Schema, tier: Tier
    ) -> tuple[str, list[str]]:
        """
        Apply tier-gated schema pruning.

        Returns:
            (schema_string, selected_table_names)
        """
        ranked = self._rank_tables(question, schema)

        if tier == Tier.SIMPLE:
            selected = [n for n, _ in ranked[:2]]
            include_samples = False
        elif tier == Tier.MEDIUM:
            top4 = [n for n, _ in ranked[:4]]
            neighbors = self._fk_neighbors(schema, top4)
            selected = top4 + neighbors
            include_samples = False
        else:  # COMPLEX
            selected = schema.all_table_names()
            include_samples = True

        # Build schema string
        parts: list[str] = []
        for tname in selected:
            parts.append(self._format_create_table(schema.tables[tname]))
            if include_samples:
                rows = self._sample_rows(schema, tname, n=3)
                if rows:
                    parts.append(
                        f"-- 3 sample rows from {tname}:\n"
                        + "\n".join(f"-- {r}" for r in rows)
                    )

        fk_text = self._format_fks_for_tables(schema, set(selected))
        if fk_text:
            parts.append("-- Foreign keys:\n" + fk_text)

        return "\n\n".join(parts), selected


if __name__ == "__main__":
    # Smoke test on a synthetic in-memory schema
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = Path(f.name)

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER,
            FOREIGN KEY(dept_id) REFERENCES departments(id));
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, building TEXT);
        CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, credits INTEGER);
        CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade REAL);
        INSERT INTO students VALUES (1,'Alice',10), (2,'Bob',20);
        INSERT INTO departments VALUES (10,'CS','Hall A'), (20,'Math','Hall B');
        """
    )
    conn.commit()
    conn.close()

    schema = load_schema_from_sqlite(path, "test_db")
    cgsp = CGSP()
    for tier in [Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX]:
        s, sel = cgsp.prune("list all students and their departments", schema, tier)
        print(f"\n=== Tier: {tier.value} ===")
        print(f"Selected tables: {sel}")
        print(f"Schema length: {len(s)} chars")

    path.unlink()
