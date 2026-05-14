"""CAS-SQL pipeline orchestrator. Stage 1 → Stage 2 → Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

from cas_sql.ccsg import CCSG, GenerationResult
from cas_sql.cgsp import CGSP, Schema
from cas_sql.hce import HCEResult, Tier, estimate


@dataclass
class PipelineResult:
    question: str
    db_id: str
    tier: str
    score: int
    fired: dict[str, bool]
    selected_tables: list[str]
    sql: str
    raw_output: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "db_id": self.db_id,
            "tier": self.tier,
            "score": self.score,
            "fired": self.fired,
            "selected_tables": self.selected_tables,
            "sql": self.sql,
            "raw_output": self.raw_output,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "model": self.model,
            "error": self.error,
        }


class CASSQLPipeline:
    def __init__(self, cgsp: CGSP, ccsg: CCSG):
        self.cgsp = cgsp
        self.ccsg = ccsg

    def run(self, question: str, schema: Schema) -> PipelineResult:
        hce_res: HCEResult = estimate(question)
        schema_str, selected = self.cgsp.prune(question, schema, hce_res.tier)
        gen: GenerationResult = self.ccsg.generate(question, schema_str, hce_res.tier)
        return PipelineResult(
            question=question,
            db_id=schema.db_id,
            tier=hce_res.tier.value,
            score=hce_res.score,
            fired=hce_res.fired,
            selected_tables=selected,
            sql=gen.sql,
            raw_output=gen.raw_output,
            prompt_tokens=gen.prompt_tokens,
            output_tokens=gen.output_tokens,
            latency_s=gen.latency_s,
            model=gen.model,
            error=gen.error,
        )
