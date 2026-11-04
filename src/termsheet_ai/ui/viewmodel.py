"""Pure helpers that shape a PipelinePayload for display (no Streamlit import)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from termsheet_ai.schemas.contracts import (
    ClauseSpan,
    ClauseSummary,
    PipelinePayload,
    RiskFlag,
)
from termsheet_ai.ui.theme import sev_rank, sev_value


@dataclass(frozen=True)
class ClauseRow:
    clause: ClauseSpan
    summary: Optional[ClauseSummary]
    flags: Tuple[RiskFlag, ...]

    @property
    def worst_severity(self) -> Optional[str]:
        if not self.flags:
            return None
        return sev_value(min(self.flags, key=lambda f: sev_rank(f.severity)).severity)


def build_clause_rows(payload: PipelinePayload) -> List[ClauseRow]:
    summaries = {s.clause_id: s for s in payload.summaries}
    flags = payload.audit_trace.flags if payload.audit_trace else []
    rows = []
    for clause in payload.clauses:
        own = tuple(
            sorted(
                (f for f in flags if f.clause_id == clause.clause_id),
                key=lambda f: sev_rank(f.severity),
            )
        )
        rows.append(ClauseRow(clause=clause, summary=summaries.get(clause.clause_id), flags=own))
    return rows


def sorted_flags(payload: PipelinePayload) -> List[RiskFlag]:
    """Worst first; ties keep document order of their clause."""
    if not payload.audit_trace:
        return []
    order = {c.clause_id: i for i, c in enumerate(payload.clauses)}
    return sorted(
        payload.audit_trace.flags,
        key=lambda f: (sev_rank(f.severity), order.get(f.clause_id, 10**6)),
    )


def clause_label(payload: PipelinePayload, clause_id: str) -> str:
    for c in payload.clauses:
        if c.clause_id == clause_id:
            return c.section_header or c.clause_id
    return clause_id
