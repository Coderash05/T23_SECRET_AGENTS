"""Plain-English Explainer for S5 Presentation and Audit Trace.

Transforms classified clauses, extracted facts, and symbolic rule justifications
into transparent, plain-English founder summaries with automated number consistency.
"""

from __future__ import annotations

from typing import List, Optional

from termsheet_ai.explain.consistency import NumberConsistencyChecker
from termsheet_ai.pipeline.base import BaseExplainer
from termsheet_ai.schemas.contracts import (
    AuditTrace,
    ClauseSpan,
    ClauseSummary,
    ExtractedFacts,
    RiskFlag,
)


class PlainEnglishExplainer(BaseExplainer):
    """Generates founder-friendly explanations backed by NVCA citations and verified numbers."""

    def __init__(
        self,
        checker: Optional[NumberConsistencyChecker] = None,
        facts: Optional[ExtractedFacts] = None,
    ):
        self.checker = checker or NumberConsistencyChecker()
        self.facts = facts

    def _format_flag_explanation(self, clause: ClauseSpan, flag: RiskFlag) -> str:
        """Builds a structured plain-English explanation for a flagged term."""
        sev = flag.severity.value.upper()
        norm = flag.market_standard_source or "NVCA Model Term Sheet"

        explanation = (
            f"[{sev} RISK] {flag.title}: {flag.explanation} "
            f"Market standard: Under {norm}, this term deviates from conventional founder-friendly terms."
        )
        return explanation

    def explain(
        self,
        clauses: List[ClauseSpan],
        audit_trace: AuditTrace,
        facts: Optional[ExtractedFacts] = None,
    ) -> List[ClauseSummary]:
        """Generates verified plain-English summaries alongside original clauses."""
        effective_facts = facts or self.facts
        flags_by_clause: dict[str, List[RiskFlag]] = {}
        for f in audit_trace.flags:
            flags_by_clause.setdefault(f.clause_id, []).append(f)

        summaries: List[ClauseSummary] = []

        for clause in clauses:
            clause_flags = flags_by_clause.get(clause.clause_id, [])

            if clause_flags:
                # Merge explanations for all flags on this clause
                details = " ".join(
                    self._format_flag_explanation(clause, flag) for flag in clause_flags
                )
                summary_text = (
                    f"Clause {clause.clause_id} ({clause.detected_family.value}): {details}"
                )
                first_justification = clause_flags[0].justification
            else:
                family_name = clause.detected_family.value.replace("_", " ").title()
                summary_text = (
                    f"Clause {clause.clause_id} ({clause.detected_family.value}): Standard {family_name} provision. "
                    f"No non-standard or off-market risks detected against NVCA guidelines."
                )
                first_justification = None

            # Verify number consistency against facts and clause text
            verified, _ = self.checker.verify_summary(
                summary_text=summary_text,
                clause=clause,
                facts=effective_facts,
                justification=first_justification,
            )

            summaries.append(
                ClauseSummary(
                    clause_id=clause.clause_id,
                    summary_text=summary_text,
                    numbers_verified=verified,
                )
            )

        return summaries
