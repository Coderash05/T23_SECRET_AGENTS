"""Plain-English Explainer for S5 Presentation and Audit Trace.

Transforms classified clauses, extracted facts, and symbolic rule justifications
into transparent, plain-English founder summaries with automated number consistency
and standard market exemplar citations retrieved via Stage S2 SBERT.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from termsheet_ai.explain.consistency import NumberConsistencyChecker
from termsheet_ai.pipeline.base import BaseExplainer
from termsheet_ai.schemas.contracts import (
    AuditTrace,
    ClauseFamily,
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
        retriever: Optional[Any] = None,
    ):
        self.checker = checker or NumberConsistencyChecker()
        self.facts = facts
        self.retriever = retriever

    def _get_market_reference(self, clause: ClauseSpan) -> Optional[Dict[str, Any]]:
        """Retrieve standard market exemplar phrasing via SBERT ChromaDB index."""
        if self.retriever is not None and hasattr(self.retriever, "retrieve_nearest_exemplars"):
            try:
                results = self.retriever.retrieve_nearest_exemplars(clause, top_k=1)
                if results:
                    top = results[0]
                    # Only cite if similarity is solid and clause has an identified family
                    if (
                        top.get("cosine_similarity", 0.0) >= 0.35
                        and clause.detected_family != ClauseFamily.OTHER
                    ):
                        return top
            except Exception:
                pass
        return None

    def _format_flag_explanation(
        self,
        clause: ClauseSpan,
        flag: RiskFlag,
        market_ref: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Builds a structured plain-English explanation for a flagged term."""
        sev = flag.severity.value.upper()
        norm = flag.market_standard_source or "NVCA Model Term Sheet"

        explanation = (
            f"[{sev} RISK] {flag.title}: {flag.explanation} "
            f"Market standard: Under {norm}, this term deviates from conventional founder-friendly terms."
        )
        if market_ref:
            explanation += (
                f' Standard market reference ({market_ref["source"]}): "{market_ref["text"]}"'
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
            market_ref = self._get_market_reference(clause)

            allowed_extras: set[float] = set()
            if market_ref:
                if "text" in market_ref:
                    allowed_extras.update(self.checker.extract_numbers(market_ref["text"]))
                if "source" in market_ref:
                    allowed_extras.update(self.checker.extract_numbers(market_ref["source"]))

            if clause_flags:
                # Merge explanations for all flags on this clause
                details = " ".join(
                    self._format_flag_explanation(clause, flag, market_ref) for flag in clause_flags
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
                if market_ref:
                    summary_text += (
                        f' Standard reference ({market_ref["source"]}): "{market_ref["text"]}"'
                    )
                first_justification = None

            # Verify number consistency against facts, clause text, and allowed exemplar numbers
            verified, _ = self.checker.verify_summary(
                summary_text=summary_text,
                clause=clause,
                facts=effective_facts,
                justification=first_justification,
                allowed_extras=allowed_extras,
            )

            summaries.append(
                ClauseSummary(
                    clause_id=clause.clause_id,
                    summary_text=summary_text,
                    numbers_verified=verified,
                )
            )

        return summaries
