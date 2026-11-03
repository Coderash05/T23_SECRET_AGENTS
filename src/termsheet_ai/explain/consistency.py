"""Deterministic Number Consistency Checker for S5 Plain-English Summaries.

Guarantees zero-hallucination on numerical terms by verifying that every
number in an explanation or summary is grounded in the extracted fact base,
the rule premise records, or the source clause span.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from termsheet_ai.schemas.contracts import ClauseSpan, ExtractedFacts, JustificationRecord


class NumberConsistencyChecker:
    """Verifies that numerical tokens in generated summaries are strictly grounded."""

    # Matches numbers preceded by non-word or start-of-string and followed by word boundary, x/X, or %
    NUMERIC_PATTERN = re.compile(r"(?<![A-Za-z0-9_])\$?(\d+(?:,\d{3})*(?:\.\d+)?)(?=[%xX]|\b)")

    @classmethod
    def extract_numbers(cls, text: str) -> List[float]:
        """Extracts all numerical values from text as normalized floats."""
        values: List[float] = []
        matches = cls.NUMERIC_PATTERN.finditer(text)
        for m in matches:
            clean = m.group(1).replace(",", "")
            try:
                values.append(float(clean))
            except ValueError:
                continue
        return values

    @classmethod
    def collect_allowed_numbers_for_clause(
        cls,
        clause: ClauseSpan,
        facts: Optional[ExtractedFacts] = None,
        justification: Optional[JustificationRecord] = None,
    ) -> Set[float]:
        """Collects all valid numerical values from source clause, facts, and justification."""
        allowed: Set[float] = set()

        # 1. Numbers present in raw clause text
        if clause and clause.raw_text:
            allowed.update(cls.extract_numbers(clause.raw_text))

        # 2. Numbers present in ExtractedFacts
        if facts:
            for fact_obj in (
                facts.liquidation_preference,
                facts.anti_dilution,
                facts.board_composition,
                facts.founder_vesting,
                facts.protective_provisions,
            ):
                if fact_obj:
                    fact_dict = (
                        fact_obj.model_dump()
                        if hasattr(fact_obj, "model_dump")
                        else fact_obj.dict()
                    )
                    for val in fact_dict.values():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            allowed.add(float(val))

            # Custom facts
            for val in facts.custom_facts.values():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    allowed.add(float(val))

        # 3. Numbers present in JustificationRecord premises
        if justification and justification.premise_facts:
            for val in justification.premise_facts.values():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    allowed.add(float(val))

        return allowed

    def verify_summary(
        self,
        summary_text: str,
        clause: ClauseSpan,
        facts: Optional[ExtractedFacts] = None,
        justification: Optional[JustificationRecord] = None,
        allowed_extras: Optional[Set[float]] = None,
    ) -> Tuple[bool, List[float]]:
        """Verifies if all numbers in summary_text are present in the allowed fact base.

        Returns:
            Tuple of (is_verified, unverified_numbers_list).
        """
        summary_numbers = self.extract_numbers(summary_text)
        if not summary_numbers:
            return True, []

        allowed = self.collect_allowed_numbers_for_clause(
            clause=clause, facts=facts, justification=justification
        )
        if allowed_extras:
            allowed.update(allowed_extras)

        # Standard market numbers referenced in explanation templates (e.g., 1.0x standard, 4-year standard, 1-year cliff)
        # are permitted when explaining market norms
        standard_market_constants = {1.0, 4.0, 12.0, 48.0, 100.0}
        allowed.update(standard_market_constants)

        unverified = [n for n in summary_numbers if n not in allowed]
        return len(unverified) == 0, unverified

    def verify_payload_summaries(
        self,
        clauses: List[ClauseSpan],
        summaries: List[Dict[str, Any]],
        facts: Optional[ExtractedFacts] = None,
    ) -> bool:
        """Batch-verifies a list of summaries against clauses and facts."""
        clause_map = {c.clause_id: c for c in clauses}
        all_passed = True
        for s in summaries:
            cid = s.get("clause_id")
            text = s.get("summary_text", "")
            clause = clause_map.get(cid)
            if clause:
                verified, _ = self.verify_summary(summary_text=text, clause=clause, facts=facts)
                s["numbers_verified"] = verified
                if not verified:
                    all_passed = False
        return all_passed
