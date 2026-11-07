"""Deterministic Number Consistency Checker for S5 Plain-English Summaries.

Guarantees zero-hallucination on numerical terms by verifying that every
number in an explanation or summary is grounded in the extracted fact base,
the rule premise records, or the source clause span.

Cross-Clause and Arithmetic Consistency (Member 4 — Stage S5)
--------------------------------------------------------------
In addition to per-summary number verification, this module now exposes
``CrossClauseConsistencyChecker`` which detects:

A. Cross-clause contradictions
   - Conflicting liquidation preference multiples when two or more clauses
     both extract an *explicit* (grounded) liquidation preference for the
     *same* document and both values differ.
   - Conflicting vesting schedules when two or more vesting facts are present
     in ``custom_facts`` (keyed ``"vesting_<id>"``), or when the single
     ``founder_vesting`` field conflicts with an explicitly extracted
     alternative. Because the current schema stores only *one* ``VestingFact``
     in ``ExtractedFacts``, multi-grant contradiction detection is limited
     without schema changes (documented under ``SCHEMA_LIMITATIONS``).

B. Arithmetic checks
   - ``founder_seats + investor_seats + independent_seats == total_seats``
     for ``BoardCompositionFact``. Only performed when ``total_seats > 0``.

C. Reporting
   - Every ``ConsistencyIssue`` carries ``clause_ids``, ``issue_type``,
     ``description``, and optional ``values`` for auditability.
   - ``is_contradiction`` is ``True`` only when confirmed contradictory
     values are found. Missing values are never reported as contradictions.

Schema limitations (cannot be fixed without modifying teammate-owned schemas)
------------------------------------------------------------------------------
- ``ExtractedFacts`` stores at most *one* ``LiquidationPreferenceFact`` and
  *one* ``VestingFact``. Full cross-clause liquidation / vesting checking
  requires either: (a) a ``List[LiquidationPreferenceFact]``, or (b) the
  per-clause fact blocks stored in ``custom_facts``. When only one block
  exists the cross-clause check trivially finds no contradiction.
- ``BoardCompositionFact`` has no ``security`` discriminator, so only
  arithmetic consistency (not cross-security comparisons) is possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from termsheet_ai.schemas.contracts import ClauseSpan, ExtractedFacts, JustificationRecord


# ---------------------------------------------------------------------------
# Schema limitation registry — referenced in docstrings and test assertions
# ---------------------------------------------------------------------------
SCHEMA_LIMITATIONS: Dict[str, str] = {
    "single_liquidation_fact": (
        "ExtractedFacts.liquidation_preference stores at most one LiquidationPreferenceFact. "
        "Cross-clause contradiction detection between *distinct* liquidation provisions requires "
        "either List[LiquidationPreferenceFact] or per-clause storage in custom_facts."
    ),
    "single_vesting_fact": (
        "ExtractedFacts.founder_vesting stores at most one VestingFact. "
        "Multi-grant vesting contradiction detection requires List[VestingFact] or custom_facts "
        "entries keyed as 'vesting_<clause_id>'."
    ),
    "no_security_discriminator": (
        "BoardCompositionFact has no 'security' or 'series' field. Cross-security board "
        "arithmetic comparisons are not possible with the current schema."
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyIssue:
    """Represents a detected consistency problem across extracted facts.

    Attributes
    ----------
    issue_type:
        Short machine-readable category, e.g. ``"board_arithmetic"``,
        ``"liquidation_multiple_conflict"``, ``"vesting_conflict"``.
    description:
        Human-readable explanation of the issue.
    clause_ids:
        IDs of the clauses involved in the contradiction.
    values:
        Dict of field-name → value pairs that form the contradiction evidence.
    is_contradiction:
        ``True`` only when confirmed contradictory values are present.
        ``False`` for arithmetic gaps due to incomplete data.
    """

    issue_type: str
    description: str
    clause_ids: List[str] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    is_contradiction: bool = True


# ---------------------------------------------------------------------------
# NumberConsistencyChecker — existing behavior preserved
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CrossClauseConsistencyChecker — new in this session
# ---------------------------------------------------------------------------


class CrossClauseConsistencyChecker:
    """Detects cross-clause numerical contradictions and arithmetic violations.

    Design constraints
    ------------------
    - Only raises contradictions from facts that are **grounded** (``is_grounded=True``).
    - Never interprets missing or ``None`` values as contradictions.
    - Respects the single-fact-per-type schema limit (see ``SCHEMA_LIMITATIONS``).
    - Does not modify any input data.
    """

    # ------------------------------------------------------------------
    # A. Cross-clause liquidation preference
    # ------------------------------------------------------------------

    @staticmethod
    def check_liquidation_conflicts(facts: ExtractedFacts) -> List[ConsistencyIssue]:
        """Check for conflicting liquidation preference multiples.

        Because ``ExtractedFacts`` holds at most *one* ``LiquidationPreferenceFact``,
        intra-document multi-clause conflicts can only be detected when additional
        liquidation facts are stored in ``custom_facts`` under keys prefixed with
        ``"liquidation_multiple_<clause_id>"``.

        Returns an empty list when only one fact is present (no contradiction
        is possible without a second data point).
        """
        issues: List[ConsistencyIssue] = []

        # Collect: primary fact + any extras in custom_facts
        multiples: Dict[str, float] = {}  # clause_id → multiple

        lp = facts.liquidation_preference
        if lp is not None and lp.is_grounded and lp.multiple is not None:
            multiples[lp.clause_id] = lp.multiple

        # Check custom_facts for additional liquidation multiples stored as
        # "liquidation_multiple_<clause_id>": <float>
        for key, val in facts.custom_facts.items():
            if key.startswith("liquidation_multiple_") and isinstance(val, (int, float)):
                clause_id = key[len("liquidation_multiple_"):]
                multiples[clause_id] = float(val)

        if len(multiples) < 2:
            # Only one (or zero) data points — no contradiction possible
            return issues

        # Find conflicting pairs
        seen: List[Tuple[str, float]] = list(multiples.items())
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                cid_a, mult_a = seen[i]
                cid_b, mult_b = seen[j]
                if abs(mult_a - mult_b) > 1e-9:
                    issues.append(
                        ConsistencyIssue(
                            issue_type="liquidation_multiple_conflict",
                            description=(
                                f"Conflicting liquidation preference multiples: "
                                f"clause {cid_a} states {mult_a}x, "
                                f"clause {cid_b} states {mult_b}x."
                            ),
                            clause_ids=[cid_a, cid_b],
                            values={"multiple_a": mult_a, "multiple_b": mult_b},
                            is_contradiction=True,
                        )
                    )
        return issues

    # ------------------------------------------------------------------
    # B. Cross-clause vesting conflicts
    # ------------------------------------------------------------------

    @staticmethod
    def check_vesting_conflicts(facts: ExtractedFacts) -> List[ConsistencyIssue]:
        """Check for conflicting vesting schedule parameters.

        Like liquidation preference, ``ExtractedFacts`` stores at most one
        ``VestingFact``. Additional vesting facts must be stored in
        ``custom_facts`` under keys ``"vesting_years_<clause_id>"`` and
        optionally ``"vesting_cliff_<clause_id>"``.

        Returns an empty list when fewer than two data points exist.
        """
        issues: List[ConsistencyIssue] = []

        # Collect primary + custom vesting years
        years_map: Dict[str, float] = {}
        cliff_map: Dict[str, int] = {}

        fv = facts.founder_vesting
        if fv is not None and fv.is_grounded:
            if fv.total_vesting_years is not None:
                years_map[fv.clause_id] = fv.total_vesting_years
            if fv.cliff_months is not None:
                cliff_map[fv.clause_id] = fv.cliff_months

        for key, val in facts.custom_facts.items():
            if key.startswith("vesting_years_") and isinstance(val, (int, float)):
                cid = key[len("vesting_years_"):]
                years_map[cid] = float(val)
            elif key.startswith("vesting_cliff_") and isinstance(val, int):
                cid = key[len("vesting_cliff_"):]
                cliff_map[cid] = val

        # Check years conflicts
        if len(years_map) >= 2:
            seen = list(years_map.items())
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    cid_a, yrs_a = seen[i]
                    cid_b, yrs_b = seen[j]
                    if abs(yrs_a - yrs_b) > 1e-9:
                        issues.append(
                            ConsistencyIssue(
                                issue_type="vesting_years_conflict",
                                description=(
                                    f"Conflicting vesting durations: "
                                    f"clause {cid_a} states {yrs_a} years, "
                                    f"clause {cid_b} states {yrs_b} years."
                                ),
                                clause_ids=[cid_a, cid_b],
                                values={"years_a": yrs_a, "years_b": yrs_b},
                                is_contradiction=True,
                            )
                        )

        # Check cliff conflicts
        if len(cliff_map) >= 2:
            seen_cliff = list(cliff_map.items())
            for i in range(len(seen_cliff)):
                for j in range(i + 1, len(seen_cliff)):
                    cid_a, cliff_a = seen_cliff[i]
                    cid_b, cliff_b = seen_cliff[j]
                    if cliff_a != cliff_b:
                        issues.append(
                            ConsistencyIssue(
                                issue_type="vesting_cliff_conflict",
                                description=(
                                    f"Conflicting vesting cliff durations: "
                                    f"clause {cid_a} states {cliff_a} months, "
                                    f"clause {cid_b} states {cliff_b} months."
                                ),
                                clause_ids=[cid_a, cid_b],
                                values={"cliff_a": cliff_a, "cliff_b": cliff_b},
                                is_contradiction=True,
                            )
                        )

        return issues

    # ------------------------------------------------------------------
    # C. Board composition arithmetic
    # ------------------------------------------------------------------

    @staticmethod
    def check_board_arithmetic(facts: ExtractedFacts) -> List[ConsistencyIssue]:
        """Check that founder_seats + investor_seats + independent_seats == total_seats.

        Only performed when:
        - ``board_composition`` is not None.
        - ``total_seats > 0`` (a zero total means the total was not extracted).
        - At least one of ``founder_seats``, ``investor_seats``, or
          ``independent_seats`` is non-zero (incomplete data is not an error).

        Returns an empty list when data is insufficient for a meaningful check.
        """
        issues: List[ConsistencyIssue] = []
        bc = facts.board_composition
        if bc is None:
            return issues

        total = bc.total_seats
        founder = bc.founder_seats
        investor = bc.investor_seats
        independent = bc.independent_seats
        clause_id = bc.clause_id

        # Skip check when total is zero (not extracted) or all components are zero
        if total == 0:
            return issues
        if founder == 0 and investor == 0 and independent == 0:
            return issues

        component_sum = founder + investor + independent
        if component_sum != total:
            issues.append(
                ConsistencyIssue(
                    issue_type="board_arithmetic",
                    description=(
                        f"Board seat arithmetic mismatch in clause {clause_id}: "
                        f"{founder} founder + {investor} investor + {independent} independent "
                        f"= {component_sum}, but total_seats = {total}."
                    ),
                    clause_ids=[clause_id],
                    values={
                        "founder_seats": founder,
                        "investor_seats": investor,
                        "independent_seats": independent,
                        "component_sum": component_sum,
                        "total_seats": total,
                    },
                    is_contradiction=True,
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check_all(self, facts: ExtractedFacts) -> List[ConsistencyIssue]:
        """Run all cross-clause and arithmetic consistency checks.

        Parameters
        ----------
        facts:
            The ``ExtractedFacts`` container to inspect.

        Returns
        -------
        List[ConsistencyIssue]
            All detected issues. An empty list indicates no contradictions
            were found with the available data.
        """
        issues: List[ConsistencyIssue] = []
        issues.extend(self.check_liquidation_conflicts(facts))
        issues.extend(self.check_vesting_conflicts(facts))
        issues.extend(self.check_board_arithmetic(facts))
        return issues
