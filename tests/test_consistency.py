"""Tests for NumberConsistencyChecker and CrossClauseConsistencyChecker (Member 4 / S5).

Coverage
--------
Existing (preserved)
  1. extract_numbers — basic extraction
  2. verify_summary — consistent summary passes
  3. verify_summary — hallucinated number fails
  4. PlainEnglishExplainer with SBERT retriever cites market exemplar

New — NumberConsistencyChecker
  5. No numbers in summary → verified
  6. Standard market constants are always allowed
  7. Numbers in JustificationRecord premises are allowed

New — CrossClauseConsistencyChecker: board arithmetic
  8. Valid board arithmetic — no issues
  9. Invalid board arithmetic — issue raised
  10. Total seats zero — skipped
  11. All component seats zero — skipped
  12. Single seat type non-zero, total provided — still checked

New — CrossClauseConsistencyChecker: liquidation conflicts
  13. Single liquidation fact — no conflict
  14. Two matching liquidation multiples in custom_facts — no conflict
  15. Two conflicting liquidation multiples — issue raised
  16. Ungrounded primary fact is ignored

New — CrossClauseConsistencyChecker: vesting conflicts
  17. Single vesting fact — no conflict
  18. Two matching vesting years in custom_facts — no conflict
  19. Two conflicting vesting years — issue raised
  20. Two conflicting cliff months — separate issue raised
  21. Missing values not treated as contradictions

New — multiple issues in one pass
  22. Board + liquidation conflict both detected

New — production extractor selection
  23. Runner's production path resolves to Member 4's FactExtractor class
"""

from __future__ import annotations

from termsheet_ai.explain.consistency import (
    SCHEMA_LIMITATIONS,
    CrossClauseConsistencyChecker,
    NumberConsistencyChecker,
)
from termsheet_ai.explain.explainer import PlainEnglishExplainer
from termsheet_ai.schemas.contracts import (
    AuditTrace,
    BoardCompositionFact,
    ClauseFamily,
    ClauseSpan,
    DocumentRiskScore,
    ExtractedFacts,
    JustificationRecord,
    LiquidationPreferenceFact,
    ParticipationType,
    SourceSpan,
    VestingFact,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _source_span(text: str = "placeholder") -> SourceSpan:
    return SourceSpan(text=text)


def _lp_fact(
    clause_id: str,
    multiple: float,
    grounded: bool = True,
    raw_text: str = "",
) -> LiquidationPreferenceFact:
    return LiquidationPreferenceFact(
        clause_id=clause_id,
        multiple=multiple,
        participation=ParticipationType.NON_PARTICIPATING,
        source_span=_source_span(raw_text or f"{multiple}x liquidation preference"),
        is_grounded=grounded,
    )


def _vesting_fact(
    clause_id: str,
    total_years: float,
    cliff_months: int = 12,
    grounded: bool = True,
) -> VestingFact:
    return VestingFact(
        clause_id=clause_id,
        total_vesting_years=total_years,
        cliff_months=cliff_months,
        source_span=_source_span(f"{total_years} years vesting"),
        is_grounded=grounded,
    )


def _board_fact(
    clause_id: str,
    founder: int,
    investor: int,
    independent: int,
    total: int,
) -> BoardCompositionFact:
    return BoardCompositionFact(
        clause_id=clause_id,
        founder_seats=founder,
        investor_seats=investor,
        independent_seats=independent,
        total_seats=total,
        investor_majority=(investor > founder),
        source_span=_source_span(f"board of {total}"),
    )


def _empty_facts(doc_id: str = "doc_1") -> ExtractedFacts:
    return ExtractedFacts(doc_id=doc_id)


# ===========================================================================
# SECTION 1 — Existing tests (preserved exactly)
# ===========================================================================


def test_extract_numbers():
    text = (
        "The Series A receives a 2.5x multiple with a $5,000,000 cap and 20% interest over 4 years."
    )
    numbers = NumberConsistencyChecker.extract_numbers(text)
    assert 2.5 in numbers
    assert 5000000.0 in numbers
    assert 20.0 in numbers
    assert 4.0 in numbers


def test_consistent_summary_passes():
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="Series A Preferred will receive a 2.0x liquidation preference, participating fully.",
        page_number=1,
    )
    facts = ExtractedFacts(
        doc_id="doc_1",
        liquidation_preference=LiquidationPreferenceFact(
            clause_id="c_1",
            multiple=2.0,
            participation=ParticipationType.FULLY_PARTICIPATING,
            source_span=SourceSpan(text=clause.raw_text, page_number=1),
        ),
    )

    checker = NumberConsistencyChecker()
    summary = (
        "This clause awards Series A a 2.0x liquidation preference payout, above the standard 1.0x."
    )
    verified, unverified = checker.verify_summary(summary, clause=clause, facts=facts)

    assert verified is True
    assert unverified == []


def test_hallucinated_number_fails():
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="Series A Preferred will receive a 2.0x liquidation preference.",
        page_number=1,
    )
    facts = ExtractedFacts(
        doc_id="doc_1",
        liquidation_preference=LiquidationPreferenceFact(
            clause_id="c_1",
            multiple=2.0,
            participation=ParticipationType.NON_PARTICIPATING,
            source_span=SourceSpan(text=clause.raw_text, page_number=1),
        ),
    )

    checker = NumberConsistencyChecker()
    hallucinated_summary = "This clause gives investors a 3.5x liquidation multiple payout."
    verified, unverified = checker.verify_summary(hallucinated_summary, clause=clause, facts=facts)

    assert verified is False
    assert 3.5 in unverified


def test_explainer_with_sbert_retriever_cites_market_exemplar():
    class MockRetriever:
        def retrieve_nearest_exemplars(self, clause, top_k=1):
            return [
                {
                    "text": "1.0x non-participating preferred is standard.",
                    "family": ClauseFamily.LIQUIDATION_PREFERENCE,
                    "source": "NVCA Model Term Sheet §3",
                    "cosine_similarity": 0.88,
                }
            ]

    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="The Series A Preferred shall have a 1.0x non-participating liquidation preference.",
        page_number=1,
        detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
    )
    trace = AuditTrace(
        doc_id="doc_1",
        risk_score=DocumentRiskScore(overall_score=0.0),
        flags=[],
        rule_traces=[],
    )

    explainer = PlainEnglishExplainer(retriever=MockRetriever())
    summaries = explainer.explain([clause], trace)

    assert len(summaries) == 1
    assert "NVCA Model Term Sheet §3" in summaries[0].summary_text
    assert summaries[0].numbers_verified is True


# ===========================================================================
# SECTION 2 — NumberConsistencyChecker extended tests
# ===========================================================================


def test_no_numbers_in_summary_is_verified():
    """Summary with no numerical tokens is always verified."""
    clause = ClauseSpan(clause_id="c_x", raw_text="Anti-dilution clause.", page_number=1)
    checker = NumberConsistencyChecker()
    verified, unverified = checker.verify_summary(
        "This clause provides standard anti-dilution protection.", clause=clause
    )
    assert verified is True
    assert unverified == []


def test_standard_market_constants_are_always_allowed():
    """Standard constants 1.0, 4.0, 12.0, 48.0, 100.0 are allowed even without matching facts."""
    clause = ClauseSpan(clause_id="c_x", raw_text="Some clause text.", page_number=1)
    checker = NumberConsistencyChecker()
    # 1.0 and 4.0 appear in the summary but NOT in clause text — should be allowed via constants
    verified, _ = checker.verify_summary(
        "Market standard is a 1.0x preference over 4.0 years.", clause=clause
    )
    assert verified is True


def test_justification_premise_numbers_are_allowed():
    """Numbers in JustificationRecord.premise_facts are included in the allowed set."""
    clause = ClauseSpan(clause_id="c_1", raw_text="Some clause.", page_number=1)
    justification = JustificationRecord(
        rule_id="RULE_X",
        rule_name="Test rule",
        market_norm_cite="NVCA §1",
        premise_facts={"multiple": 2.5},
        conclusion_flag="High multiple",
    )
    checker = NumberConsistencyChecker()
    # 2.5 is NOT in the clause text, but IS in premise_facts
    verified, _ = checker.verify_summary(
        "The 2.5x multiple is flagged as high risk.",
        clause=clause,
        justification=justification,
    )
    assert verified is True


# ===========================================================================
# SECTION 3 — Board arithmetic checks
# ===========================================================================


def test_board_arithmetic_valid_no_issues():
    """2 + 2 + 1 = 5 — no issue raised."""
    facts = ExtractedFacts(
        doc_id="doc_1",
        board_composition=_board_fact("c_board", founder=2, investor=2, independent=1, total=5),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_board_arithmetic(facts)
    assert issues == []


def test_board_arithmetic_mismatch_raises_issue():
    """2 + 2 + 1 = 5 but total says 6 — contradiction."""
    facts = ExtractedFacts(
        doc_id="doc_1",
        board_composition=_board_fact("c_board", founder=2, investor=2, independent=1, total=6),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_board_arithmetic(facts)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "board_arithmetic"
    assert issue.is_contradiction is True
    assert "c_board" in issue.clause_ids
    assert issue.values["component_sum"] == 5
    assert issue.values["total_seats"] == 6


def test_board_arithmetic_zero_total_skipped():
    """When total_seats == 0, no arithmetic check is performed."""
    facts = ExtractedFacts(
        doc_id="doc_1",
        board_composition=_board_fact("c_board", founder=2, investor=2, independent=1, total=0),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_board_arithmetic(facts)
    assert issues == []


def test_board_arithmetic_all_components_zero_skipped():
    """When all component seats are 0 (incomplete data), no check is performed."""
    facts = ExtractedFacts(
        doc_id="doc_1",
        board_composition=_board_fact("c_board", founder=0, investor=0, independent=0, total=5),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_board_arithmetic(facts)
    assert issues == []


def test_board_arithmetic_no_board_fact_no_issues():
    """No board fact — returns empty list."""
    facts = _empty_facts()
    checker = CrossClauseConsistencyChecker()
    assert checker.check_board_arithmetic(facts) == []


def test_board_arithmetic_partial_seats_still_checked():
    """Investor-only seats with known total — arithmetic still checked when non-zero total."""
    # 0 + 3 + 0 = 3 but total says 5
    facts = ExtractedFacts(
        doc_id="doc_1",
        board_composition=_board_fact("c_b", founder=0, investor=3, independent=0, total=5),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_board_arithmetic(facts)
    assert len(issues) == 1
    assert issues[0].values["component_sum"] == 3


def test_board_arithmetic_exact_match_one_seat_type():
    """3 investor + 0 + 0 = 3 total — no issue."""
    facts = ExtractedFacts(
        doc_id="doc_1",
        board_composition=_board_fact("c_b", founder=0, investor=3, independent=0, total=3),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_board_arithmetic(facts)
    assert issues == []


# ===========================================================================
# SECTION 4 — Liquidation preference cross-clause checks
# ===========================================================================


def test_liquidation_single_fact_no_conflict():
    """With only one liquidation fact, no contradiction is possible."""
    facts = ExtractedFacts(doc_id="d", liquidation_preference=_lp_fact("c1", 2.0))
    checker = CrossClauseConsistencyChecker()
    assert checker.check_liquidation_conflicts(facts) == []


def test_liquidation_matching_custom_facts_no_conflict():
    """Two matching multiples (2.0 and 2.0) — no conflict."""
    facts = ExtractedFacts(
        doc_id="d",
        liquidation_preference=_lp_fact("c1", 2.0),
        custom_facts={"liquidation_multiple_c2": 2.0},
    )
    checker = CrossClauseConsistencyChecker()
    assert checker.check_liquidation_conflicts(facts) == []


def test_liquidation_conflicting_custom_facts_raises_issue():
    """Primary fact says 2.0x; custom fact says 3.0x — contradiction."""
    facts = ExtractedFacts(
        doc_id="d",
        liquidation_preference=_lp_fact("c1", 2.0),
        custom_facts={"liquidation_multiple_c2": 3.0},
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_liquidation_conflicts(facts)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "liquidation_multiple_conflict"
    assert issue.is_contradiction is True
    assert "c1" in issue.clause_ids or "c2" in issue.clause_ids
    vals = set(issue.values.values())
    assert 2.0 in vals and 3.0 in vals


def test_liquidation_ungrounded_primary_fact_ignored():
    """Ungrounded primary fact is not considered for conflict detection."""
    facts = ExtractedFacts(
        doc_id="d",
        liquidation_preference=_lp_fact("c1", 2.0, grounded=False),
        custom_facts={"liquidation_multiple_c2": 3.0},
    )
    checker = CrossClauseConsistencyChecker()
    # Only c2 survives — only 1 data point, no conflict
    issues = checker.check_liquidation_conflicts(facts)
    assert issues == []


def test_liquidation_no_facts_no_issues():
    """No liquidation facts at all — empty result."""
    checker = CrossClauseConsistencyChecker()
    assert checker.check_liquidation_conflicts(_empty_facts()) == []


def test_liquidation_multiple_conflicts_detected():
    """Three clauses with three different multiples — two pairs detected."""
    facts = ExtractedFacts(
        doc_id="d",
        liquidation_preference=_lp_fact("c1", 1.0),
        custom_facts={
            "liquidation_multiple_c2": 2.0,
            "liquidation_multiple_c3": 3.0,
        },
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_liquidation_conflicts(facts)
    # 3 pairs: (c1,c2), (c1,c3), (c2,c3) — all different
    assert len(issues) == 3


# ===========================================================================
# SECTION 5 — Vesting cross-clause checks
# ===========================================================================


def test_vesting_single_fact_no_conflict():
    """Only one vesting fact — no conflict possible."""
    facts = ExtractedFacts(doc_id="d", founder_vesting=_vesting_fact("c1", 4.0, 12))
    checker = CrossClauseConsistencyChecker()
    assert checker.check_vesting_conflicts(facts) == []


def test_vesting_matching_years_no_conflict():
    """Two vesting facts with identical durations — no conflict."""
    facts = ExtractedFacts(
        doc_id="d",
        founder_vesting=_vesting_fact("c1", 4.0, 12),
        custom_facts={"vesting_years_c2": 4.0},
    )
    checker = CrossClauseConsistencyChecker()
    assert checker.check_vesting_conflicts(facts) == []


def test_vesting_conflicting_years_raises_issue():
    """Primary says 4 years; custom says 3 years — conflict."""
    facts = ExtractedFacts(
        doc_id="d",
        founder_vesting=_vesting_fact("c1", 4.0, 12),
        custom_facts={"vesting_years_c2": 3.0},
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_vesting_conflicts(facts)
    assert len(issues) >= 1
    types = {i.issue_type for i in issues}
    assert "vesting_years_conflict" in types


def test_vesting_conflicting_cliff_raises_separate_issue():
    """Two cliff periods that differ — cliff conflict raised."""
    facts = ExtractedFacts(
        doc_id="d",
        founder_vesting=_vesting_fact("c1", 4.0, 12),
        custom_facts={
            "vesting_years_c2": 4.0,  # same years — no years conflict
            "vesting_cliff_c2": 6,  # but different cliff
        },
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_vesting_conflicts(facts)
    assert len(issues) == 1
    assert issues[0].issue_type == "vesting_cliff_conflict"
    assert issues[0].values["cliff_a"] == 12 or issues[0].values["cliff_b"] == 12
    assert issues[0].values["cliff_a"] == 6 or issues[0].values["cliff_b"] == 6


def test_vesting_missing_values_not_contradiction():
    """No vesting facts at all — empty list, never a contradiction."""
    checker = CrossClauseConsistencyChecker()
    assert checker.check_vesting_conflicts(_empty_facts()) == []


def test_vesting_ungrounded_fact_ignored():
    """Ungrounded vesting fact is excluded from conflict detection."""
    facts = ExtractedFacts(
        doc_id="d",
        founder_vesting=_vesting_fact("c1", 4.0, grounded=False),
        custom_facts={"vesting_years_c2": 3.0},
    )
    checker = CrossClauseConsistencyChecker()
    # Only c2 survives; only 1 data point, no conflict
    issues = checker.check_vesting_conflicts(facts)
    assert issues == []


# ===========================================================================
# SECTION 6 — check_all aggregation
# ===========================================================================


def test_check_all_no_issues_on_clean_facts():
    """check_all returns empty list when all facts are consistent."""
    facts = ExtractedFacts(
        doc_id="d",
        liquidation_preference=_lp_fact("c1", 1.0),
        board_composition=_board_fact("c_b", founder=2, investor=2, independent=1, total=5),
        founder_vesting=_vesting_fact("c_v", 4.0, 12),
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_all(facts)
    assert issues == []


def test_check_all_multiple_violations_detected():
    """check_all aggregates board arithmetic + liquidation conflict."""
    facts = ExtractedFacts(
        doc_id="d",
        # Board arithmetic violation: 2+2+1=5 but total=7
        board_composition=_board_fact("c_b", founder=2, investor=2, independent=1, total=7),
        # Liquidation conflict: primary=1.0, custom=2.0
        liquidation_preference=_lp_fact("c_lp", 1.0),
        custom_facts={"liquidation_multiple_c_lp2": 2.0},
    )
    checker = CrossClauseConsistencyChecker()
    issues = checker.check_all(facts)
    types = {i.issue_type for i in issues}
    assert "board_arithmetic" in types
    assert "liquidation_multiple_conflict" in types
    assert len(issues) >= 2


# ===========================================================================
# SECTION 7 — Schema limitations are documented
# ===========================================================================


def test_schema_limitations_are_documented():
    """SCHEMA_LIMITATIONS dict is non-empty and covers key limitation areas."""
    assert "single_liquidation_fact" in SCHEMA_LIMITATIONS
    assert "single_vesting_fact" in SCHEMA_LIMITATIONS
    assert "no_security_discriminator" in SCHEMA_LIMITATIONS
    for _key, val in SCHEMA_LIMITATIONS.items():
        assert isinstance(val, str) and len(val) > 10


# ===========================================================================
# SECTION 8 — Production extractor selection (Deliverable 1)
# ===========================================================================


def test_runner_production_uses_member4_extractor():
    """The production runner's default extractor must be Member 4's Gemini-powered FactExtractor.

    This test verifies identity by class name and module path WITHOUT
    actually instantiating the production pipeline (which would load SBERT).
    It inspects what the runner *would* import by importing and comparing
    the class directly.
    """
    # Member 4's authoritative extractor
    # Inspect the runner source rather than constructing the production pipeline,
    # which would load the optional SBERT dependency.
    import inspect

    from termsheet_ai.extract.extractor import FactExtractor as M4FactExtractor
    from termsheet_ai.pipeline.runner import TermSheetPipeline

    runner_source = inspect.getsource(TermSheetPipeline.__init__)
    assert "from termsheet_ai.extract.extractor import FactExtractor" in runner_source
    assert M4FactExtractor.__module__ == "termsheet_ai.extract.extractor"
    assert hasattr(M4FactExtractor, "with_gemini")
