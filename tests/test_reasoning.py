"""Comprehensive test suite for the symbolic Horn-clause Forward Chaining Engine
and Rule Base Unit Tests across all 31 YAML rules.
"""

import pytest

from termsheet_ai.reasoning.engine import (
    ForwardChainingEngine,
    HornRule,
    WorkingMemory,
    safe_eval_condition,
)
from termsheet_ai.schemas.contracts import (
    AntiDilutionFact,
    AntiDilutionType,
    BoardCompositionFact,
    ExtractedFacts,
    LiquidationPreferenceFact,
    ParticipationType,
    ProtectiveProvisionsFact,
    SeverityLevel,
    SourceSpan,
    VestingFact,
)
from termsheet_ai.ui.trace import all_trace_records, build_trace_tree, format_trace_tree


def test_horn_rule_creation():
    rule = HornRule(
        rule_id="TEST_01",
        family="liquidation_preference",
        condition="multiple > 1.0",
        severity=SeverityLevel.HIGH,
        message="Non-standard liquidation multiple",
        source="NVCA Model Term Sheet",
        conclusion="HighLiquidationMultiple",
    )
    assert rule.rule_id == "TEST_01"
    assert rule.family == "liquidation_preference"
    assert rule.severity == SeverityLevel.HIGH
    assert rule.conclusion == "HighLiquidationMultiple"


def test_safe_eval_condition():
    # True condition
    matched, premises = safe_eval_condition(
        "multiple > 1.0 and participation == True",
        {"multiple": 2.0, "participation": True, "cap": None},
    )
    assert matched is True
    assert premises == {"multiple": 2.0, "participation": True}

    # False condition
    matched, premises = safe_eval_condition(
        "multiple > 2.0",
        {"multiple": 1.5},
    )
    assert matched is False

    # Handling None / null
    matched, premises = safe_eval_condition(
        "participation == True and cap == None",
        {"participation": True, "cap": None},
    )
    assert matched is True


def test_working_memory_assertion():
    wm = WorkingMemory()
    wm.assert_fact("multiple", 2.0, clause_id="c_1")
    wm.assert_fact("liquidation_preference.multiple", 2.0, clause_id="c_1")

    assert wm.get("multiple") == 2.0
    assert wm.get_clause_for_key("multiple") == "c_1"
    assert wm.get_clause_for_key("liquidation_preference.multiple") == "c_1"

    context = wm.to_context(family="liquidation_preference")
    assert context["multiple"] == 2.0


def test_forward_chaining_on_facts():
    engine = ForwardChainingEngine()

    facts = ExtractedFacts(
        doc_id="test_doc_001",
        liquidation_preference=LiquidationPreferenceFact(
            clause_id="c_1",
            multiple=2.0,
            participation=ParticipationType.FULLY_PARTICIPATING,
            cap_multiple=None,
            source_span=SourceSpan(text="2.0x fully participating", page_number=1),
        ),
        anti_dilution=AntiDilutionFact(
            clause_id="c_2",
            anti_dilution_type=AntiDilutionType.FULL_RATCHET,
            source_span=SourceSpan(text="full-ratchet", page_number=1),
        ),
        board_composition=BoardCompositionFact(
            clause_id="c_3",
            founder_seats=1,
            investor_seats=3,
            independent_seats=0,
            total_seats=4,
            investor_majority=True,
            source_span=SourceSpan(text="1 founder, 3 investors", page_number=1),
        ),
        founder_vesting=VestingFact(
            clause_id="c_4",
            total_vesting_years=3.0,
            cliff_months=18,
            acceleration_type="none",
            source_span=SourceSpan(text="3 years with 18 months cliff", page_number=1),
        ),
    )

    trace = engine.reason(facts)

    assert trace.doc_id == "test_doc_001"
    assert len(trace.flags) >= 4

    flag_rule_ids = {f.flag_id.replace("flag_", "") for f in trace.flags}

    assert "R-LP-01" in flag_rule_ids
    assert "R-LP-02" in flag_rule_ids
    assert "R-LP-05" in flag_rule_ids
    assert "R-AD-01" in flag_rule_ids
    assert "R-CT-02" in flag_rule_ids
    assert "R-VS-01" in flag_rule_ids
    assert "R-VS-02" in flag_rule_ids

    critical_flags = [f for f in trace.flags if f.severity == SeverityLevel.CRITICAL]
    critical_ids = {f.flag_id.replace("flag_", "") for f in critical_flags}
    assert "R-AD-01" in critical_ids
    assert "R-LP-05" in critical_ids

    for flag in trace.flags:
        assert flag.justification is not None
        assert flag.justification.market_norm_cite != ""
        assert len(flag.justification.premise_facts) > 0

    assert trace.risk_score.overall_score > 0
    assert trace.risk_score.critical_count == len(critical_flags)


def test_multi_step_chained_inference_r_ct_02_to_r_ct_05():
    """Verifies that R-CT-02 derives InvestorControl, firing R-CT-05 and building a depth-2 proof tree."""
    engine = ForwardChainingEngine()

    facts = ExtractedFacts(
        doc_id="doc_chained",
        board_composition=BoardCompositionFact(
            clause_id="c_board",
            founder_seats=1,
            investor_seats=3,
            independent_seats=1,
            total_seats=5,
            investor_majority=True,
            source_span=SourceSpan(
                text="3 investor seats, 1 founder seat, 1 independent", page_number=1
            ),
        ),
        founder_vesting=VestingFact(
            clause_id="c_vesting",
            total_vesting_years=4.0,
            cliff_months=18,
            acceleration_type="none",
            source_span=SourceSpan(
                text="All Founder shares will be re-vested over 48 months with an 18-month cliff.",
                page_number=2,
            ),
        ),
    )

    trace = engine.reason(facts)
    records = all_trace_records(trace)
    rule_ids = {r.rule_id for r in records}

    assert "R-CT-02" in rule_ids
    assert "R-CT-05" in rule_ids

    # Verify R-CT-05 is attached to the vesting clause and unrolls to depth 2 via R-CT-02
    flag_ct05 = next(f for f in trace.flags if f.justification.rule_id == "R-CT-05")
    assert flag_ct05.clause_id == "c_vesting"
    assert flag_ct05.severity == SeverityLevel.HIGH

    tree = build_trace_tree(flag_ct05.justification, records)
    assert tree.depth() == 2
    assert len(tree.children) == 1
    assert tree.children[0].rule_id == "R-CT-02"
    assert tree.ground_facts == {"VestingReset": True}
    rendered = "\n".join(format_trace_tree(tree))
    assert "derived by R-CT-02" in rendered


def test_r_ct_05_does_not_fire_without_investor_control():
    """Verifies R-CT-05 does not fire when founders control the board even if vesting resets."""
    engine = ForwardChainingEngine()
    facts = ExtractedFacts(
        doc_id="doc_no_chain",
        board_composition=BoardCompositionFact(
            clause_id="c_board",
            founder_seats=2,
            investor_seats=1,
            independent_seats=0,
            total_seats=3,
            investor_majority=False,
            source_span=SourceSpan(text="2 founder seats, 1 investor seat", page_number=1),
        ),
        founder_vesting=VestingFact(
            clause_id="c_vesting",
            total_vesting_years=4.0,
            cliff_months=12,
            acceleration_type="double_trigger",
            source_span=SourceSpan(text="Founder shares re-vested over 48 months", page_number=1),
        ),
    )
    trace = engine.reason(facts)
    rule_ids = {r.rule_id for r in trace.rule_traces}
    assert "R-CT-02" not in rule_ids
    assert "R-CT-05" not in rule_ids


@pytest.mark.parametrize(
    "rule_id,positive_facts,negative_facts,expected_severity",
    [
        (
            "R-LP-01",
            {"multiple": 1.5},
            {"multiple": 1.0},
            SeverityLevel.HIGH,
        ),
        (
            "R-LP-02",
            {"participation": True},
            {"participation": False},
            SeverityLevel.HIGH,
        ),
        (
            "R-LP-03",
            {"seniority": "senior"},
            {"seniority": "pari_passu"},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-LP-04",
            {"multiple": 3.0},
            {"multiple": 2.0},
            SeverityLevel.CRITICAL,
        ),
        (
            "R-LP-05",
            {"participation": True, "cap": None},
            {"participation": True, "cap": 3.0},
            SeverityLevel.CRITICAL,
        ),
        (
            "R-AD-01",
            {"type": "full_ratchet"},
            {"type": "broad_based_weighted_average"},
            SeverityLevel.CRITICAL,
        ),
        (
            "R-AD-02",
            {"type": "broad_based_weighted_average"},
            {"type": "full_ratchet"},
            SeverityLevel.LOW,
        ),
        (
            "R-AD-03",
            {"type": "narrow_based_weighted_average"},
            {"type": "broad_based_weighted_average"},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-CT-02",
            {"investor_seats": 3, "founder_seats": 1},
            {"investor_seats": 1, "founder_seats": 2},
            SeverityLevel.HIGH,
        ),
        (
            "R-CT-03",
            {"independent_seats": 0},
            {"independent_seats": 1},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-CT-04",
            {"founder_seats": 1, "total_seats": 5},
            {"founder_seats": 2, "total_seats": 5},
            SeverityLevel.HIGH,
        ),
        (
            "R-CT-05",
            {"InvestorControl": True, "VestingReset": True},
            {"InvestorControl": True, "VestingReset": False},
            SeverityLevel.HIGH,
        ),
        (
            "R-VS-01",
            {"vesting_months": 36},
            {"vesting_months": 48},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-VS-02",
            {"cliff_months": 18},
            {"cliff_months": 12},
            SeverityLevel.HIGH,
        ),
        (
            "R-VS-03",
            {"acceleration": "none"},
            {"acceleration": "double_trigger"},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-VS-04",
            {"acceleration": "double_trigger"},
            {"acceleration": "none"},
            SeverityLevel.LOW,
        ),
        (
            "R-PP-01",
            {"veto_day_to_day_operations": True},
            {"veto_day_to_day_operations": False},
            SeverityLevel.HIGH,
        ),
        (
            "R-PP-02",
            {"requires_series_majority": True},
            {"requires_series_majority": False},
            SeverityLevel.INFO,
        ),
        (
            "R-DA-01",
            {"approval_threshold": 40},
            {"approval_threshold": 50},
            SeverityLevel.HIGH,
        ),
        (
            "R-DA-02",
            {"requires_founder_consent": False, "approval_threshold": 55},
            {"requires_founder_consent": True, "approval_threshold": 55},
            SeverityLevel.HIGH,
        ),
        (
            "R-OP-01",
            {"pool_percentage": 25},
            {"pool_percentage": 15},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-OP-02",
            {"pre_money_pool": True, "pool_percentage": 20},
            {"pre_money_pool": False, "pool_percentage": 20},
            SeverityLevel.HIGH,
        ),
        (
            "R-RD-01",
            {"redemption_years": 3},
            {"redemption_years": 5},
            SeverityLevel.HIGH,
        ),
        (
            "R-RD-02",
            {"interest_rate": 12},
            {"interest_rate": 8},
            SeverityLevel.HIGH,
        ),
        (
            "R-EX-01",
            {"duration_days": 60},
            {"duration_days": 45},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-DV-01",
            {"cumulative": True},
            {"cumulative": False},
            SeverityLevel.HIGH,
        ),
        (
            "R-RO-01",
            {"investor_co_sale": True},
            {"investor_co_sale": False},
            SeverityLevel.INFO,
        ),
        (
            "R-RO-02",
            {"full_founder_transfer_lockup": True},
            {"full_founder_transfer_lockup": False},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-IR-01",
            {"has_major_investor_threshold": False},
            {"has_major_investor_threshold": True},
            SeverityLevel.MEDIUM,
        ),
        (
            "R-IR-02",
            {"monthly_reporting_required": True},
            {"monthly_reporting_required": False},
            SeverityLevel.LOW,
        ),
        (
            "R-CV-01",
            {"automatic_conversion_ipo_threshold": 150},
            {"automatic_conversion_ipo_threshold": 50},
            SeverityLevel.MEDIUM,
        ),
    ],
)
def test_individual_rule_unit_triggers_and_boundaries(
    rule_id, positive_facts, negative_facts, expected_severity
):
    """Unit test every single rule in rules_v1.yaml on positive and negative boundary cases."""
    engine = ForwardChainingEngine()

    # Positive case must trigger the rule with expected severity
    wm_pos = WorkingMemory()
    for k, v in positive_facts.items():
        wm_pos.assert_fact(k, v, clause_id="c_test")
    trace_pos = engine.infer(wm_pos, doc_id=f"pos_{rule_id}")
    pos_flags = {f.justification.rule_id: f for f in trace_pos.flags}
    assert rule_id in pos_flags, f"Expected {rule_id} to trigger on {positive_facts}"
    assert pos_flags[rule_id].severity == expected_severity

    # Negative boundary case must NOT trigger the rule
    wm_neg = WorkingMemory()
    for k, v in negative_facts.items():
        wm_neg.assert_fact(k, v, clause_id="c_test")
    trace_neg = engine.infer(wm_neg, doc_id=f"neg_{rule_id}")
    neg_rule_ids = {f.justification.rule_id for f in trace_neg.flags}
    assert rule_id not in neg_rule_ids, f"Expected {rule_id} NOT to trigger on {negative_facts}"


def test_protective_provisions_working_memory_grounding():
    engine = ForwardChainingEngine()
    facts = ExtractedFacts(
        doc_id="doc_pp",
        protective_provisions=ProtectiveProvisionsFact(
            clause_id="c_pp",
            requires_series_majority=True,
            covered_actions=["day-to-day operational budget approval", "sale of company"],
            source_span=SourceSpan(
                text="Preferred majority must approve day-to-day operational decisions.",
                page_number=1,
            ),
        ),
    )
    trace = engine.reason(facts)
    fired = {r.rule_id for r in trace.rule_traces}
    assert "R-PP-01" in fired
    assert "R-PP-02" in fired
