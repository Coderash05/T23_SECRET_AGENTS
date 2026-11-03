"""Comprehensive test suite for the symbolic Horn-clause Forward Chaining Engine."""

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
    SeverityLevel,
    SourceSpan,
    VestingFact,
)


def test_horn_rule_creation():
    rule = HornRule(
        rule_id="TEST_01",
        family="liquidation_preference",
        condition="multiple > 1.0",
        severity=SeverityLevel.HIGH,
        message="Non-standard liquidation multiple",
        source="NVCA Model Term Sheet",
    )
    assert rule.rule_id == "TEST_01"
    assert rule.family == "liquidation_preference"
    assert rule.severity == SeverityLevel.HIGH


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
