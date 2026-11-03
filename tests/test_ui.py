"""Tests for the Streamlit UI layer: mock data, trace tree, view-models and a smoke run."""

from pathlib import Path

import pytest

from termsheet_ai.schemas.contracts import SeverityLevel
from termsheet_ai.ui.data_source import load_mock_payload, run_pipeline_on_upload
from termsheet_ai.ui.theme import SEVERITY_STYLE, sev_rank
from termsheet_ai.ui.trace import all_trace_records, build_trace_tree, format_trace_tree
from termsheet_ai.ui.viewmodel import build_clause_rows, sorted_flags

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def payload():
    return load_mock_payload()


def test_mock_payload_is_consistent(payload):
    clause_ids = {c.clause_id for c in payload.clauses}
    assert payload.metadata.total_clauses == len(payload.clauses)
    assert all(f.clause_id in clause_ids for f in payload.audit_trace.flags)
    assert {s.clause_id for s in payload.summaries} <= clause_ids

    score = payload.audit_trace.risk_score
    flags = payload.audit_trace.flags
    for sev in SeverityLevel:
        expected = sum(1 for f in flags if f.severity == sev)
        assert getattr(score, f"{sev.value}_count") == expected


def test_mock_source_spans_appear_in_their_clause(payload):
    by_id = {c.clause_id: c for c in payload.clauses}
    for name in (
        "liquidation_preference",
        "anti_dilution",
        "board_composition",
        "founder_vesting",
        "protective_provisions",
    ):
        fact = getattr(payload.facts, name)
        assert fact.source_span.text in by_id[fact.clause_id].raw_text


def test_theme_covers_every_severity():
    assert {s.value for s in SeverityLevel} == set(SEVERITY_STYLE)
    assert sev_rank("critical") < sev_rank("high") < sev_rank("info")


def test_chained_trace_nests_investor_control(payload):
    records = all_trace_records(payload.audit_trace)
    chain = next(r for r in records if r.rule_id == "R-CT-05")
    tree = build_trace_tree(chain, records)
    assert tree.depth() == 2
    assert tree.children[0].rule_id == "R-CT-02"
    assert tree.ground_facts == {"VestingReset": True}
    text = "\n".join(format_trace_tree(tree))
    assert "derived by R-CT-02" in text


def test_trace_tree_survives_circular_rules(payload):
    records = all_trace_records(payload.audit_trace)
    a, b = records[0].model_copy(), records[1].model_copy()
    a.conclusion_flag, a.premise_facts = "A", {"B": 1}
    b.conclusion_flag, b.premise_facts = "B", {"A": 1}
    assert build_trace_tree(a, [a, b]).depth() == 2


def test_viewmodel_orders_and_groups(payload):
    flags = sorted_flags(payload)
    assert flags[0].severity == SeverityLevel.CRITICAL
    ranks = [sev_rank(f.severity) for f in flags]
    assert ranks == sorted(ranks)

    rows = {r.clause.clause_id: r for r in build_clause_rows(payload)}
    assert rows["c_4"].worst_severity == "high" and len(rows["c_4"].flags) == 2
    assert rows["c_5"].worst_severity is None
    assert rows["c_8"].clause.is_unclassified


def test_live_pipeline_output_can_be_shaped_for_display(tmp_path):
    payload = run_pipeline_on_upload(
        "sample.txt",
        b"1. Liquidation Preference: 2.0x liquidation preference, participating fully.\n"
        b"2. Anti-Dilution: full-ratchet anti-dilution.\n",
    )
    assert payload.metadata.filename == "sample.txt"
    assert len(build_clause_rows(payload)) == len(payload.clauses)
    assert sorted_flags(payload)


def test_streamlit_app_smoke_run():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 5
