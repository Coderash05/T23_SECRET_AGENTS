"""Offline tests for the single-prompt LLM-only comparison baseline."""

import importlib.util
import json
from pathlib import Path

import pytest

from termsheet_ai.baseline import LLMOnlyBaseline
from termsheet_ai.schemas.contracts import ClauseFamily

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_llm_baseline.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_llm_baseline", _SCRIPT)
assert _SPEC and _SPEC.loader
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def test_baseline_uses_one_direct_prompt_and_normalizes_output():
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({
            "clause_family": "liquidation_preference",
            "risk_flags": ["R-LP-01", "NOT-A-RULE", "R-LP-01"],
            "summary": "The investor receives a 2x preference.",
        })

    prediction = LLMOnlyBaseline(fake_llm).predict("Investor receives a 2x liquidation preference.")

    assert len(prompts) == 1
    assert "retrieval" in prompts[0].lower()
    assert prediction.clause_family == ClauseFamily.LIQUIDATION_PREFERENCE
    assert prediction.risk_flags == ("R-LP-01",)
    assert prediction.summary


def test_baseline_invalid_response_is_visible_as_empty_other_prediction():
    prediction = LLMOnlyBaseline(lambda _: "not JSON").predict("Board clause")

    assert prediction.clause_family == ClauseFamily.OTHER
    assert prediction.risk_flags == ()
    assert prediction.summary == ""
    assert prediction.raw_response == "not JSON"


def test_baseline_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        LLMOnlyBaseline(lambda _: "{}").predict(" ")


def test_evaluation_reports_family_and_multilabel_metrics():
    responses = iter([
        '{"clause_family":"liquidation_preference","risk_flags":["R-LP-01"],"summary":"x"}',
        '{"clause_family":"board_composition","risk_flags":["R-CT-02"],"summary":"y"}',
    ])
    records = [
        {"id": "a", "text": "x", "gold_family": "liquidation_preference", "gold_flags": ["R-LP-01"]},
        {"id": "b", "text": "y", "gold_family": "anti_dilution", "gold_flags": ["R-AD-01"]},
    ]

    metrics, output = evaluation.evaluate(records, LLMOnlyBaseline(lambda _: next(responses)))

    assert metrics == {
        "records": 2,
        "family_accuracy": 0.5,
        "risk_flag_micro": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    }
    assert [row["id"] for row in output] == ["a", "b"]
