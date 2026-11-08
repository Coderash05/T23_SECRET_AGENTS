"""Unit and integration tests for Pipeline Robustness and Input Guardrails."""

from pathlib import Path

from termsheet_ai.pipeline.guardrails import InputGuardrail
from termsheet_ai.pipeline.runner import TermSheetPipeline
from termsheet_ai.schemas.contracts import ClauseSpan


def test_guardrail_missing_file(tmp_path: Path):
    guard = InputGuardrail()
    missing_file = tmp_path / "does_not_exist.txt"
    ok, err = guard.validate_file(missing_file)
    assert ok is False
    assert "File does not exist" in err


def test_guardrail_empty_file(tmp_path: Path):
    guard = InputGuardrail()
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    ok, err = guard.validate_file(empty_file)
    assert ok is False
    assert "below minimum size" in err


def test_guardrail_domain_assessment():
    guard = InputGuardrail(min_domain_keyword_hits=2)

    # In-domain clauses
    in_domain = [
        ClauseSpan(
            clause_id="c_1",
            raw_text="Series A Preferred stock shall have 1.0x non-participating liquidation preference.",
        ),
        ClauseSpan(
            clause_id="c_2",
            raw_text="The board of directors shall consist of three members.",
        ),
    ]
    is_rel, hits, kws = guard.assess_domain_relevance(in_domain)
    assert is_rel is True
    assert hits >= 2
    assert "preferred" in kws or "liquidation" in kws

    # Out-of-domain clauses (e.g., pancake recipe)
    out_of_domain = [
        ClauseSpan(
            clause_id="c_1",
            raw_text="Combine flour, sugar, baking powder, and salt in a large bowl.",
        ),
        ClauseSpan(
            clause_id="c_2",
            raw_text="Heat a lightly oiled griddle or frying pan over medium-high heat.",
        ),
    ]
    is_rel, hits, _ = guard.assess_domain_relevance(out_of_domain)
    assert is_rel is False
    assert hits == 0


def test_pipeline_nonexistent_file_graceful_handling(tmp_path: Path):
    pipeline = TermSheetPipeline.create_mock()
    missing_path = tmp_path / "nonexistent_sheet.txt"

    payload = pipeline.run(missing_path)

    assert payload is not None
    assert len(payload.clauses) == 0
    assert len(payload.execution_errors) > 0
    assert "File does not exist" in payload.execution_errors[0]


def test_pipeline_out_of_domain_warning(tmp_path: Path):
    recipe_file = tmp_path / "recipe.txt"
    recipe_file.write_text(
        "1. Preheat oven to 375 degrees F.\n"
        "2. Mix butter and sugar until creamy.\n"
        "3. Bake for 10 to 12 minutes until golden.\n",
        encoding="utf-8",
    )

    pipeline = TermSheetPipeline.create_mock()
    payload = pipeline.run(recipe_file)

    assert payload is not None
    # Pipeline still runs S1-S5 gracefully but notes out-of-domain warning
    assert any("Out-of-domain warning" in err for err in payload.execution_errors)
