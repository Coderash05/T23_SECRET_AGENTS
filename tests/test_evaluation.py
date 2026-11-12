import json
from pathlib import Path

import pytest

from termsheet_ai.evaluation import CorpusEvaluationError, evaluate_corpus, validate_corpus


def _fixture(tmp_path: Path, *, span: str = "1x preference") -> tuple[Path, Path]:
    (tmp_path / "gold").mkdir()
    (tmp_path / "doc.txt").write_text("1x preference")
    (tmp_path / "gold" / "doc.txt.gold.json").write_text(
        json.dumps(
            {
                "doc_id": "eval-001",
                "clauses": [
                    {
                        "clause_id": "c001",
                        "clause_family": "liquidation_preference",
                        "clause_text": "1x preference",
                        "source_span": span,
                        "expected_flags": ["R-LP-01"],
                    }
                ],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "evaluation_corpus": {
                    "documents": [
                        {
                            "id": "eval-001",
                            "filename": "doc.txt",
                            "is_synthetic": False,
                        }
                    ]
                }
            }
        )
    )
    return manifest, tmp_path / "predictions.json"


def test_validate_literal_span(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    report = validate_corpus(manifest)
    assert report.invalid_spans == []
    assert report.gold_clauses == 1


def test_validate_invalid_span(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path, span="missing")
    report = validate_corpus(manifest)
    assert report.invalid_spans == ["eval-001:c001"]


def test_missing_document_and_gold_are_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"evaluation_corpus": {"documents": [{"id": "eval-001", "filename": "missing.txt"}]}})
    )
    report = validate_corpus(manifest)
    assert len(report.missing_files) == 1
    assert len(report.missing_annotations) == 1


def test_duplicate_clause_ids_are_reported(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    gold = tmp_path / "gold" / "doc.txt.gold.json"
    data = json.loads(gold.read_text())
    data["clauses"].append(data["clauses"][0])
    gold.write_text(json.dumps(data))
    report = validate_corpus(manifest)
    assert report.duplicate_clause_ids == ["eval-001:c001"]


def test_evaluate_metrics_and_confusion_matrix(tmp_path: Path) -> None:
    manifest, predictions = _fixture(tmp_path)
    predictions.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "eval-001",
                        "clauses": [
                            {
                                "clause_id": "c001",
                                "clause_family": "liquidation_preference",
                                "expected_flags": ["R-LP-01"],
                            }
                        ],
                    }
                ]
            }
        )
    )
    report = evaluate_corpus(manifest, predictions)
    assert report.family_metrics["liquidation_preference"]["f1"] == 1.0
    assert report.confusion_matrix["liquidation_preference"]["liquidation_preference"] == 1
    assert report.flag_metrics == {"true_positive": 1, "false_positive": 0, "false_negative": 0}


def test_invalid_prediction_shape_is_rejected(tmp_path: Path) -> None:
    manifest, predictions = _fixture(tmp_path)
    predictions.write_text(json.dumps({"documents": [{"clauses": [{}]}]}))
    with pytest.raises(CorpusEvaluationError, match="missing doc_id"):
        evaluate_corpus(manifest, predictions)
