"""Deterministic evaluation and validation for the term-sheet corpus."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


class CorpusEvaluationError(ValueError):
    """Raised when corpus metadata, gold labels, or predictions are invalid."""


@dataclass
class EvaluationReport:
    """Machine-readable corpus evaluation result."""

    documents: int = 0
    evaluated_documents: int = 0
    gold_clauses: int = 0
    predicted_clauses: int = 0
    missing_annotations: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    invalid_spans: list[str] = field(default_factory=list)
    duplicate_clause_ids: list[str] = field(default_factory=list)
    family_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    flag_metrics: dict[str, int] = field(
        default_factory=lambda: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)

    def summary(self) -> str:
        """Return a concise human-readable summary."""
        return (
            f"Documents: {self.evaluated_documents}/{self.documents}; "
            f"gold clauses: {self.gold_clauses}; predicted clauses: {self.predicted_clauses}; "
            f"missing annotations: {len(self.missing_annotations)}; "
            f"invalid spans: {len(self.invalid_spans)}; "
            f"missing files: {len(self.missing_files)}"
        )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a corpus manifest."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusEvaluationError(f"Unable to load manifest {path}: {exc}") from exc
    try:
        documents = data["evaluation_corpus"]["documents"]
    except KeyError as exc:
        raise CorpusEvaluationError("Manifest is missing evaluation_corpus.documents") from exc
    if not isinstance(documents, list):
        raise CorpusEvaluationError("Manifest evaluation_corpus.documents must be a list")
    return data


def _gold_path(root: Path, entry: dict[str, Any]) -> Path:
    filename = entry.get("filename", "")
    if not filename:
        return root / "gold" / f"{entry['id']}.gold.json"
    if entry.get("is_synthetic"):
        return root / Path(filename).with_suffix(".gold.json")
    return root / "gold" / f"{Path(filename).name}.gold.json"


def _document_path(root: Path, entry: dict[str, Any]) -> Path:
    filename = entry.get("filename", "")
    return root / filename if filename else root / "__missing_filename__"


def _clauses(gold: dict[str, Any]) -> list[dict[str, Any]]:
    clauses = gold.get("clauses")
    if clauses is None:
        return []
    if not isinstance(clauses, list):
        raise CorpusEvaluationError("Gold clauses must be a list")
    return clauses


def _validate_clauses(
    doc_id: str, clauses: Iterable[dict[str, Any]], report: EvaluationReport
) -> None:
    seen: set[str] = set()
    for clause in clauses:
        clause_id = clause.get("clause_id")
        if not clause_id:
            raise CorpusEvaluationError(f"{doc_id}: clause is missing clause_id")
        if clause_id in seen:
            report.duplicate_clause_ids.append(f"{doc_id}:{clause_id}")
        seen.add(clause_id)
        span = clause.get("source_span")
        text = clause.get("clause_text", clause.get("raw_text", ""))
        if span is not None and (not isinstance(span, str) or span not in text):
            report.invalid_spans.append(f"{doc_id}:{clause_id}")


def validate_corpus(manifest_path: Path) -> EvaluationReport:
    """Validate manifest paths, gold sidecars, duplicate IDs, and literal spans."""
    root = manifest_path.parent
    manifest = load_manifest(manifest_path)
    entries = manifest["evaluation_corpus"]["documents"]
    report = EvaluationReport(documents=len(entries))

    for entry in entries:
        doc_id = entry.get("id", "<missing-id>")
        document_path = _document_path(root, entry)
        gold_path = _gold_path(root, entry)
        if not document_path.is_file():
            report.missing_files.append(f"{doc_id}:document:{document_path}")
        if not gold_path.is_file():
            report.missing_annotations.append(f"{doc_id}:gold:{gold_path}")
            continue
        try:
            gold = json.loads(gold_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusEvaluationError(f"{doc_id}: invalid gold file {gold_path}: {exc}") from exc
        clauses = _clauses(gold)
        _validate_clauses(doc_id, clauses, report)
        report.gold_clauses += len(clauses)
        report.evaluated_documents += 1
    return report


def _metric(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _prediction_map(predictions: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for document in predictions.get("documents", []):
        doc_id = document.get("doc_id")
        if not doc_id:
            raise CorpusEvaluationError("Prediction document is missing doc_id")
        result[doc_id] = {}
        for clause in document.get("clauses", []):
            clause_id = clause.get("clause_id")
            if not clause_id:
                raise CorpusEvaluationError(f"{doc_id}: prediction is missing clause_id")
            if clause_id in result[doc_id]:
                raise CorpusEvaluationError(f"{doc_id}:{clause_id}: duplicate prediction")
            result[doc_id][clause_id] = clause
    return result


def evaluate_corpus(manifest_path: Path, predictions_path: Path) -> EvaluationReport:
    """Compare clause-family and expected-flag predictions with gold labels."""
    report = validate_corpus(manifest_path)
    try:
        predictions = json.loads(predictions_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusEvaluationError(f"Unable to load predictions {predictions_path}: {exc}") from exc
    predicted = _prediction_map(predictions)
    root = manifest_path.parent
    family_counts: Counter[tuple[str, str]] = Counter()
    flag_tp = flag_fp = flag_fn = 0

    for entry in load_manifest(manifest_path)["evaluation_corpus"]["documents"]:
        doc_id = entry["id"]
        gold_path = _gold_path(root, entry)
        if not gold_path.is_file():
            continue
        gold = json.loads(gold_path.read_text())
        gold_clauses = _clauses(gold)
        predictions_for_doc = predicted.get(doc_id, {})
        report.predicted_clauses += len(predictions_for_doc)
        for clause in gold_clauses:
            clause_id = clause["clause_id"]
            gold_family = clause.get("clause_family", "other")
            prediction = predictions_for_doc.get(clause_id)
            predicted_family = prediction.get("clause_family", "other") if prediction else "other"
            family_counts[(gold_family, predicted_family)] += 1
            gold_flags = set(clause.get("expected_flags", []))
            predicted_flags = set(prediction.get("expected_flags", [])) if prediction else set()
            flag_tp += len(gold_flags & predicted_flags)
            flag_fp += len(predicted_flags - gold_flags)
            flag_fn += len(gold_flags - predicted_flags)

    labels = sorted({label for pair in family_counts for label in pair})
    for label in labels:
        tp = family_counts[(label, label)]
        fp = sum(value for (gold, predicted_label), value in family_counts.items() if predicted_label == label and gold != label)
        fn = sum(value for (gold, predicted_label), value in family_counts.items() if gold == label and predicted_label != label)
        report.family_metrics[label] = _metric(tp, fp, fn)
    report.confusion_matrix = {
        gold: {predicted_label: family_counts[(gold, predicted_label)] for predicted_label in labels}
        for gold in labels
    }
    report.flag_metrics = {
        "true_positive": flag_tp,
        "false_positive": flag_fp,
        "false_negative": flag_fn,
    }
    return report
