"""Evaluate the LLM-only comparator on a JSONL annotation set.

Each input line must have ``id`` and ``text``.  Add optional ``gold_family`` and
``gold_flags`` fields to score clause-family accuracy and document-level flag
micro precision/recall/F1.  Predictions are written as JSONL for auditability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from termsheet_ai.baseline import LLMOnlyBaseline  # noqa: E402
from termsheet_ai.schemas.contracts import ClauseFamily  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load validated evaluation records from a JSONL file."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"Line {line_number} must contain a string 'text' field")
        records.append(record)
    return records


def flag_metrics(gold_sets: Iterable[set[str]], predicted_sets: Iterable[set[str]]) -> dict[str, float]:
    """Return micro-averaged multi-label precision, recall, and F1."""
    true_positive = false_positive = false_negative = 0
    for gold, predicted in zip(gold_sets, predicted_sets, strict=True):
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate(records: list[dict[str, Any]], baseline: LLMOnlyBaseline) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run predictions and calculate only metrics for labels present in the input."""
    output: list[dict[str, Any]] = []
    family_total = family_correct = 0
    gold_flags: list[set[str]] = []
    predicted_flags: list[set[str]] = []
    for index, record in enumerate(records):
        prediction = baseline.predict(record["text"])
        result = {"id": record.get("id", str(index)), **prediction.to_dict()}
        output.append(result)
        gold_family = record.get("gold_family")
        if gold_family is not None:
            family_total += 1
            try:
                family_correct += prediction.clause_family == ClauseFamily(str(gold_family))
            except ValueError:
                pass
        if "gold_flags" in record:
            raw_gold_flags = record["gold_flags"]
            if not isinstance(raw_gold_flags, list):
                raise ValueError("gold_flags must be a list when supplied")
            gold_flags.append({str(flag) for flag in raw_gold_flags})
            predicted_flags.append(set(prediction.risk_flags))
    metrics: dict[str, Any] = {"records": len(records)}
    if family_total:
        metrics["family_accuracy"] = round(family_correct / family_total, 4)
    if gold_flags:
        metrics["risk_flag_micro"] = flag_metrics(gold_flags, predicted_flags)
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TermSheet AI LLM-only baseline.")
    parser.add_argument("input", type=Path, help="JSONL records containing text and optional gold labels")
    parser.add_argument("--output", type=Path, default=Path("llm_baseline_predictions.jsonl"))
    parser.add_argument("--model", default=None, help="Optional Gemini model override")
    parser.add_argument("--no-cache", action="store_true", help="Disable the existing prompt cache")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    baseline = LLMOnlyBaseline.with_gemini(model=args.model, use_cache=not args.no_cache)
    metrics, predictions = evaluate(records, baseline)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
