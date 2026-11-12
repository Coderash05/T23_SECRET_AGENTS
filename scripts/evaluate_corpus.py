#!/usr/bin/env python3
"""Validate or evaluate the TermSheet AI corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from termsheet_ai.evaluation import CorpusEvaluationError, evaluate_corpus, validate_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("corpus/manifest.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Write the machine-readable report to this path"
    )
    args = parser.parse_args()
    try:
        report = (
            evaluate_corpus(args.manifest, args.predictions)
            if args.predictions
            else validate_corpus(args.manifest)
        )
    except CorpusEvaluationError as exc:
        parser.error(str(exc))
    payload = json.dumps(report.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(report.summary())
    if report.missing_files or report.missing_annotations or report.invalid_spans:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
