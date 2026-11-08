"""Input Guardrails and Domain Validation for TermSheet AI Pipeline.

Enforces Week 5 robustness requirements:
- Validates file presence, format, and content readability.
- Detects non-term-sheet / out-of-domain documents prior to reasoning.
- Checks minimum text volume and clause segmentation sanity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from termsheet_ai.schemas.contracts import ClauseSpan

# Core domain keyword markers indicating venture financing / term sheet content
TERM_SHEET_KEYWORDS = {
    "preferred",
    "liquidation",
    "anti-dilution",
    "valuation",
    "pre-money",
    "post-money",
    "investor",
    "shares",
    "dividends",
    "conversion",
    "board of directors",
    "protective provisions",
    "voting rights",
    "vesting",
    "drag-along",
    "rofr",
    "first refusal",
    "co-sale",
    "option pool",
    "redemption",
}


class InputGuardrail:
    """Pre-flight validator for raw documents entering the pipeline."""

    def __init__(
        self,
        min_file_size_bytes: int = 10,
        min_clause_count: int = 1,
        min_domain_keyword_hits: int = 2,
    ):
        self.min_file_size_bytes = min_file_size_bytes
        self.min_clause_count = min_clause_count
        self.min_domain_keyword_hits = min_domain_keyword_hits

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Verifies file existence and non-zero size."""
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {path}"
        if not path.is_file():
            return False, f"Path is not a regular file: {path}"
        try:
            size = path.stat().st_size
            if size < self.min_file_size_bytes:
                return False, f"File is empty or below minimum size ({size} bytes): {path.name}"
        except OSError as e:
            return False, f"Unable to read file metadata for {path.name}: {e}"

        return True, None

    def assess_domain_relevance(self, clauses: List[ClauseSpan]) -> Tuple[bool, int, List[str]]:
        """Evaluates whether segmented clauses contain sufficient venture term sheet markers."""
        if not clauses:
            return False, 0, []

        corpus_text = " ".join(c.raw_text.lower() for c in clauses)
        detected_keywords: List[str] = []

        for kw in TERM_SHEET_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", corpus_text):
                detected_keywords.append(kw)

        hit_count = len(detected_keywords)
        is_relevant = hit_count >= self.min_domain_keyword_hits
        return is_relevant, hit_count, detected_keywords

    def validate_ingested_clauses(self, clauses: List[ClauseSpan]) -> Tuple[bool, List[str]]:
        """Validates clause counts and basic readability."""
        warnings: List[str] = []

        if len(clauses) < self.min_clause_count:
            return False, ["Ingestion yielded 0 clauses. Document may be blank or unreadable."]

        is_relevant, hit_count, hits = self.assess_domain_relevance(clauses)
        if not is_relevant:
            warnings.append(
                f"Out-of-domain warning: document contains only {hit_count} venture keyword marker(s) {hits}. "
                "Document may not be a venture financing term sheet."
            )

        return True, warnings
