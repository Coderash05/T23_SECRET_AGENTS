"""Source span grounding and validation for Stage S3 extraction.

This module validates that extracted facts are strictly grounded in the source text:
1. Verifies that the cited source span text actually exists in the clause.
2. Verifies offsets when provided (preventing offset drift or incorrect indexing).
3. Resolves character offsets when missing, requiring an unambiguous match to avoid
   arbitrary offset assignments when a snippet appears multiple times.
4. Distinguishes exact matches from normalized whitespace matches, ensuring non-whitespace
   characters, case, and punctuation are strictly preserved without partial-word drift.
5. Verifies page-number consistency against the parent clause without discarding
   explicit metadata silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from termsheet_ai.schemas.contracts import ClauseSpan, SourceSpan


@dataclass
class SpanValidationResult:
    """Result of source span validation against a clause.

    Attributes:
        is_valid: True if the span text exists in the clause and offsets match.
        resolved_span: The validated SourceSpan with exact character offsets populated.
                       None if validation failed.
        is_exact: True if the match was verbatim without whitespace normalization.
        match_count: Total occurrences of the candidate citation found in clause.raw_text
                     (exact occurrences if found; or normalized occurrences if matching via
                     whitespace normalization).
        error_message: Detailed diagnostic message if validation failed.
    """

    is_valid: bool
    resolved_span: Optional[SourceSpan] = None
    is_exact: bool = False
    match_count: int = 0
    error_message: Optional[str] = None


def _find_exact_occurrences(raw_text: str, query: str) -> List[Tuple[int, int]]:
    """Find all (start_char, end_char) intervals for exact occurrences of query in raw_text."""
    if not query:
        return []

    occurrences: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = raw_text.find(query, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(query)))
        start = idx + 1
    return occurrences


def _find_normalized_occurrences(raw_text: str, query: str) -> List[Tuple[int, int]]:
    r"""Find (start_char, end_char) in raw_text where whitespace-collapsed text matches query.

    Rules:
    - Splits query into non-whitespace tokens (preserving punctuation and case).
    - Requires boundary awareness: if a token begins or ends with a word character (\w),
      it enforces word boundary (\b) so partial words (e.g. 'part' in 'participating')
      do not match unintentionally.
    - Joins tokens with \s+ so only whitespace differences are permitted between tokens.
    """
    tokens = query.split()
    if not tokens:
        return []

    pattern_parts: List[str] = []
    for i, token in enumerate(tokens):
        escaped = re.escape(token)
        prefix = r"\b" if re.match(r"^\w", token) and i == 0 else ""
        suffix = r"\b" if re.match(r".*\w$", token) and i == len(tokens) - 1 else ""
        pattern_parts.append(f"{prefix}{escaped}{suffix}")

    pattern_str = r"\s+".join(pattern_parts)
    try:
        pattern = re.compile(pattern_str)
    except re.error:
        return []

    matches: List[Tuple[int, int]] = []
    for match in pattern.finditer(raw_text):
        matches.append((match.start(), match.end()))
    return matches


def validate_source_span(
    clause: ClauseSpan,
    span: SourceSpan,
    allow_whitespace_normalization: bool = True,
) -> SpanValidationResult:
    """Validate and ground a SourceSpan against its parent ClauseSpan.

    Validation rules:
    1. Empty or whitespace-only span text is rejected.
    2. Offset validation:
       - Requires both start_char and end_char to be present, or both to be None.
         Incomplete offsets are rejected.
       - Bounds check: requires 0 <= start_char < end_char <= len(clause.raw_text).
       - Negative, reversed, zero-length, or out-of-range offsets are rejected.
       - Exact verification: clause.raw_text[start_char:end_char] must exactly equal span.text.
    3. Missing offsets (None):
       - Exact occurrences are searched first.
       - If exact match occurs exactly once: resolved with offsets and match_count=1.
       - If exact match occurs multiple times: fails with an explicit ambiguity message
         and match_count=len(exact_matches).
       - If not found exactly and allow_whitespace_normalization is True:
         searches normalized occurrences (word-boundary safe, preserving punctuation).
         If uniquely found (len=1): resolves exact slice from raw_text with match_count=1,
         is_exact=False. If multiple (len>1): fails as ambiguous.
    4. Page-number consistency:
       - If span.page_number is missing (None): inherits clause.page_number.
       - If span.page_number is provided and differs from clause.page_number:
         rejected as a cross-page metadata conflict.
    5. Match count:
       - Always reports the total occurrences of the candidate citation found in the clause,
         even when explicit offsets identify one valid occurrence among multiple.

    Args:
        clause: The parent ClauseSpan containing raw_text and page_number.
        span: The SourceSpan citing grounding text, optional offsets, and page_number.
        allow_whitespace_normalization: Whether to permit whitespace-flexible matches.

    Returns:
        SpanValidationResult with validation status, populated span, and diagnostic message.
    """
    raw_text = clause.raw_text
    span_text = span.text

    if not span_text or not span_text.strip():
        return SpanValidationResult(
            is_valid=False,
            error_message="Source span text is empty or contains only whitespace.",
        )

    # Validate page number consistency before offset matching
    target_page = clause.page_number
    if span.page_number is not None and span.page_number != target_page:
        return SpanValidationResult(
            is_valid=False,
            error_message=(
                f"Page number mismatch: span cites page {span.page_number}, "
                f"but clause is on page {clause.page_number}."
            ),
        )

    has_start = span.start_char is not None
    has_end = span.end_char is not None

    # Incomplete offsets (one present, one missing)
    if has_start ^ has_end:
        return SpanValidationResult(
            is_valid=False,
            error_message="Incomplete offsets: both start_char and end_char must be provided or both must be None.",
        )

    # Compute overall occurrences in the clause for consistent match_count reporting
    exact_matches = _find_exact_occurrences(raw_text, span_text)

    # Case A: Explicit offsets provided
    if has_start and has_end:
        start = span.start_char
        end = span.end_char

        # Bounds and validity check: 0 <= start < end <= len(raw_text)
        if start < 0 or end > len(raw_text) or start >= end:
            return SpanValidationResult(
                is_valid=False,
                match_count=len(exact_matches),
                error_message=(
                    f"Invalid offset range [{start}:{end}]: must satisfy "
                    f"0 <= start < end <= {len(raw_text)}."
                ),
            )

        actual_slice = raw_text[start:end]

        # Exact match at given offsets
        if actual_slice == span_text:
            resolved = span.model_copy(
                update={"page_number": target_page}
            )
            return SpanValidationResult(
                is_valid=True,
                resolved_span=resolved,
                is_exact=True,
                match_count=len(exact_matches) if exact_matches else 1,
            )

        # Offsets point to text that does not match span.text
        # Check if the discrepancy is merely whitespace difference to provide clear diagnostics
        actual_norm = " ".join(actual_slice.split())
        query_norm = " ".join(span_text.split())
        if actual_norm == query_norm:
            return SpanValidationResult(
                is_valid=False,
                match_count=len(exact_matches),
                error_message=(
                    f"Provided offsets [{start}:{end}] do not exactly match span text: "
                    f"expected '{span_text!r}', found '{actual_slice!r}' (differ only by whitespace)."
                ),
            )

        return SpanValidationResult(
            is_valid=False,
            match_count=len(exact_matches),
            error_message=(
                f"Provided offsets [{start}:{end}] mismatch: "
                f"found '{actual_slice[:50]!r}...', expected '{span_text[:50]!r}...'."
            ),
        )

    # Case B: Offsets are missing (None) -> Unambiguous auto-resolution
    if len(exact_matches) == 1:
        start, end = exact_matches[0]
        resolved = span.model_copy(
            update={
                "start_char": start,
                "end_char": end,
                "page_number": target_page,
            }
        )
        return SpanValidationResult(
            is_valid=True,
            resolved_span=resolved,
            is_exact=True,
            match_count=1,
        )

    if len(exact_matches) > 1:
        return SpanValidationResult(
            is_valid=False,
            match_count=len(exact_matches),
            error_message=(
                f"Ambiguous match: span text occurs {len(exact_matches)} times in clause. "
                "Explicit offsets must be provided to ground this fact."
            ),
        )

    # Exact match not found; evaluate normalized match if enabled
    if allow_whitespace_normalization:
        norm_matches = _find_normalized_occurrences(raw_text, span_text)

        if len(norm_matches) == 1:
            start, end = norm_matches[0]
            matched_slice = raw_text[start:end]
            resolved = span.model_copy(
                update={
                    "text": matched_slice,
                    "start_char": start,
                    "end_char": end,
                    "page_number": target_page,
                }
            )
            return SpanValidationResult(
                is_valid=True,
                resolved_span=resolved,
                is_exact=False,
                match_count=1,
            )

        if len(norm_matches) > 1:
            return SpanValidationResult(
                is_valid=False,
                match_count=len(norm_matches),
                error_message=(
                    f"Ambiguous match: span text matches {len(norm_matches)} times after "
                    "whitespace normalization. Explicit offsets required."
                ),
            )

    # Not found in raw_text
    return SpanValidationResult(
        is_valid=False,
        match_count=0,
        error_message=(
            f"Source span text '{span_text[:60]}...' not found in clause {clause.clause_id}."
        ),
    )
