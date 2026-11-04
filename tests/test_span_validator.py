"""Unit tests for the source-span validator (Stage S3)."""

import pytest
from pydantic import ValidationError

from termsheet_ai.extract.span_validator import (
    validate_source_span,
)
from termsheet_ai.schemas.contracts import ClauseSpan, SourceSpan


@pytest.fixture
def sample_clause() -> ClauseSpan:
    return ClauseSpan(
        clause_id="c_10",
        raw_text=(
            "Series A Preferred will receive a 2.0x liquidation preference, "
            "participating fully in any remaining proceeds. In the event of "
            "liquidation, holders receive 2.0x of their original issue price."
        ),
        page_number=2,
    )


def test_exact_span_match_with_valid_offsets(sample_clause: ClauseSpan):
    """Test when valid start and end offsets identify the exact text."""
    target_text = "2.0x liquidation preference"
    start = sample_clause.raw_text.index(target_text)
    end = start + len(target_text)

    span = SourceSpan(text=target_text, start_char=start, end_char=end, page_number=2)
    result = validate_source_span(sample_clause, span)

    assert result.is_valid is True
    assert result.is_exact is True
    assert result.match_count == 1
    assert result.resolved_span is not None
    assert result.resolved_span.start_char == start
    assert result.resolved_span.end_char == end
    assert result.resolved_span.text == target_text
    assert result.resolved_span.page_number == 2


def test_invalid_offsets_mismatch_fails(sample_clause: ClauseSpan):
    """Test when provided offsets point to completely different text."""
    span = SourceSpan(
        text="participating fully",
        start_char=0,
        end_char=10,  # "Series A P"
        page_number=2,
    )
    result = validate_source_span(sample_clause, span)

    assert result.is_valid is False
    assert result.resolved_span is None
    assert "mismatch" in result.error_message.lower()


def test_out_of_bounds_offsets_fail(sample_clause: ClauseSpan):
    """Test offsets that exceed text length or are reversed."""
    # Exceed length
    span_oob = SourceSpan(
        text="2.0x",
        start_char=500,
        end_char=504,
        page_number=2,
    )
    res_oob = validate_source_span(sample_clause, span_oob)
    assert res_oob.is_valid is False
    assert "invalid offset range" in res_oob.error_message.lower()

    # Inverted range
    span_inv = SourceSpan(
        text="2.0x",
        start_char=30,
        end_char=20,
        page_number=2,
    )
    res_inv = validate_source_span(sample_clause, span_inv)
    assert res_inv.is_valid is False
    assert "invalid offset range" in res_inv.error_message.lower()


def test_zero_length_offset_range_fails(sample_clause: ClauseSpan):
    """Test start_char == end_char is rejected."""
    span = SourceSpan(
        text="2.0x",
        start_char=10,
        end_char=10,
        page_number=2,
    )
    res = validate_source_span(sample_clause, span)
    assert res.is_valid is False
    assert "invalid offset range" in res.error_message.lower()


def test_negative_offsets_fail(sample_clause: ClauseSpan):
    """Test negative offsets are rejected."""
    span = SourceSpan(
        text="2.0x",
        start_char=-5,
        end_char=10,
        page_number=2,
    )
    res = validate_source_span(sample_clause, span)
    assert res.is_valid is False
    assert "invalid offset range" in res.error_message.lower()


def test_missing_offsets_unambiguous_auto_resolution(sample_clause: ClauseSpan):
    """Test that missing offsets (None) are uniquely resolved when text appears once."""
    target_text = "participating fully"
    span = SourceSpan(text=target_text, page_number=2)  # offsets default to None

    result = validate_source_span(sample_clause, span)

    assert result.is_valid is True
    assert result.is_exact is True
    assert result.resolved_span is not None
    assert result.resolved_span.start_char == sample_clause.raw_text.index(target_text)
    assert result.resolved_span.end_char == result.resolved_span.start_char + len(target_text)
    assert result.resolved_span.page_number == 2


def test_repeated_occurrences_missing_offsets_ambiguous_failure(sample_clause: ClauseSpan):
    """Test that repeated occurrences fail with an explicit ambiguity message when offsets are missing."""
    # "2.0x" appears twice in sample_clause
    span = SourceSpan(text="2.0x", page_number=2)
    result = validate_source_span(sample_clause, span)

    assert result.is_valid is False
    assert result.match_count == 2
    assert "ambiguous match" in result.error_message.lower()
    assert result.resolved_span is None


def test_repeated_occurrences_succeed_with_explicit_offsets(sample_clause: ClauseSpan):
    """Test that repeated text succeeds if explicit offsets disambiguate which one is cited."""
    target = "2.0x"
    # Find second occurrence
    first_idx = sample_clause.raw_text.index(target)
    second_idx = sample_clause.raw_text.index(target, first_idx + 1)

    span = SourceSpan(
        text=target,
        start_char=second_idx,
        end_char=second_idx + len(target),
        page_number=2,
    )
    result = validate_source_span(sample_clause, span)

    assert result.is_valid is True
    assert result.is_exact is True
    assert result.match_count == 2  # Total occurrences in clause is 2
    assert result.resolved_span.start_char == second_idx


def test_whitespace_normalization_resolution():
    """Test normalized whitespace matching without conflating offsets with query string length."""
    clause = ClauseSpan(
        clause_id="c_20",
        raw_text="The company    shall issue\n\nSeries A Preferred shares at $1.00 per share.",
        page_number=1,
    )
    # Query has normalized single spaces
    span = SourceSpan(text="company shall issue Series A")

    # With allow_whitespace_normalization=True
    result = validate_source_span(clause, span, allow_whitespace_normalization=True)
    assert result.is_valid is True
    assert result.is_exact is False
    assert result.match_count == 1
    assert result.resolved_span is not None
    # Verify that the resolved span offsets slice the original text exactly
    slice_in_clause = clause.raw_text[
        result.resolved_span.start_char : result.resolved_span.end_char
    ]
    assert "company" in slice_in_clause and "Series A" in slice_in_clause
    # And the resolved span text matches the raw slice
    assert result.resolved_span.text == slice_in_clause

    # With allow_whitespace_normalization=False
    result_strict = validate_source_span(clause, span, allow_whitespace_normalization=False)
    assert result_strict.is_valid is False
    assert "not found" in result_strict.error_message.lower()


def test_provided_offsets_with_whitespace_difference_fails_strictly():
    """Provided offsets must match verbatim; differing whitespace at specified offsets is rejected."""
    clause = ClauseSpan(
        clause_id="c_21",
        raw_text="Founders have 4-year   vesting.",
        page_number=1,
    )
    target_slice = "4-year   vesting"
    start = clause.raw_text.index(target_slice)
    end = start + len(target_slice)

    # Span claims text is "4-year vesting" (single space) at offsets that contain 3 spaces
    span = SourceSpan(text="4-year vesting", start_char=start, end_char=end, page_number=1)
    result = validate_source_span(clause, span)

    assert result.is_valid is False
    assert "differ only by whitespace" in result.error_message


def test_hallucinated_text_rejected(sample_clause: ClauseSpan):
    """Test that text not present anywhere in the clause is rejected as hallucination."""
    span = SourceSpan(text="full-ratchet anti-dilution protection", page_number=2)
    result = validate_source_span(sample_clause, span)

    assert result.is_valid is False
    assert result.match_count == 0
    assert result.resolved_span is None
    assert "not found in clause" in result.error_message.lower()


def test_partial_offsets_rejected(sample_clause: ClauseSpan):
    """Test when only start_char or only end_char is provided."""
    span1 = SourceSpan(
        text="2.0x liquidation preference", start_char=10, end_char=None, page_number=2
    )
    res1 = validate_source_span(sample_clause, span1)
    assert res1.is_valid is False
    assert "incomplete offsets" in res1.error_message.lower()

    span2 = SourceSpan(
        text="2.0x liquidation preference", start_char=None, end_char=20, page_number=2
    )
    res2 = validate_source_span(sample_clause, span2)
    assert res2.is_valid is False
    assert "incomplete offsets" in res2.error_message.lower()


def test_empty_span_rejected():
    """Test Pydantic rejects empty text and whitespace-only strings."""
    with pytest.raises(ValidationError):
        SourceSpan(text="")

    with pytest.raises(ValidationError):
        SourceSpan(text="   ")


def test_partial_word_match_prevented():
    """Ensure partial words like 'part' do not match 'participating' during normalized matching."""
    clause = ClauseSpan(
        clause_id="c_30",
        raw_text="Holders are participating in proceeds.",
        page_number=1,
    )
    span = SourceSpan(text="part in proceeds")
    result = validate_source_span(clause, span, allow_whitespace_normalization=True)
    assert result.is_valid is False
    assert "not found in clause" in result.error_message.lower()


def test_punctuation_preserved_in_normalized_matching():
    """Punctuation must be preserved: '1.0x, non-participating' must not match '1.0x non-participating'."""
    clause = ClauseSpan(
        clause_id="c_31",
        raw_text="The shares are 1.0x, non-participating preferred.",
        page_number=1,
    )
    # Span is missing the comma
    span = SourceSpan(text="1.0x non-participating")
    result = validate_source_span(clause, span, allow_whitespace_normalization=True)
    assert result.is_valid is False
    assert "not found in clause" in result.error_message.lower()


def test_ambiguous_normalized_matches_rejected():
    """If normalized match occurs multiple times without explicit offsets, it must be rejected."""
    clause = ClauseSpan(
        clause_id="c_32",
        raw_text="Item A: 4-year   vesting. Item B: 4-year\n\nvesting.",
        page_number=1,
    )
    span = SourceSpan(text="4-year vesting")
    result = validate_source_span(clause, span, allow_whitespace_normalization=True)
    assert result.is_valid is False
    assert result.match_count == 2
    assert "ambiguous match" in result.error_message.lower()


def test_page_number_consistency_inherits_when_none():
    """When span page_number is None, it inherits clause page_number."""
    clause = ClauseSpan(
        clause_id="c_33",
        raw_text="Clause text on page 3.",
        page_number=3,
    )
    span = SourceSpan(text="Clause text", page_number=None)
    result = validate_source_span(clause, span)
    assert result.is_valid is True
    assert result.resolved_span.page_number == 3


def test_page_number_mismatch_rejected():
    """When span page_number conflicts with clause page_number, it is rejected."""
    clause = ClauseSpan(
        clause_id="c_34",
        raw_text="Clause text on page 3.",
        page_number=3,
    )
    # Span explicitly claims it is on page 1
    span = SourceSpan(text="Clause text", page_number=1)
    result = validate_source_span(clause, span)
    assert result.is_valid is False
    assert "page number mismatch" in result.error_message.lower()
