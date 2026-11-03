"""Tests for the S5 Number Consistency Checker."""

from termsheet_ai.explain.consistency import NumberConsistencyChecker
from termsheet_ai.schemas.contracts import (
    ClauseSpan,
    ExtractedFacts,
    LiquidationPreferenceFact,
    ParticipationType,
    SourceSpan,
)


def test_extract_numbers():
    text = (
        "The Series A receives a 2.5x multiple with a $5,000,000 cap and 20% interest over 4 years."
    )
    numbers = NumberConsistencyChecker.extract_numbers(text)
    assert 2.5 in numbers
    assert 5000000.0 in numbers
    assert 20.0 in numbers
    assert 4.0 in numbers


def test_consistent_summary_passes():
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="Series A Preferred will receive a 2.0x liquidation preference, participating fully.",
        page_number=1,
    )
    facts = ExtractedFacts(
        doc_id="doc_1",
        liquidation_preference=LiquidationPreferenceFact(
            clause_id="c_1",
            multiple=2.0,
            participation=ParticipationType.FULLY_PARTICIPATING,
            source_span=SourceSpan(text=clause.raw_text, page_number=1),
        ),
    )

    checker = NumberConsistencyChecker()
    summary = (
        "This clause awards Series A a 2.0x liquidation preference payout, above the standard 1.0x."
    )
    verified, unverified = checker.verify_summary(summary, clause=clause, facts=facts)

    assert verified is True
    assert unverified == []


def test_hallucinated_number_fails():
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="Series A Preferred will receive a 2.0x liquidation preference.",
        page_number=1,
    )
    facts = ExtractedFacts(
        doc_id="doc_1",
        liquidation_preference=LiquidationPreferenceFact(
            clause_id="c_1",
            multiple=2.0,
            participation=ParticipationType.NON_PARTICIPATING,
            source_span=SourceSpan(text=clause.raw_text, page_number=1),
        ),
    )

    checker = NumberConsistencyChecker()
    hallucinated_summary = "This clause gives investors a 3.5x liquidation multiple payout."
    verified, unverified = checker.verify_summary(hallucinated_summary, clause=clause, facts=facts)

    assert verified is False
    assert 3.5 in unverified
