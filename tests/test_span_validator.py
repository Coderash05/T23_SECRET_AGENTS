"""Tests for the S3 SpanValidator fact grounding."""

from termsheet_ai.extraction.span_validator import SpanValidator
from termsheet_ai.schemas.contracts import (
    BoardCompositionFact,
    ExtractedFacts,
    LiquidationPreferenceFact,
    ParticipationType,
    SourceSpan,
)


def test_span_validator_valid_numbers():
    validator = SpanValidator()
    raw_span = "Holders of Series A Preferred shall receive a 2.0x liquidation preference."

    fact = LiquidationPreferenceFact(
        clause_id="c_1",
        multiple=2.0,
        participation=ParticipationType.NON_PARTICIPATING,
        source_span=SourceSpan(text=raw_span, page_number=1),
    )

    validated = validator.validate_liquidation_preference(fact)
    assert validated.is_grounded is True
    assert validated.multiple == 2.0


def test_span_validator_hallucinated_number():
    validator = SpanValidator()
    # Span says 1.0x, but extracted multiple claims 3.0x
    raw_span = "Holders of Series A Preferred shall receive a 1.0x liquidation preference."

    fact = LiquidationPreferenceFact(
        clause_id="c_1",
        multiple=3.0,
        participation=ParticipationType.NON_PARTICIPATING,
        source_span=SourceSpan(text=raw_span, page_number=1),
    )

    validated = validator.validate_liquidation_preference(fact)
    assert validated.is_grounded is False
    assert validated.multiple == 1.0  # Reset to standard baseline upon hallucination


def test_span_validator_board_fact():
    validator = SpanValidator()
    raw_span = "The board consists of 3 directors: 2 investor seats and 1 founder seat."

    fact = BoardCompositionFact(
        clause_id="c_2",
        founder_seats=1,
        investor_seats=2,
        total_seats=3,
        source_span=SourceSpan(text=raw_span, page_number=1),
    )

    validated = validator.validate_board_composition(fact)
    assert validated.is_grounded is True


def test_validate_all_facts():
    validator = SpanValidator()
    facts = ExtractedFacts(
        doc_id="test_doc",
        liquidation_preference=LiquidationPreferenceFact(
            clause_id="c_1",
            multiple=2.0,
            participation=ParticipationType.NON_PARTICIPATING,
            source_span=SourceSpan(text="2.0x preference", page_number=1),
        ),
    )

    validated = validator.validate_all(facts)
    assert validated.liquidation_preference is not None
    assert validated.liquidation_preference.multiple == 2.0
    assert validated.liquidation_preference.is_grounded is True
