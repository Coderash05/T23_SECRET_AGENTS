"""Tests for the S5 Number Consistency Checker and PlainEnglishExplainer."""

from termsheet_ai.explain.consistency import NumberConsistencyChecker
from termsheet_ai.explain.explainer import PlainEnglishExplainer
from termsheet_ai.schemas.contracts import (
    AuditTrace,
    ClauseFamily,
    ClauseSpan,
    DocumentRiskScore,
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


def test_explainer_with_sbert_retriever_cites_market_exemplar():
    class MockRetriever:
        def retrieve_nearest_exemplars(self, clause, top_k=1):
            return [
                {
                    "text": "1.0x non-participating preferred is standard.",
                    "family": ClauseFamily.LIQUIDATION_PREFERENCE,
                    "source": "NVCA Model Term Sheet §3",
                    "cosine_similarity": 0.88,
                }
            ]

    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="The Series A Preferred shall have a 1.0x non-participating liquidation preference.",
        page_number=1,
        detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
    )
    trace = AuditTrace(
        doc_id="doc_1",
        risk_score=DocumentRiskScore(overall_score=0.0),
        flags=[],
        rule_traces=[],
    )

    explainer = PlainEnglishExplainer(retriever=MockRetriever())
    summaries = explainer.explain([clause], trace)

    assert len(summaries) == 1
    assert "NVCA Model Term Sheet §3" in summaries[0].summary_text
    assert summaries[0].numbers_verified is True
