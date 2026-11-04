"""Unit and integration tests for FactExtractor (Stage S3)."""

from __future__ import annotations

import json
import os
import socket

import pytest
from dotenv import load_dotenv

from termsheet_ai.extract.extractor import (
    FactExtractor,
    _extract_json_block,
    create_fact_extractor,
)
from termsheet_ai.schemas.contracts import (
    AntiDilutionType,
    ClauseFamily,
    ClauseSpan,
    ParticipationType,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Network availability check for live integration tests
# ---------------------------------------------------------------------------

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))


def _network_available() -> bool:
    """Return True if the Gemini API hostname can be resolved."""
    try:
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
        return True
    except (socket.gaierror, OSError):
        return False


_NETWORK_OK = _network_available()


# ===========================================================================
# 1. JSON parsing tests
# ===========================================================================


def test_extract_json_block_formats():
    """Verify JSON block extraction from markdown, raw json, and noisy wrappers."""
    raw = '{"multiple": 2.0, "participation": "fully_participating"}'
    assert _extract_json_block(raw) == {"multiple": 2.0, "participation": "fully_participating"}

    fenced = 'Here is the output:\n```json\n{"multiple": 1.0}\n```\nHope that helps!'
    assert _extract_json_block(fenced) == {"multiple": 1.0}

    noisy = 'The answer is {"cap_multiple": 3.0} according to section 2.'
    assert _extract_json_block(noisy) == {"cap_multiple": 3.0}

    # Array containing dict
    array_json = '[{"multiple": 1.5}]'
    assert _extract_json_block(array_json) == {"multiple": 1.5}

    assert _extract_json_block("No json present here") is None
    assert _extract_json_block('{"unclosed": ') is None
    assert _extract_json_block("") is None


# ===========================================================================
# 2. Mocked LLM Extraction Tests for All Five Categories
# ===========================================================================


def test_gemini_mocked_liquidation_preference():
    """Verify LiquidationPreferenceFact extraction via mocked Gemini."""
    clause = ClauseSpan(
        clause_id="c_lp",
        raw_text=(
            "In the event of liquidation, Series A Preferred receives a 1.5x preference, "
            "participating with a 3.0x cap, pari passu with Series Seed."
        ),
        page_number=1,
        detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
    )

    def gemini_mock(prompt: str) -> str:
        assert "liquidation" in prompt.lower()
        return json.dumps(
            {
                "multiple": 1.5,
                "participation": "capped_participating",
                "cap_multiple": 3.0,
                "seniority": "pari_passu",
                "quote": "1.5x preference, participating with a 3.0x cap",
            }
        )

    extractor = FactExtractor(llm_callable=gemini_mock)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.liquidation_preference is not None
    fact = facts.liquidation_preference
    assert fact.multiple == 1.5
    assert fact.participation == ParticipationType.CAPPED_PARTICIPATING
    assert fact.cap_multiple == 3.0
    assert fact.seniority == "pari_passu"
    assert fact.is_grounded is True
    assert extractor.extraction_modes.get("c_lp") == "llm"


def test_gemini_mocked_anti_dilution():
    """Verify AntiDilutionFact extraction via mocked Gemini."""
    clause = ClauseSpan(
        clause_id="c_ad",
        raw_text="The conversion price shall be adjusted on a broad-based weighted average anti-dilution basis.",
        page_number=2,
        detected_family=ClauseFamily.ANTI_DILUTION,
    )

    def gemini_mock(prompt: str) -> str:
        assert "anti-dilution" in prompt.lower() or "anti_dilution" in prompt.lower()
        return json.dumps(
            {
                "anti_dilution_type": "broad_based_weighted_average",
                "quote": "broad-based weighted average anti-dilution",
            }
        )

    extractor = FactExtractor(llm_callable=gemini_mock)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.anti_dilution is not None
    fact = facts.anti_dilution
    assert fact.anti_dilution_type == AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE
    assert fact.is_grounded is True
    assert extractor.extraction_modes.get("c_ad") == "llm"


def test_gemini_mocked_board_composition():
    """Verify BoardCompositionFact extraction and majority predicate via mocked Gemini."""
    clause = ClauseSpan(
        clause_id="c_bc",
        raw_text="The Board shall consist of 5 members: 2 Founder directors, 2 Preferred directors, and 1 Independent director.",
        page_number=3,
        detected_family=ClauseFamily.BOARD_COMPOSITION,
    )

    def gemini_mock(prompt: str) -> str:
        return json.dumps(
            {
                "founder_seats": 2,
                "investor_seats": 2,
                "independent_seats": 1,
                "total_seats": 5,
                "quote": "2 Founder directors, 2 Preferred directors, and 1 Independent director",
            }
        )

    extractor = FactExtractor(llm_callable=gemini_mock)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.board_composition is not None
    board = facts.board_composition
    assert board.founder_seats == 2
    assert board.investor_seats == 2
    assert board.independent_seats == 1
    assert board.total_seats == 5
    assert board.investor_majority is False  # 2 is not strictly > 2
    assert board.is_grounded is True
    assert extractor.extraction_modes.get("c_bc") == "llm"


def test_gemini_mocked_founder_vesting():
    """Verify VestingFact extraction via mocked Gemini."""
    clause = ClauseSpan(
        clause_id="c_vest",
        raw_text="Common stock shall vest over 4 years with a 12-month cliff and single-trigger acceleration upon change of control.",
        page_number=4,
        detected_family=ClauseFamily.FOUNDER_VESTING,
    )

    def gemini_mock(prompt: str) -> str:
        return json.dumps(
            {
                "total_vesting_years": 4.0,
                "cliff_months": 12,
                "acceleration_type": "single_trigger",
                "quote": "vest over 4 years with a 12-month cliff",
            }
        )

    extractor = FactExtractor(llm_callable=gemini_mock)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.founder_vesting is not None
    vesting = facts.founder_vesting
    assert vesting.total_vesting_years == 4.0
    assert vesting.cliff_months == 12
    assert vesting.acceleration_type == "single_trigger"
    assert vesting.is_grounded is True
    assert extractor.extraction_modes.get("c_vest") == "llm"


def test_gemini_mocked_protective_provisions():
    """Verify ProtectiveProvisionsFact extraction via mocked Gemini."""
    clause = ClauseSpan(
        clause_id="c_prot",
        raw_text="Protective Provisions: Company shall not without Series A consent amend Charter or create senior shares.",
        page_number=5,
        detected_family=ClauseFamily.PROTECTIVE_PROVISIONS,
    )

    def gemini_mock(prompt: str) -> str:
        return json.dumps(
            {
                "requires_series_majority": True,
                "covered_actions": ["amend_charter", "create_senior_shares"],
                "quote": "Company shall not without Series A consent",
            }
        )

    extractor = FactExtractor(llm_callable=gemini_mock)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.protective_provisions is not None
    prot = facts.protective_provisions
    assert prot.requires_series_majority is True
    assert "amend_charter" in prot.covered_actions
    assert "create_senior_shares" in prot.covered_actions
    assert prot.is_grounded is True
    assert extractor.extraction_modes.get("c_prot") == "llm"


# ===========================================================================
# 3. Guarding Against Hallucination & Missing Information
# ===========================================================================


def test_extractor_hallucinated_quote_marked_ungrounded():
    """When LLM returns a hallucinated quote that doesn't exist in clause, fact is marked is_grounded=False."""
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="The Series A shares have a 1.0x non-participating liquidation preference.",
        page_number=1,
        detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
    )

    def hallucinating_llm(prompt: str) -> str:
        return json.dumps(
            {
                "multiple": 3.0,
                "participation": "fully_participating",
                "quote": "3.0x super-priority participation with guaranteed return",  # Absent from text
            }
        )

    extractor = FactExtractor(llm_callable=hallucinating_llm)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.liquidation_preference is not None
    assert facts.liquidation_preference.is_grounded is False
    assert facts.liquidation_preference.multiple == 3.0


def test_extractor_contradictory_number_in_quote_marked_ungrounded():
    """When quote text exists verbatim but contradicts the claimed fact number."""
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="The Series A shares shall have a 1.0x liquidation preference.",
        page_number=1,
        detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
    )

    def contradictory_num_llm(prompt: str) -> str:
        return json.dumps(
            {
                "multiple": 2.0,  # Contradicts the text
                "participation": "non_participating",
                "quote": "1.0x liquidation preference",  # Exists verbatim
            }
        )

    extractor = FactExtractor(llm_callable=contradictory_num_llm)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.liquidation_preference is not None
    assert facts.liquidation_preference.multiple == 2.0
    # The quote contains 1.0x, not 2.0x; numerical verification marks is_grounded=False
    assert facts.liquidation_preference.is_grounded is False


def test_extractor_malformed_llm_falls_back_to_deterministic():
    """When LLM returns garbage or invalid JSON, extractor falls back to deterministic extraction."""
    clause = ClauseSpan(
        clause_id="c_2",
        raw_text="Anti-Dilution: The shares feature full-ratchet anti-dilution adjustments.",
        page_number=2,
        detected_family=ClauseFamily.ANTI_DILUTION,
    )

    def broken_llm(prompt: str) -> str:
        return "I am unable to answer this question as an AI language model."

    extractor = FactExtractor(llm_callable=broken_llm)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    assert facts.anti_dilution is not None
    assert facts.anti_dilution.anti_dilution_type == AntiDilutionType.FULL_RATCHET
    assert facts.anti_dilution.is_grounded is True
    assert "full-ratchet" in facts.anti_dilution.source_span.text
    assert extractor.extraction_modes.get("c_2") == "fallback"


def test_extractor_missing_clause_yields_none_fact():
    """Clauses lacking relevant facts or numeric terms do not fabricate facts."""
    clause = ClauseSpan(
        clause_id="c_99",
        raw_text="Liquidation: General discussions regarding dissolution were held without term agreements.",
        page_number=1,
        detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
    )

    extractor = FactExtractor(llm_callable=None)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    # Should remain None because no multiple is present in the text
    assert facts.liquidation_preference is None


def test_extractor_missing_info_llm_returns_null_never_fabricated():
    """When LLM reports null/unknown values for absent attributes, no facts are invented."""
    clause = ClauseSpan(
        clause_id="c_empty_vest",
        raw_text="Vesting: Subject to standard company policies as determined by the Board.",
        page_number=1,
        detected_family=ClauseFamily.FOUNDER_VESTING,
    )

    def null_llm(prompt: str) -> str:
        return json.dumps(
            {
                "total_vesting_years": None,
                "cliff_months": None,
                "acceleration_type": "none",
                "quote": "",
            }
        )

    extractor = FactExtractor(llm_callable=null_llm)
    facts = extractor.extract(doc_id="test_doc", clauses=[clause])

    # Should remain None because no vesting schedule was found
    assert facts.founder_vesting is None


# ===========================================================================
# 4. Factory and Wiring Tests
# ===========================================================================


def test_create_fact_extractor_wiring():
    """Test create_fact_extractor without Gemini returns extractor with no llm_callable."""
    extractor_no_llm = create_fact_extractor(use_gemini=False)
    assert extractor_no_llm.llm_callable is None

    # With Gemini and mocked key
    extractor_gemini = create_fact_extractor(
        use_gemini=True,
        use_cache=False,
        api_key="mock-key-for-test",
    )
    assert extractor_gemini.llm_callable is not None
    assert callable(extractor_gemini.llm_callable)


def test_fact_extractor_with_gemini_classmethod():
    """Test FactExtractor.with_gemini classmethod sets up callable."""
    extractor = FactExtractor.with_gemini(
        api_key="mock-key-for-test",
        use_cache=False,
    )
    assert extractor.llm_callable is not None
    assert callable(extractor.llm_callable)


# ===========================================================================
# 5. Live Gemini Integration Test (Optional — runs with network & key)
# ===========================================================================


@pytest.mark.live
@pytest.mark.skipif(
    not _HAS_KEY,
    reason="GEMINI_API_KEY not set — live test skipped",
)
@pytest.mark.skipif(
    not _NETWORK_OK,
    reason="No outbound network access — live test skipped",
)
class TestFactExtractorLiveIntegration:
    """End-to-end integration test calling real Gemini on a non-confidential clause."""

    def test_live_liquidation_preference_extraction(self):
        """Extract a real 1.0x non-participating liquidation preference via Gemini."""
        clause = ClauseSpan(
            clause_id="live_lp_1",
            raw_text=(
                "The Series A Preferred Stock shall have a non-participating liquidation "
                "preference equal to 1.0x the original issue price, plus any declared but unpaid dividends."
            ),
            page_number=1,
            detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
        )

        extractor = FactExtractor.with_gemini(model="gemini-3.5-flash")
        facts = extractor.extract(doc_id="live_doc", clauses=[clause])

        assert facts.liquidation_preference is not None
        fact = facts.liquidation_preference

        assert fact.multiple == pytest.approx(1.0, abs=0.01)
        assert fact.participation == ParticipationType.NON_PARTICIPATING
        assert fact.is_grounded is True
        assert extractor.extraction_modes.get("live_lp_1") == "llm"

    def test_live_anti_dilution_extraction(self):
        """Extract a broad-based weighted average anti-dilution provision via Gemini."""
        clause = ClauseSpan(
            clause_id="live_ad_1",
            raw_text="The Series A Conversion Price will be subject to customary broad-based weighted average anti-dilution adjustments in the event of down-round issuances.",
            page_number=2,
            detected_family=ClauseFamily.ANTI_DILUTION,
        )

        extractor = FactExtractor.with_gemini(model="gemini-3.5-flash")
        facts = extractor.extract(doc_id="live_doc", clauses=[clause])

        assert facts.anti_dilution is not None
        assert (
            facts.anti_dilution.anti_dilution_type == AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE
        )
        assert facts.anti_dilution.is_grounded is True
        assert extractor.extraction_modes.get("live_ad_1") == "llm"

    def test_live_board_composition_extraction(self):
        """Extract a 5-member board composition via Gemini."""
        clause = ClauseSpan(
            clause_id="live_bc_1",
            raw_text="The Board shall consist of five (5) directors: two (2) appointed by the Founders, two (2) appointed by Series A, and one (1) mutually agreed independent director.",
            page_number=3,
            detected_family=ClauseFamily.BOARD_COMPOSITION,
        )

        extractor = FactExtractor.with_gemini(model="gemini-3.5-flash")
        facts = extractor.extract(doc_id="live_doc", clauses=[clause])

        assert facts.board_composition is not None
        board = facts.board_composition
        assert board.founder_seats == 2
        assert board.investor_seats == 2
        assert board.independent_seats == 1
        assert board.total_seats == 5
        assert board.investor_majority is False
        assert board.is_grounded is True
        assert extractor.extraction_modes.get("live_bc_1") == "llm"

    def test_live_founder_vesting_extraction(self):
        """Extract a 4-year vesting schedule with 12-month cliff and no acceleration via Gemini."""
        clause = ClauseSpan(
            clause_id="live_vest_1",
            raw_text="Founder equity shall vest over a 4-year schedule with a 12-month cliff and no acceleration.",
            page_number=4,
            detected_family=ClauseFamily.FOUNDER_VESTING,
        )

        extractor = FactExtractor.with_gemini(model="gemini-3.5-flash")
        facts = extractor.extract(doc_id="live_doc", clauses=[clause])

        assert facts.founder_vesting is not None
        vesting = facts.founder_vesting
        assert vesting.total_vesting_years == 4.0
        assert vesting.cliff_months == 12
        assert vesting.acceleration_type == "none"
        assert vesting.is_grounded is True
        assert extractor.extraction_modes.get("live_vest_1") == "llm"

    def test_live_protective_provisions_extraction(self):
        """Extract protective provisions requiring Series A majority consent via Gemini."""
        clause = ClauseSpan(
            clause_id="live_prot_1",
            raw_text="Consent of the holders of a majority of Series A Preferred shall be required for: (a) any merger, sale, or liquidation of the Company; and (b) creating new senior equity.",
            page_number=5,
            detected_family=ClauseFamily.PROTECTIVE_PROVISIONS,
        )

        extractor = FactExtractor.with_gemini(model="gemini-3.5-flash")
        facts = extractor.extract(doc_id="live_doc", clauses=[clause])

        assert facts.protective_provisions is not None
        prot = facts.protective_provisions
        assert prot.requires_series_majority is True
        assert len(prot.covered_actions) >= 1
        assert prot.is_grounded is True
        assert extractor.extraction_modes.get("live_prot_1") == "llm"

    def test_live_missing_information_never_fabricated(self):
        """Verify that Gemini returns null and FactExtractor does not fabricate facts when terms are absent."""
        clause = ClauseSpan(
            clause_id="live_missing_1",
            raw_text="Liquidation Matters: General governance procedures for corporate dissolution will follow Delaware state law without preferential distributions.",
            page_number=6,
            detected_family=ClauseFamily.LIQUIDATION_PREFERENCE,
        )

        extractor = FactExtractor.with_gemini(model="gemini-3.5-flash")
        facts = extractor.extract(doc_id="live_doc", clauses=[clause])

        # No multiple is agreed upon or stated; fact must remain None
        assert facts.liquidation_preference is None
        assert extractor.extraction_modes.get("live_missing_1") == "llm"
