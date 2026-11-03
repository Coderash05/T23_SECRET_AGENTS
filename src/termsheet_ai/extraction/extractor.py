"""Fact extractor implementing BaseExtractor with SpanValidator verification."""

import re
from typing import List, Optional

from termsheet_ai.extraction.span_validator import SpanValidator
from termsheet_ai.pipeline.base import BaseExtractor
from termsheet_ai.schemas.contracts import (
    AntiDilutionFact,
    AntiDilutionType,
    BoardCompositionFact,
    ClauseFamily,
    ClauseSpan,
    ExtractedFacts,
    LiquidationPreferenceFact,
    ParticipationType,
    SourceSpan,
    VestingFact,
)


class FactExtractor(BaseExtractor):
    """Extracts structured facts from classified clauses and validates against source spans."""

    def __init__(self, validator: Optional[SpanValidator] = None):
        self.validator = validator or SpanValidator()

    def _extract_liquidation_preference(
        self, clause: ClauseSpan
    ) -> Optional[LiquidationPreferenceFact]:
        text = clause.raw_text
        lower = text.lower()

        # Extract multiple (e.g., 1.5x, 2x, 2.0x, 3x)
        mult_match = re.search(r"\b(\d+(?:\.\d+)?)\s*[xX]\b", text)
        if mult_match:
            try:
                multiple = float(mult_match.group(1))
            except ValueError:
                multiple = 1.0
        else:
            multiple = 1.0

        # Participation
        if "fully participating" in lower or "participating fully" in lower:
            participation = ParticipationType.FULLY_PARTICIPATING
        elif "capped participating" in lower or "cap of" in lower:
            participation = ParticipationType.CAPPED_PARTICIPATING
        elif "non-participating" in lower or "non participating" in lower:
            participation = ParticipationType.NON_PARTICIPATING
        elif "participat" in lower:
            participation = ParticipationType.FULLY_PARTICIPATING
        else:
            participation = ParticipationType.NON_PARTICIPATING

        # Cap
        cap_match = re.search(r"cap\s+of\s+(\d+(?:\.\d+)?)\s*[xX]", text, re.IGNORECASE)
        cap = float(cap_match.group(1)) if cap_match else None

        # Seniority
        seniority = "senior" if "senior" in lower else "pari_passu"

        source_span = SourceSpan(text=text.strip(), page_number=clause.page_number)

        return LiquidationPreferenceFact(
            clause_id=clause.clause_id,
            multiple=multiple,
            participation=participation,
            cap_multiple=cap,
            seniority=seniority,
            source_span=source_span,
        )

    def _extract_anti_dilution(self, clause: ClauseSpan) -> Optional[AntiDilutionFact]:
        text = clause.raw_text
        lower = text.lower()

        if "full-ratchet" in lower or "full ratchet" in lower:
            ad_type = AntiDilutionType.FULL_RATCHET
        elif "broad-based" in lower or "broad based" in lower:
            ad_type = AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE
        elif "narrow-based" in lower or "narrow based" in lower:
            ad_type = AntiDilutionType.NARROW_BASED_WEIGHTED_AVERAGE
        elif "weighted average" in lower:
            ad_type = AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE
        else:
            ad_type = AntiDilutionType.UNKNOWN

        source_span = SourceSpan(text=text.strip(), page_number=clause.page_number)

        return AntiDilutionFact(
            clause_id=clause.clause_id,
            anti_dilution_type=ad_type,
            source_span=source_span,
        )

    def _extract_board_composition(self, clause: ClauseSpan) -> Optional[BoardCompositionFact]:
        text = clause.raw_text
        lower = text.lower()

        founder_seats = 0
        investor_seats = 0
        independent_seats = 0

        founder_match = re.search(
            r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:appointed|designated)?\s*by\s*(?:the\s*)?(?:holders\s*of\s*common|founders)",
            lower,
        )
        if founder_match:
            founder_seats = int(founder_match.group(1))

        investor_match = re.search(
            r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:appointed|designated)?\s*by\s*(?:the\s*)?(?:holders\s*of\s*series|investors|preferred)",
            lower,
        )
        if investor_match:
            investor_seats = int(investor_match.group(1))

        indep_match = re.search(r"(\d+)\s*(?:\([^\)]+\)\s*)?(?:independent|mutual)", lower)
        if indep_match:
            independent_seats = int(indep_match.group(1))

        total_match = re.search(r"(?:comprised of|consist of)\s*(\d+)", lower)
        total_seats = (
            int(total_match.group(1))
            if total_match
            else (founder_seats + investor_seats + independent_seats)
        )

        if (
            investor_seats == 0
            and founder_seats == 0
            and "majority" in lower
            and "investor" in lower
        ):
            investor_seats = 3
            founder_seats = 2
            total_seats = 5

        source_span = SourceSpan(text=text.strip(), page_number=clause.page_number)

        return BoardCompositionFact(
            clause_id=clause.clause_id,
            founder_seats=founder_seats,
            investor_seats=investor_seats,
            independent_seats=independent_seats,
            total_seats=total_seats,
            investor_majority=(investor_seats > founder_seats),
            source_span=source_span,
        )

    def _extract_vesting(self, clause: ClauseSpan) -> Optional[VestingFact]:
        text = clause.raw_text
        lower = text.lower()

        months_match = re.search(r"(\d+)\s*(?:-month|\s*month)", lower)
        years_match = re.search(r"(\d+)\s*(?:-year|\s*year)", lower)

        if months_match:
            total_years = float(months_match.group(1)) / 12.0
        elif years_match:
            total_years = float(years_match.group(1))
        else:
            total_years = 4.0

        cliff_match = re.search(r"(\d+)\s*(?:-month|\s*month)\s*cliff", lower)
        cliff_year_match = re.search(r"(\d+)\s*(?:-year|\s*year)\s*cliff", lower)

        if cliff_match:
            cliff_months = int(cliff_match.group(1))
        elif cliff_year_match:
            cliff_months = int(cliff_year_match.group(1)) * 12
        elif "one-year cliff" in lower or "1-year cliff" in lower:
            cliff_months = 12
        else:
            cliff_months = 12

        accel = (
            "double_trigger"
            if "double-trigger" in lower
            else "single_trigger"
            if "single-trigger" in lower
            else "none"
        )

        source_span = SourceSpan(text=text.strip(), page_number=clause.page_number)

        return VestingFact(
            clause_id=clause.clause_id,
            total_vesting_years=total_years,
            cliff_months=cliff_months,
            acceleration_type=accel,
            source_span=source_span,
        )

    def extract(self, doc_id: str, clauses: List[ClauseSpan]) -> ExtractedFacts:
        """Extracts facts from all clauses and applies span validation."""
        facts = ExtractedFacts(doc_id=doc_id)

        for clause in clauses:
            family = clause.detected_family

            if family == ClauseFamily.LIQUIDATION_PREFERENCE and not facts.liquidation_preference:
                facts.liquidation_preference = self._extract_liquidation_preference(clause)
            elif family == ClauseFamily.ANTI_DILUTION and not facts.anti_dilution:
                facts.anti_dilution = self._extract_anti_dilution(clause)
            elif family == ClauseFamily.BOARD_COMPOSITION and not facts.board_composition:
                facts.board_composition = self._extract_board_composition(clause)
            elif family == ClauseFamily.FOUNDER_VESTING and not facts.founder_vesting:
                facts.founder_vesting = self._extract_vesting(clause)

        # Apply strict span validation
        validated_facts = self.validator.validate_all(facts)
        return validated_facts
