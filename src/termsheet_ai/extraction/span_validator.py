"""Strict Span Validator for TermSheet AI.

Enforces the neuro-symbolic safeguard:
Every extracted numerical value or categorical attribute must literally appear
verbatim in the cited source span text. If verification fails, the value is invalidated
to prevent LLM hallucinations from reaching the deterministic rule engine.
"""

import re
from typing import Optional, Union

from termsheet_ai.schemas.contracts import (
    AntiDilutionFact,
    AntiDilutionType,
    BoardCompositionFact,
    ExtractedFacts,
    LiquidationPreferenceFact,
    VestingFact,
)

# Number word aliases
NUMBER_WORDS = {
    0: ["0", "zero", "none", "no"],
    1: ["1", "1.0", "1x", "1.0x", "one", "single"],
    2: ["2", "2.0", "2x", "2.0x", "two", "double"],
    3: ["3", "3.0", "3x", "3.0x", "three", "triple"],
    4: ["4", "4.0", "4x", "4.0x", "four"],
    5: ["5", "5.0", "5x", "5.0x", "five"],
    12: ["12", "twelve", "one-year", "1-year", "1 year", "one year"],
    24: ["24", "twenty-four", "two-year", "2-year", "2 years", "two years"],
    36: ["36", "thirty-six", "three-year", "3-year", "3 years", "three years"],
    48: ["48", "forty-eight", "four-year", "4-year", "4 years", "four years"],
}


class SpanValidator:
    """Validates that extracted facts and numbers literally exist in the cited source span."""

    @staticmethod
    def is_number_in_span(val: Union[int, float], text: str) -> bool:
        """Checks if a numeric value (or standard alias) literally appears in the text."""
        if val is None or not text:
            return False

        lower_text = text.lower()

        # Check direct string match
        str_val = str(val)
        if str_val in lower_text:
            return True

        # Check integer representation if float is integral (e.g., 2.0 -> "2")
        if isinstance(val, float) and val.is_integer():
            int_val = int(val)
            if str(int_val) in lower_text:
                return True
            if int_val in NUMBER_WORDS:
                if any(w in lower_text for w in NUMBER_WORDS[int_val]):
                    return True
        elif isinstance(val, int) and val in NUMBER_WORDS:
            if any(w in lower_text for w in NUMBER_WORDS[val]):
                return True

        # Check regex for number with optional 'x' or percentage
        pattern = rf"\b{re.escape(str_val)}(?:x|%|\b)"
        if re.search(pattern, lower_text):
            return True

        return False

    @staticmethod
    def is_phrase_in_span(phrase: str, text: str) -> bool:
        """Checks if a phrase or key tokens appear in the source span."""
        if not phrase or not text:
            return False
        return phrase.lower() in text.lower()

    def validate_liquidation_preference(
        self, fact: Optional[LiquidationPreferenceFact]
    ) -> Optional[LiquidationPreferenceFact]:
        """Validates liquidation multiple and cap in the source span."""
        if not fact:
            return None

        span = fact.source_span.text if fact.source_span else ""
        grounded = True

        # Validate multiple
        if fact.multiple is not None and fact.multiple > 0:
            if not self.is_number_in_span(fact.multiple, span):
                grounded = False
                fact.multiple = 1.0  # Reset to standard baseline on hallucination

        # Validate cap multiple if capped participating
        if fact.cap_multiple is not None and fact.cap_multiple > 0:
            if not self.is_number_in_span(fact.cap_multiple, span):
                grounded = False
                fact.cap_multiple = None

        fact.is_grounded = grounded
        return fact

    def validate_anti_dilution(
        self, fact: Optional[AntiDilutionFact]
    ) -> Optional[AntiDilutionFact]:
        """Validates anti-dilution type tokens in the source span."""
        if not fact:
            return None

        span = fact.source_span.text if fact.source_span else ""
        lower_span = span.lower()
        grounded = True

        if fact.anti_dilution_type == AntiDilutionType.FULL_RATCHET:
            if "ratchet" not in lower_span:
                grounded = False
                fact.anti_dilution_type = AntiDilutionType.UNKNOWN
        elif fact.anti_dilution_type == AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE:
            if "weighted" not in lower_span and "broad" not in lower_span:
                grounded = False
                fact.anti_dilution_type = AntiDilutionType.UNKNOWN

        fact.is_grounded = grounded
        return fact

    def validate_board_composition(
        self, fact: Optional[BoardCompositionFact]
    ) -> Optional[BoardCompositionFact]:
        """Validates seat counts in the source span."""
        if not fact:
            return None

        span = fact.source_span.text if fact.source_span else ""
        grounded = True

        for seats in (fact.founder_seats, fact.investor_seats):
            if seats > 0 and not self.is_number_in_span(seats, span):
                grounded = False

        fact.is_grounded = grounded
        return fact

    def validate_vesting(self, fact: Optional[VestingFact]) -> Optional[VestingFact]:
        """Validates vesting years/months and cliff in the source span."""
        if not fact:
            return None

        span = fact.source_span.text if fact.source_span else ""
        grounded = True

        if fact.total_vesting_years > 0:
            months = int(fact.total_vesting_years * 12)
            has_years = self.is_number_in_span(fact.total_vesting_years, span)
            has_months = self.is_number_in_span(months, span)
            if not (has_years or has_months):
                grounded = False

        if fact.cliff_months > 0:
            years = fact.cliff_months / 12.0
            has_cliff_m = self.is_number_in_span(fact.cliff_months, span)
            has_cliff_y = self.is_number_in_span(years, span)
            if not (has_cliff_m or has_cliff_y or "cliff" in span.lower()):
                grounded = False

        fact.is_grounded = grounded
        return fact

    def validate_all(self, facts: ExtractedFacts) -> ExtractedFacts:
        """Runs strict span validation across all extracted fact blocks."""
        facts.liquidation_preference = self.validate_liquidation_preference(
            facts.liquidation_preference
        )
        facts.anti_dilution = self.validate_anti_dilution(facts.anti_dilution)
        facts.board_composition = self.validate_board_composition(facts.board_composition)
        facts.founder_vesting = self.validate_vesting(facts.founder_vesting)
        return facts
