"""Synthetic Term-Sheet Generator (v0).

Generates realistic term-sheet documents (TXT) by sampling clause attribute
values from configurable distributions and rendering them through paraphrased
surface-text templates.  Each generated document ships with a gold-label JSON
sidecar containing the ground-truth attribute values and expected risk flags,
ready for use in the evaluation harness.

Design decisions
~~~~~~~~~~~~~~~~
* Templates are stored as Python strings (not external files) so the module
  is self-contained with no extra I/O at import time.
* Every template has several surface-text variants to ensure the classifier
  sees lexical diversity, not just numeric variation.
* The generator deliberately produces at least 15 *adversarial* term sheets
  (out of a default batch of 30) to exercise extreme-case rule-engine paths
  such as 3× participating preferred, full-ratchet anti-dilution, and
  investor-majority boards.
* Gold labels follow the same attribute schema used by ``docs/attribute_schema_v0.json``
  and are compatible with ``src/termsheet_ai/schemas/contracts.py`` fact models.

Usage::

    from termsheet_ai.data.generator import SyntheticGenerator

    gen = SyntheticGenerator(seed=42)
    sheets = gen.generate_batch(count=30, adversarial_ratio=0.5)
    gen.write_to_disk(sheets, output_dir="corpus/synthetic")
"""

from __future__ import annotations

import json
import logging
import random
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Term-value distributions
# ---------------------------------------------------------------------------

# Each distribution is a list of (value, weight) tuples.
# "Standard" distributions produce mostly market-norm values.
# "Adversarial" distributions skew toward founder-hostile terms.

STANDARD_LIQUIDATION_MULTIPLES = [(1.0, 70), (1.5, 15), (2.0, 10), (3.0, 5)]
ADVERSARIAL_LIQUIDATION_MULTIPLES = [(2.0, 30), (3.0, 40), (4.0, 20), (5.0, 10)]

STANDARD_PARTICIPATION = [
    ("non_participating", 60),
    ("fully_participating", 20),
    ("capped_participating", 20),
]
ADVERSARIAL_PARTICIPATION = [
    ("fully_participating", 60),
    ("capped_participating", 25),
    ("non_participating", 15),
]

STANDARD_CAP_MULTIPLES = [(2.0, 30), (3.0, 50), (5.0, 20)]
ADVERSARIAL_CAP_MULTIPLES = [(5.0, 30), (8.0, 40), (10.0, 30)]

STANDARD_ANTI_DILUTION = [
    ("broad_based_weighted_average", 75),
    ("narrow_based_weighted_average", 15),
    ("full_ratchet", 10),
]
ADVERSARIAL_ANTI_DILUTION = [
    ("full_ratchet", 60),
    ("narrow_based_weighted_average", 25),
    ("broad_based_weighted_average", 15),
]

STANDARD_BOARD = [
    # (investor_seats, founder_seats, independent_seats)
    ((1, 2, 1), 40),
    ((2, 2, 1), 35),
    ((1, 1, 1), 15),
    ((2, 1, 0), 10),
]
ADVERSARIAL_BOARD = [
    ((3, 1, 0), 40),
    ((3, 2, 0), 25),
    ((2, 1, 0), 20),
    ((4, 1, 0), 15),
]

STANDARD_VESTING = [
    # (total_months, cliff_months)
    ((48, 12), 70),
    ((36, 12), 15),
    ((48, 6), 10),
    ((36, 0), 5),
]
ADVERSARIAL_VESTING = [
    ((48, 18), 30),
    ((48, 24), 30),
    ((60, 12), 20),
    ((24, 12), 20),
]

STANDARD_OPTION_POOL = [(10, 30), (15, 40), (20, 25), (25, 5)]
ADVERSARIAL_OPTION_POOL = [(20, 20), (25, 30), (30, 30), (35, 20)]

STANDARD_EXCLUSIVITY_DAYS = [(30, 40), (45, 40), (60, 15), (90, 5)]
ADVERSARIAL_EXCLUSIVITY_DAYS = [(60, 25), (90, 40), (120, 25), (180, 10)]

STANDARD_DRAG_THRESHOLD = [(50, 20), (67, 50), (75, 20), (90, 10)]
ADVERSARIAL_DRAG_THRESHOLD = [(25, 30), (33, 30), (50, 30), (10, 10)]

STANDARD_REDEMPTION_YEARS = [(5, 40), (7, 35), (10, 20), (3, 5)]
ADVERSARIAL_REDEMPTION_YEARS = [(3, 40), (2, 30), (4, 20), (1, 10)]

STANDARD_DIVIDENDS_CUMULATIVE = [(False, 75), (True, 25)]
ADVERSARIAL_DIVIDENDS_CUMULATIVE = [(True, 70), (False, 30)]

STANDARD_SENIORITY = [("pari_passu", 70), ("senior", 30)]
ADVERSARIAL_SENIORITY = [("senior", 65), ("pari_passu", 35)]

STANDARD_ACCELERATION = [("double_trigger", 50), ("single_trigger", 20), ("none", 30)]
ADVERSARIAL_ACCELERATION = [("none", 60), ("single_trigger", 25), ("double_trigger", 15)]

# Company name parts for generating realistic headers
COMPANY_PREFIXES = [
    "Nova", "Apex", "Vertex", "Quantum", "Helix", "Zephyr", "Cipher", "Prism",
    "Atlas", "Vanguard", "Nexus", "Orion", "Pulsar", "Zenith", "Forge", "Aether",
    "Stratos", "Kyber", "Ionic", "Flux", "Photon", "Cortex", "Nimbus", "Cobalt",
]
COMPANY_SUFFIXES = [
    "Labs", "AI", "Tech", "Robotics", "Bio", "Health", "Systems", "Solutions",
    "Analytics", "Networks", "Cloud", "Data", "Genomics", "Therapeutics", "Energy",
    "Dynamics", "Innovations", "Ventures", "Scientific", "Digital",
]
INVESTOR_NAMES = [
    "Sequoia Capital", "Andreessen Horowitz", "Accel Partners", "Founders Fund",
    "Khosla Ventures", "Lightspeed Venture Partners", "Bessemer Venture Partners",
    "Greylock Partners", "Index Ventures", "General Catalyst", "NEA", "Benchmark",
    "First Round Capital", "Insight Partners", "Tiger Global Management",
    "Ribbit Capital", "Coatue Management", "Thrive Capital",
]
SERIES_NAMES = ["Series Seed", "Series A", "Series A-1", "Series B", "Series C"]


def _weighted_choice(distribution: list, rng: random.Random) -> Any:
    """Pick a value from a (value, weight) distribution using the given RNG."""
    values, weights = zip(*distribution, strict=False)
    return rng.choices(values, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Gold-label data class
# ---------------------------------------------------------------------------

@dataclass
class GoldLabel:
    """Ground-truth attribute values for a synthetic term sheet."""

    doc_id: str
    is_adversarial: bool

    # Liquidation preference
    liq_multiple: float
    liq_participation: str
    liq_cap: Optional[float]  # None when non-participating or uncapped
    liq_seniority: str

    # Anti-dilution
    anti_dilution_type: str

    # Board composition
    board_investor_seats: int
    board_founder_seats: int
    board_independent_seats: int

    # Vesting
    vesting_months: int
    cliff_months: int
    acceleration: str

    # Option pool
    option_pool_pct: int

    # Exclusivity / no-shop
    exclusivity_days: int

    # Drag-along
    drag_threshold_pct: int

    # Redemption
    redemption_years: int

    # Dividends
    dividends_cumulative: bool

    # Expected risk flags (populated after sampling)
    expected_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Surface-text templates (multiple variants per clause family)
# ---------------------------------------------------------------------------

_LIQUIDATION_TEMPLATES = [
    (
        "Liquidation Preference\n"
        "In the event of any liquidation, dissolution, or winding up of the Company, "
        "the holders of Series {series} Preferred Stock shall be entitled to receive, "
        "prior and in preference to any distribution to the holders of Common Stock, "
        "an amount per share equal to {multiple}x the Original Purchase Price "
        "(the \"Liquidation Preference\"). "
        "{participation_text}"
    ),
    (
        "LIQUIDATION PREFERENCE\n"
        "Upon any Liquidation Event, each holder of {series} Preferred shall receive "
        "a per-share amount equal to {multiple} times the Original Issue Price, "
        "plus any declared but unpaid dividends. "
        "{participation_text}"
    ),
    (
        "Section – Liquidation Preference\n"
        "The {series} Preferred Stock shall have a liquidation preference equal to "
        "{multiple}x the Original Purchase Price per share. "
        "{participation_text}"
        "Any remaining proceeds shall be distributed pro rata among holders of Common Stock."
    ),
]

_PARTICIPATION_TEXTS = {
    "non_participating": [
        "The {series} Preferred Stock shall be non-participating.",
        "Holders of {series} Preferred shall not participate in remaining distributions "
        "beyond the Liquidation Preference.",
        "No participation rights beyond the initial Liquidation Preference amount.",
    ],
    "fully_participating": [
        "After receiving the Liquidation Preference, the {series} Preferred shall participate "
        "pro rata with the Common Stock on an as-converted basis, with no cap.",
        "Full participation: following payment of the Liquidation Preference, holders of "
        "{series} Preferred shall share in remaining proceeds alongside Common holders.",
        "The {series} Preferred shall be fully participating preferred stock.",
    ],
    "capped_participating": [
        "After receiving the Liquidation Preference, the {series} Preferred shall participate "
        "pro rata with Common Stock on an as-converted basis up to a total return of "
        "{cap}x the Original Purchase Price (the \"Cap\").",
        "Participation is capped at an aggregate of {cap}x the Original Purchase Price per share.",
        "The {series} Preferred shall participate with Common Stock up to a maximum return "
        "of {cap} times the Original Issue Price, after which they receive no further distributions.",
    ],
}

_ANTI_DILUTION_TEMPLATES = [
    (
        "Anti-Dilution Provisions\n"
        "The Conversion Price of the {series} Preferred Stock shall be subject to "
        "{ad_type_text} anti-dilution adjustment in the event that the Company issues "
        "additional equity securities at a purchase price less than the then-effective "
        "Conversion Price."
    ),
    (
        "ANTI-DILUTION PROTECTION\n"
        "In the event of a down-round issuance, the conversion price of the {series} "
        "Preferred shall be adjusted pursuant to a {ad_type_text} formula."
    ),
    (
        "Section – Price-Based Anti-Dilution\n"
        "The {series} Preferred Stock will have {ad_type_text} anti-dilution protection."
    ),
]

_AD_TYPE_DISPLAY = {
    "broad_based_weighted_average": "broad-based weighted average",
    "narrow_based_weighted_average": "narrow-based weighted average",
    "full_ratchet": "full ratchet",
}

_BOARD_TEMPLATES = [
    (
        "Board of Directors\n"
        "The Board of Directors shall consist of {total} members: "
        "{inv} seat(s) designated by the holders of {series} Preferred Stock, "
        "{fnd} seat(s) designated by the holders of Common Stock{ind_text}."
    ),
    (
        "BOARD COMPOSITION\n"
        "The Company's Board shall be composed of {total} directors: "
        "{inv} appointed by the Investors, {fnd} appointed by the Founders{ind_text}."
    ),
    (
        "Section – Board of Directors\n"
        "The Board shall have {total} seats, allocated as follows: "
        "Investors shall designate {inv} director(s), "
        "Founders shall designate {fnd} director(s){ind_text}."
    ),
]

_VESTING_TEMPLATES = [
    (
        "Founder Vesting\n"
        "All shares held by each Founder shall be subject to vesting over "
        "{years} years ({months} months total), with a {cliff}-month cliff. "
        "Shares shall vest monthly following the cliff on a pro-rata basis. "
        "{accel_text}"
    ),
    (
        "VESTING\n"
        "Founder shares will vest over a period of {years} years with a "
        "{cliff}-month cliff period. After the cliff, {monthly_pct:.2f}% of the "
        "remaining shares shall vest each month. "
        "{accel_text}"
    ),
    (
        "Section – Founder Share Vesting\n"
        "{months} months total vesting schedule. "
        "Cliff: {cliff} months. Monthly vesting thereafter. "
        "{accel_text}"
    ),
]

_ACCELERATION_TEXTS = {
    "double_trigger": [
        "Double-trigger acceleration: 100% of unvested shares shall accelerate upon a "
        "Change of Control followed by termination without Cause within 12 months.",
        "Acceleration shall be double-trigger, requiring both a Change of Control event "
        "and involuntary termination.",
    ],
    "single_trigger": [
        "Single-trigger acceleration: all unvested shares shall immediately vest upon "
        "a Change of Control.",
        "Upon a Change of Control event, 100% of unvested Founder shares shall accelerate.",
    ],
    "none": [
        "No acceleration of vesting upon Change of Control.",
        "Vesting shall not accelerate under any circumstances.",
    ],
}

_OPTION_POOL_TEMPLATES = [
    (
        "Option Pool\n"
        "Prior to the closing of the financing, the Company shall reserve "
        "{pct}% of its post-money fully-diluted capitalization for issuance "
        "to directors, officers, employees, and consultants under the Company's "
        "equity incentive plan."
    ),
    (
        "EMPLOYEE OPTION POOL\n"
        "An option pool equal to {pct}% of the post-money fully-diluted shares "
        "shall be established prior to closing."
    ),
    (
        "Section – Equity Incentive Pool\n"
        "The Company's Option Pool shall be set at {pct}% of the fully-diluted "
        "post-money capitalization, to be created prior to the Closing."
    ),
]

_EXCLUSIVITY_TEMPLATES = [
    (
        "Exclusivity / No-Shop\n"
        "The Company agrees that for a period of {days} days following the "
        "execution of this Term Sheet, it shall not solicit, encourage, or "
        "accept any proposal from any other party for the financing of the Company."
    ),
    (
        "NO-SHOP AGREEMENT\n"
        "For {days} days from execution of this term sheet, the Company and the "
        "Founders shall not directly or indirectly solicit, initiate, or encourage "
        "any competing financing proposal."
    ),
    (
        "Section – Exclusivity Period\n"
        "The Founders agree to an exclusivity period of {days} days during which "
        "no competing offers shall be sought or entertained."
    ),
]

_DRAG_ALONG_TEMPLATES = [
    (
        "Drag-Along Agreement\n"
        "If holders of at least {threshold}% of the outstanding shares approve a "
        "Sale of the Company, all other shareholders shall be required to vote in "
        "favor of such transaction and take all necessary actions to consummate it."
    ),
    (
        "DRAG-ALONG RIGHTS\n"
        "A Sale of the Company approved by {threshold}% or more of the outstanding "
        "shares shall be binding on all shareholders."
    ),
    (
        "Section – Drag-Along\n"
        "Shareholders holding at least {threshold}% of the shares may compel all "
        "remaining shareholders to participate in an approved Sale of the Company."
    ),
]

_REDEMPTION_TEMPLATES = [
    (
        "Redemption Rights\n"
        "At any time after {years} years from the date of issuance, upon the "
        "request of the holders of at least a majority of the then-outstanding "
        "{series} Preferred Stock, the Company shall redeem all outstanding shares "
        "at the Original Purchase Price plus any declared but unpaid dividends."
    ),
    (
        "REDEMPTION\n"
        "Beginning {years} years after closing, the holders of {series} Preferred "
        "may require the Company to redeem their shares at the Original Issue Price."
    ),
    (
        "Section – Mandatory Redemption\n"
        "The {series} Preferred Stock is redeemable at the option of the holders "
        "commencing {years} years from the Closing Date."
    ),
]

_DIVIDENDS_TEMPLATES = [
    (
        "Dividends\n"
        "The holders of {series} Preferred Stock shall be entitled to receive "
        "{div_type} dividends at a rate of 8% per annum on the Original Purchase Price. "
        "Dividends on Common Stock may not be paid unless all accrued Preferred dividends "
        "have been paid."
    ),
    (
        "DIVIDEND RIGHTS\n"
        "The {series} Preferred shall accrue {div_type} dividends at 8% of the "
        "Original Issue Price, payable when and if declared by the Board."
    ),
    (
        "Section – Dividends\n"
        "Dividends shall be {div_type} at a rate of 8% per annum on the {series} "
        "Preferred Stock. No dividends shall be paid on Common Stock until all "
        "Preferred dividends are satisfied."
    ),
]

_PROTECTIVE_PROVISIONS_TEMPLATES = [
    (
        "Protective Provisions\n"
        "So long as any shares of {series} Preferred Stock are outstanding, the Company "
        "shall not, without the prior written consent of the holders of at least a majority "
        "of the {series} Preferred Stock: (i) alter or change the rights of the Preferred Stock; "
        "(ii) increase or decrease the authorized number of shares; (iii) create any new class or "
        "series of shares having rights senior to or on parity with the Preferred Stock; "
        "(iv) declare or pay any dividend on Common Stock; (v) approve any merger, sale of "
        "substantially all assets, or other liquidation event; or (vi) increase the size of "
        "the Board of Directors."
    ),
    (
        "PROTECTIVE PROVISIONS\n"
        "The consent of the majority of {series} Preferred holders shall be required for: "
        "amendments to the Charter affecting the Preferred Stock; issuance of equity securities "
        "senior to the {series} Preferred; any Liquidation Event or Change of Control; "
        "changes to the Board size; and declaration of dividends on Common Stock."
    ),
]

_INFORMATION_RIGHTS_TEMPLATES = [
    (
        "Information Rights\n"
        "The Company shall deliver to each Major Investor (i) annual audited financial "
        "statements within 120 days after the end of each fiscal year, (ii) unaudited "
        "quarterly financial statements within 45 days after the end of each quarter, "
        "and (iii) an annual budget and business plan within 30 days prior to the start "
        "of each fiscal year."
    ),
    (
        "INFORMATION AND INSPECTION RIGHTS\n"
        "Major Investors shall receive annual audited financials, quarterly unaudited "
        "financials, monthly management reports, and an annual operating budget."
    ),
]

_ROFR_CO_SALE_TEMPLATES = [
    (
        "Right of First Refusal / Co-Sale\n"
        "The Company shall have a right of first refusal on any proposed transfer of shares "
        "by a Founder. If the Company does not exercise its right, the Investors holding "
        "{series} Preferred Stock shall have a secondary right of first refusal, followed "
        "by a co-sale right allowing participation in the transfer pro rata."
    ),
    (
        "ROFR AND CO-SALE AGREEMENT\n"
        "Founders may not transfer shares without first offering them to the Company, then to "
        "the Investors. If the transfer proceeds, Investors have co-sale rights to participate "
        "on a pro-rata basis."
    ),
]

_CONVERSION_TEMPLATES = [
    (
        "Conversion Rights\n"
        "Each share of {series} Preferred Stock shall be convertible, at the option of "
        "the holder, into shares of Common Stock at the then-applicable Conversion Rate "
        "(initially 1:1, subject to anti-dilution adjustments). All {series} Preferred Stock "
        "shall automatically convert into Common Stock upon (i) an IPO with aggregate proceeds "
        "of at least $50,000,000 or (ii) the consent of a majority of the Preferred holders."
    ),
    (
        "CONVERSION\n"
        "The {series} Preferred converts to Common Stock at a 1:1 ratio (as adjusted). "
        "Automatic conversion occurs upon a Qualified IPO or majority Preferred consent."
    ),
]

# Boilerplate "OTHER" sections that always appear
_BOILERPLATE_SECTIONS = [
    (
        "Governing Law\n"
        "This Term Sheet shall be governed by and construed in accordance with the laws "
        "of the State of Delaware, without regard to its conflict of laws provisions."
    ),
    (
        "Expenses\n"
        "The Company shall pay the reasonable legal fees and expenses of counsel to "
        "the Investors in connection with the financing, up to a maximum of $25,000."
    ),
    (
        "Binding Provisions\n"
        "This Term Sheet is non-binding, except for the provisions relating to "
        "Confidentiality, Exclusivity, and Expenses, which shall be binding agreements "
        "of the parties."
    ),
    (
        "Conditions Precedent\n"
        "The closing of the financing shall be subject to customary conditions, including: "
        "completion of due diligence satisfactory to the Investors; execution of definitive "
        "agreements; delivery of legal opinions; and all requisite corporate approvals."
    ),
    (
        "Confidentiality\n"
        "The existence and terms of this Term Sheet shall be treated as confidential "
        "information by all parties and shall not be disclosed to any third party without "
        "the prior written consent of the other party, except as required by law."
    ),
]


# ---------------------------------------------------------------------------
# Expected-flag computation (mirrors rules_v1.yaml)
# ---------------------------------------------------------------------------

def _compute_expected_flags(label: GoldLabel) -> List[str]:
    """Derive the list of expected rule-engine flag IDs from gold-label values.

    This mirrors the logic in ``docs/rules_v1.yaml`` so each synthetic sheet
    ships with pre-computed expected flags for evaluation.
    """
    flags: List[str] = []

    # R-LP-01: multiple > 1
    if label.liq_multiple > 1:
        flags.append("R-LP-01")

    # R-LP-02: participation == true (i.e. fully or capped participating)
    if label.liq_participation in ("fully_participating", "capped_participating"):
        flags.append("R-LP-02")

    # R-LP-03: seniority == 'senior'
    if label.liq_seniority == "senior":
        flags.append("R-LP-03")

    # R-AD-01: full_ratchet
    if label.anti_dilution_type == "full_ratchet":
        flags.append("R-AD-01")

    # R-AD-02: broad_based_weighted_average (informational, low severity)
    if label.anti_dilution_type == "broad_based_weighted_average":
        flags.append("R-AD-02")

    # R-CT-02: investor_seats > founder_seats
    if label.board_investor_seats > label.board_founder_seats:
        flags.append("R-CT-02")

    # R-CT-03: independent_seats == 0
    if label.board_independent_seats == 0:
        flags.append("R-CT-03")

    # R-VS-01: vesting_months < 48
    if label.vesting_months < 48:
        flags.append("R-VS-01")

    # R-VS-02: cliff_months > 12
    if label.cliff_months > 12:
        flags.append("R-VS-02")

    # R-OP-01: option_pool > 20%
    if label.option_pool_pct > 20:
        flags.append("R-OP-01")

    # R-EX-01: exclusivity > 45 days
    if label.exclusivity_days > 45:
        flags.append("R-EX-01")

    # R-DA-01: drag_threshold < 50%
    if label.drag_threshold_pct < 50:
        flags.append("R-DA-01")

    # R-RD-01: redemption < 5 years
    if label.redemption_years < 5:
        flags.append("R-RD-01")

    # R-DV-01: cumulative dividends
    if label.dividends_cumulative:
        flags.append("R-DV-01")

    return flags


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSheet:
    """A generated term-sheet document with its gold-label sidecar."""

    doc_id: str
    filename: str
    text: str
    gold_label: GoldLabel


class SyntheticGenerator:
    """Generates synthetic term-sheet documents with gold-label sidecars.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    # -- Sampling helpers --------------------------------------------------

    def _sample_company_name(self) -> str:
        prefix = self.rng.choice(COMPANY_PREFIXES)
        suffix = self.rng.choice(COMPANY_SUFFIXES)
        return f"{prefix} {suffix}"

    def _sample_investor(self) -> str:
        return self.rng.choice(INVESTOR_NAMES)

    def _sample_series(self) -> str:
        return self.rng.choice(SERIES_NAMES)

    def _sample_valuation(self) -> tuple[float, float]:
        """Return (pre_money_millions, raise_millions)."""
        pre = self.rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 80, 100])
        raise_amt = round(pre * self.rng.uniform(0.15, 0.5), 1)
        return float(pre), float(raise_amt)

    # -- Template rendering ------------------------------------------------

    def _render_template(self, templates: list, **kwargs: Any) -> str:
        template = self.rng.choice(templates)
        return template.format(**kwargs)

    def _render_liquidation(self, series: str, label: GoldLabel) -> str:
        if label.liq_participation == "non_participating":
            part_text = self.rng.choice(_PARTICIPATION_TEXTS["non_participating"]).format(
                series=series
            )
        elif label.liq_participation == "fully_participating":
            part_text = self.rng.choice(_PARTICIPATION_TEXTS["fully_participating"]).format(
                series=series
            )
        else:
            part_text = self.rng.choice(_PARTICIPATION_TEXTS["capped_participating"]).format(
                series=series, cap=label.liq_cap
            )
        return self._render_template(
            _LIQUIDATION_TEMPLATES,
            series=series,
            multiple=label.liq_multiple,
            participation_text=part_text,
        )

    def _render_anti_dilution(self, series: str, label: GoldLabel) -> str:
        ad_text = _AD_TYPE_DISPLAY[label.anti_dilution_type]
        return self._render_template(
            _ANTI_DILUTION_TEMPLATES,
            series=series,
            ad_type_text=ad_text,
        )

    def _render_board(self, series: str, label: GoldLabel) -> str:
        total = label.board_investor_seats + label.board_founder_seats + label.board_independent_seats
        ind_text = ""
        if label.board_independent_seats > 0:
            ind_text = (
                f", and {label.board_independent_seats} independent director(s) "
                f"mutually agreed upon by the Common and Preferred holders"
            )
        return self._render_template(
            _BOARD_TEMPLATES,
            series=series,
            total=total,
            inv=label.board_investor_seats,
            fnd=label.board_founder_seats,
            ind_text=ind_text,
        )

    def _render_vesting(self, label: GoldLabel) -> str:
        years = label.vesting_months / 12
        # Monthly vesting percentage after cliff
        remaining_months = label.vesting_months - label.cliff_months
        monthly_pct = 100.0 / label.vesting_months if remaining_months > 0 else 0.0
        accel_text = self.rng.choice(_ACCELERATION_TEXTS[label.acceleration])
        return self._render_template(
            _VESTING_TEMPLATES,
            years=years if years == int(years) else f"{years:.1f}",
            months=label.vesting_months,
            cliff=label.cliff_months,
            monthly_pct=monthly_pct,
            accel_text=accel_text,
        )

    def _render_option_pool(self, label: GoldLabel) -> str:
        return self._render_template(
            _OPTION_POOL_TEMPLATES,
            pct=label.option_pool_pct,
        )

    def _render_exclusivity(self, label: GoldLabel) -> str:
        return self._render_template(
            _EXCLUSIVITY_TEMPLATES,
            days=label.exclusivity_days,
        )

    def _render_drag_along(self, label: GoldLabel) -> str:
        return self._render_template(
            _DRAG_ALONG_TEMPLATES,
            threshold=label.drag_threshold_pct,
        )

    def _render_redemption(self, series: str, label: GoldLabel) -> str:
        return self._render_template(
            _REDEMPTION_TEMPLATES,
            series=series,
            years=label.redemption_years,
        )

    def _render_dividends(self, series: str, label: GoldLabel) -> str:
        div_type = "cumulative" if label.dividends_cumulative else "non-cumulative"
        return self._render_template(
            _DIVIDENDS_TEMPLATES,
            series=series,
            div_type=div_type,
        )

    def _render_protective_provisions(self, series: str) -> str:
        return self.rng.choice(_PROTECTIVE_PROVISIONS_TEMPLATES).format(series=series)

    def _render_information_rights(self) -> str:
        return self.rng.choice(_INFORMATION_RIGHTS_TEMPLATES)

    def _render_rofr(self, series: str) -> str:
        return self.rng.choice(_ROFR_CO_SALE_TEMPLATES).format(series=series)

    def _render_conversion(self, series: str) -> str:
        return self.rng.choice(_CONVERSION_TEMPLATES).format(series=series)

    def _render_boilerplate(self) -> List[str]:
        """Pick 2-4 boilerplate sections at random."""
        k = self.rng.randint(2, min(4, len(_BOILERPLATE_SECTIONS)))
        return self.rng.sample(_BOILERPLATE_SECTIONS, k)

    # -- Core generation ---------------------------------------------------

    def _sample_label(self, doc_id: str, adversarial: bool) -> GoldLabel:
        """Sample all term-sheet attribute values from the appropriate distributions."""
        if adversarial:
            liq_mult = _weighted_choice(ADVERSARIAL_LIQUIDATION_MULTIPLES, self.rng)
            liq_part = _weighted_choice(ADVERSARIAL_PARTICIPATION, self.rng)
            ad_type = _weighted_choice(ADVERSARIAL_ANTI_DILUTION, self.rng)
            board = _weighted_choice(ADVERSARIAL_BOARD, self.rng)
            vesting = _weighted_choice(ADVERSARIAL_VESTING, self.rng)
            pool = _weighted_choice(ADVERSARIAL_OPTION_POOL, self.rng)
            excl = _weighted_choice(ADVERSARIAL_EXCLUSIVITY_DAYS, self.rng)
            drag = _weighted_choice(ADVERSARIAL_DRAG_THRESHOLD, self.rng)
            redeem = _weighted_choice(ADVERSARIAL_REDEMPTION_YEARS, self.rng)
            divs = _weighted_choice(ADVERSARIAL_DIVIDENDS_CUMULATIVE, self.rng)
            seniority = _weighted_choice(ADVERSARIAL_SENIORITY, self.rng)
            accel = _weighted_choice(ADVERSARIAL_ACCELERATION, self.rng)
        else:
            liq_mult = _weighted_choice(STANDARD_LIQUIDATION_MULTIPLES, self.rng)
            liq_part = _weighted_choice(STANDARD_PARTICIPATION, self.rng)
            ad_type = _weighted_choice(STANDARD_ANTI_DILUTION, self.rng)
            board = _weighted_choice(STANDARD_BOARD, self.rng)
            vesting = _weighted_choice(STANDARD_VESTING, self.rng)
            pool = _weighted_choice(STANDARD_OPTION_POOL, self.rng)
            excl = _weighted_choice(STANDARD_EXCLUSIVITY_DAYS, self.rng)
            drag = _weighted_choice(STANDARD_DRAG_THRESHOLD, self.rng)
            redeem = _weighted_choice(STANDARD_REDEMPTION_YEARS, self.rng)
            divs = _weighted_choice(STANDARD_DIVIDENDS_CUMULATIVE, self.rng)
            seniority = _weighted_choice(STANDARD_SENIORITY, self.rng)
            accel = _weighted_choice(STANDARD_ACCELERATION, self.rng)

        # Cap only relevant for capped_participating
        cap = None
        if liq_part == "capped_participating":
            cap_dist = ADVERSARIAL_CAP_MULTIPLES if adversarial else STANDARD_CAP_MULTIPLES
            cap = _weighted_choice(cap_dist, self.rng)

        label = GoldLabel(
            doc_id=doc_id,
            is_adversarial=adversarial,
            liq_multiple=liq_mult,
            liq_participation=liq_part,
            liq_cap=cap,
            liq_seniority=seniority,
            anti_dilution_type=ad_type,
            board_investor_seats=board[0],
            board_founder_seats=board[1],
            board_independent_seats=board[2],
            vesting_months=vesting[0],
            cliff_months=vesting[1],
            acceleration=accel,
            option_pool_pct=pool,
            exclusivity_days=excl,
            drag_threshold_pct=drag,
            redemption_years=redeem,
            dividends_cumulative=divs,
        )
        label.expected_flags = _compute_expected_flags(label)
        return label

    def generate_one(self, index: int, adversarial: bool) -> SyntheticSheet:
        """Generate a single synthetic term sheet.

        Args:
            index: Sequential index (used to build the document ID and filename).
            adversarial: If True, sample from founder-hostile distributions.

        Returns:
            A ``SyntheticSheet`` containing the rendered text and gold labels.
        """
        tag = "adv" if adversarial else "std"
        doc_id = f"synth_{tag}_{index:03d}"
        filename = f"ts_synth_{tag}_{index:03d}.txt"

        label = self._sample_label(doc_id, adversarial)

        company = self._sample_company_name()
        investor = self._sample_investor()
        series = self._sample_series()
        pre_money, raise_amt = self._sample_valuation()

        # -- Assemble the full document --
        sections: List[str] = []

        # Header
        header = textwrap.dedent(f"""\
            {'=' * 72}
            TERM SHEET
            FOR {series.upper()} PREFERRED STOCK FINANCING OF
            {company.upper()}, INC.

            Date: {self.rng.randint(1, 28)} {self.rng.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} {self.rng.randint(2022, 2026)}

            Issuer:           {company}, Inc. (the "Company")
            Investors:        {investor} (the "Lead Investor")
            Series:           {series} Preferred Stock
            Pre-Money Valuation:  ${pre_money:,.1f} million
            Amount of Financing:  ${raise_amt:,.1f} million
            {'=' * 72}

            This term sheet summarizes the principal terms of the proposed financing.
            Except for the binding provisions set forth below, this term sheet is
            non-binding and is intended solely as a basis for further discussion.
            {'─' * 72}
        """)
        sections.append(header)

        # Core clauses (in standard term-sheet order)
        sections.append(self._render_liquidation(series, label))
        sections.append(self._render_anti_dilution(series, label))
        sections.append(self._render_board(series, label))
        sections.append(self._render_vesting(label))
        sections.append(self._render_dividends(series, label))
        sections.append(self._render_protective_provisions(series))
        sections.append(self._render_option_pool(label))
        sections.append(self._render_information_rights())
        sections.append(self._render_rofr(series))
        sections.append(self._render_conversion(series))
        sections.append(self._render_drag_along(label))
        sections.append(self._render_redemption(series, label))
        sections.append(self._render_exclusivity(label))

        # Boilerplate
        for bp in self._render_boilerplate():
            sections.append(bp)

        # Join with visual separators
        full_text = ("\n\n" + "─" * 72 + "\n\n").join(sections)

        return SyntheticSheet(
            doc_id=doc_id,
            filename=filename,
            text=full_text,
            gold_label=label,
        )

    def generate_batch(
        self,
        count: int = 30,
        adversarial_ratio: float = 0.5,
    ) -> List[SyntheticSheet]:
        """Generate a batch of synthetic term sheets.

        Args:
            count: Total number of documents to generate.
            adversarial_ratio: Fraction of adversarial (founder-hostile) sheets.
                Must be between 0.0 and 1.0. Default 0.5 ensures at least
                15 adversarial sheets in a batch of 30.

        Returns:
            List of ``SyntheticSheet`` objects.
        """
        n_adversarial = max(1, int(count * adversarial_ratio))
        n_standard = count - n_adversarial

        sheets: List[SyntheticSheet] = []

        for i in range(1, n_adversarial + 1):
            sheets.append(self.generate_one(index=i, adversarial=True))

        for i in range(1, n_standard + 1):
            sheets.append(self.generate_one(index=i, adversarial=False))

        # Shuffle so adversarial and standard are interleaved
        self.rng.shuffle(sheets)

        logger.info(
            "Generated %d synthetic term sheets (%d adversarial, %d standard)",
            len(sheets),
            n_adversarial,
            n_standard,
        )
        return sheets

    # -- Disk I/O ----------------------------------------------------------

    def write_to_disk(
        self,
        sheets: List[SyntheticSheet],
        output_dir: str | Path = "corpus/synthetic",
    ) -> Path:
        """Write generated sheets and gold-label sidecars to disk.

        Creates ``output_dir`` if it does not exist.  Each sheet produces:
        - ``<filename>.txt``  — the rendered term-sheet text.
        - ``<filename>.gold.json`` — the gold-label sidecar.

        Also writes a combined ``_batch_manifest.json`` summarizing the batch.

        Args:
            sheets: List of ``SyntheticSheet`` objects from :meth:`generate_batch`.
            output_dir: Target directory.

        Returns:
            Path to the output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        manifest_entries: List[Dict[str, Any]] = []

        for sheet in sheets:
            # Write the term-sheet text
            txt_path = out / sheet.filename
            txt_path.write_text(sheet.text, encoding="utf-8")

            # Write the gold-label sidecar JSON
            gold_path = out / (txt_path.stem + ".gold.json")
            gold_dict = asdict(sheet.gold_label)
            gold_path.write_text(
                json.dumps(gold_dict, indent=2, default=str),
                encoding="utf-8",
            )

            manifest_entries.append({
                "doc_id": sheet.doc_id,
                "filename": sheet.filename,
                "gold_label_file": gold_path.name,
                "is_adversarial": sheet.gold_label.is_adversarial,
                "expected_flag_count": len(sheet.gold_label.expected_flags),
                "expected_flags": sheet.gold_label.expected_flags,
            })

            logger.debug("Wrote %s + %s", txt_path.name, gold_path.name)

        # Write batch manifest
        batch_manifest = {
            "_meta": {
                "description": "Synthetic term-sheet batch manifest with gold labels.",
                "total_count": len(sheets),
                "adversarial_count": sum(1 for s in sheets if s.gold_label.is_adversarial),
                "standard_count": sum(1 for s in sheets if not s.gold_label.is_adversarial),
            },
            "documents": manifest_entries,
        }
        manifest_path = out / "_batch_manifest.json"
        manifest_path.write_text(
            json.dumps(batch_manifest, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info("Batch written to %s (%d files)", out, len(sheets))
        return out
