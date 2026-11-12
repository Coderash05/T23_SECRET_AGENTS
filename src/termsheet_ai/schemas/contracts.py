"""Interface contracts and Pydantic schemas for the TermSheet AI pipeline.

These schemas define the interface contracts between all modules (S1 Ingest, S2 Classify,
S3 Extract, S4 Reason, S5 Explain) to enforce deterministic, verifiable data flow.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class DocumentType(str, Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ClauseFamily(str, Enum):
    """The 12 standard clause families plus 'other' as defined in the project ontology."""

    LIQUIDATION_PREFERENCE = "liquidation_preference"
    ANTI_DILUTION = "anti_dilution"
    BOARD_COMPOSITION = "board_composition"
    PROTECTIVE_PROVISIONS = "protective_provisions"
    FOUNDER_VESTING = "founder_vesting"
    INFORMATION_RIGHTS = "information_rights"
    ROFR_CO_SALE = "rofr_co_sale"
    DRAG_ALONG = "drag_along"
    OPTION_POOL = "option_pool"
    EXCLUSIVITY_NO_SHOP = "exclusivity_no_shop"
    CONVERSION_RIGHTS = "conversion_rights"
    REDEMPTION_RIGHTS = "redemption_rights"
    OTHER = "other"


class SeverityLevel(str, Enum):
    """Risk severity classifications for rule-engine flags."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ParticipationType(str, Enum):
    """Liquidation preference participation modes."""

    NON_PARTICIPATING = "non_participating"
    FULLY_PARTICIPATING = "fully_participating"
    CAPPED_PARTICIPATING = "capped_participating"
    UNKNOWN = "unknown"


class AntiDilutionType(str, Enum):
    """Anti-dilution mechanisms."""

    BROAD_BASED_WEIGHTED_AVERAGE = "broad_based_weighted_average"
    NARROW_BASED_WEIGHTED_AVERAGE = "narrow_based_weighted_average"
    FULL_RATCHET = "full_ratchet"
    NONE = "none"
    UNKNOWN = "unknown"


# --- S1: Ingest & Segmentation Contracts ---


class SourceSpan(BaseModel):
    """Grounding span pointing to the exact character positions and text in source document."""

    text: str = Field(description="Exact verbatim excerpt from the document backing the fact.")
    start_char: Optional[int] = Field(default=None, description="Start character offset in clause.")
    end_char: Optional[int] = Field(default=None, description="End character offset in clause.")
    page_number: Optional[int] = Field(
        default=1, description="1-indexed page number in source document."
    )

    @model_validator(mode="after")
    def validate_non_empty_text(self) -> "SourceSpan":
        if not self.text.strip():
            raise ValueError("Source span text cannot be empty or solely whitespace.")
        return self


class ClauseSpan(BaseModel):
    """Segmented clause contract produced by S1 (Ingest) and labeled by S2 (Classify)."""

    clause_id: str = Field(description="Unique identifier for the clause within the document.")
    raw_text: str = Field(description="Unprocessed verbatim clause text extracted from document.")
    page_number: int = Field(default=1, description="Page number where the clause begins.")
    section_header: Optional[str] = Field(
        default=None, description="Heading or numbered prefix if present."
    )
    detected_family: ClauseFamily = Field(
        default=ClauseFamily.OTHER, description="Assigned clause family from S2 classification."
    )
    classification_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from SBERT k-NN / classifier (0.0 to 1.0).",
    )
    is_unclassified: bool = Field(
        default=False,
        description="True if confidence is below threshold, prompting manual user review.",
    )


class DocumentMetadata(BaseModel):
    """Metadata describing the uploaded term sheet."""

    doc_id: str = Field(description="Unique document hash or identifier.")
    filename: str = Field(description="Original filename.")
    file_type: DocumentType = Field(description="Detected format (pdf, docx, txt).")
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_clauses: int = Field(default=0, ge=0)


# --- S3: Extracted Facts Contracts (with Span Validation) ---


class BaseFact(BaseModel):
    """Base class for all extracted clause attributes, enforcing source span grounding."""

    clause_id: str = Field(description="ID of the clause this fact was extracted from.")
    source_span: SourceSpan = Field(description="Verbatim text span grounding this fact.")
    is_grounded: bool = Field(
        default=True, description="Indicates if source span validator passed."
    )


class LiquidationPreferenceFact(BaseFact):
    """Extracted facts for liquidation preference clauses."""

    multiple: float = Field(
        default=1.0, ge=0.0, description="Liquidation multiple, e.g. 1.0x, 2.0x, 3.0x."
    )
    participation: ParticipationType = Field(
        default=ParticipationType.NON_PARTICIPATING,
        description="Participation mode (market standard is non-participating).",
    )
    cap_multiple: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Participation cap multiple if capped participating (e.g. 3.0x).",
    )
    seniority: Optional[str] = Field(
        default="pari_passu",
        description="Seniority tier (e.g., senior to common, pari passu with Series A).",
    )


class AntiDilutionFact(BaseFact):
    """Extracted facts for anti-dilution provisions."""

    anti_dilution_type: AntiDilutionType = Field(
        default=AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE,
        description="Anti-dilution formula type.",
    )


class BoardCompositionFact(BaseFact):
    """Extracted facts for board of directors composition."""

    founder_seats: int = Field(default=0, ge=0, description="Number of founder-appointed seats.")
    investor_seats: int = Field(default=0, ge=0, description="Number of investor-appointed seats.")
    independent_seats: int = Field(
        default=0, ge=0, description="Number of mutually agreed independent seats."
    )
    total_seats: int = Field(default=0, ge=0, description="Total board seats.")
    investor_majority: bool = Field(
        default=False, description="True if investor seats strictly exceed founder seats."
    )


class VestingFact(BaseFact):
    """Extracted facts for founder/key employee vesting."""

    total_vesting_years: float = Field(
        default=4.0, ge=0.0, description="Total vesting duration (std is 4 years)."
    )
    cliff_months: int = Field(
        default=12, ge=0, description="Cliff duration in months (std is 12 months)."
    )
    acceleration_type: str = Field(
        default="double_trigger",
        description="Acceleration mechanism ('none', 'single_trigger', 'double_trigger').",
    )


class ProtectiveProvisionsFact(BaseFact):
    """Extracted facts for veto rights and negative covenants."""

    requires_series_majority: bool = Field(
        default=True,
        description="Whether major corporate events require separate preferred majority vote.",
    )
    covered_actions: List[str] = Field(
        default_factory=list,
        description="List of restricted corporate actions (e.g. sale, liquidation, new share class).",
    )


class ExtractedFacts(BaseModel):
    """Aggregated container of all validated facts extracted from the term sheet."""

    doc_id: str
    liquidation_preference: Optional[LiquidationPreferenceFact] = None
    anti_dilution: Optional[AntiDilutionFact] = None
    board_composition: Optional[BoardCompositionFact] = None
    founder_vesting: Optional[VestingFact] = None
    protective_provisions: Optional[ProtectiveProvisionsFact] = None
    custom_facts: Dict[str, Any] = Field(default_factory=dict)


# --- S4: Reasoning & Audit Trace Contracts ---


class JustificationRecord(BaseModel):
    """Audit record capturing the proof trace of a derived fact or flag."""

    rule_id: str = Field(
        description="Unique identifier of the Horn rule (e.g., 'RULE_LIQ_MULT_GT_1X')."
    )
    rule_name: str = Field(description="Human-readable rule title.")
    market_norm_cite: str = Field(
        description="Citation to NVCA Model Term Sheet section or empirical VC dataset."
    )
    premise_facts: Dict[str, Any] = Field(
        description="Ground facts that satisfied the rule antecedents."
    )
    conclusion_flag: str = Field(description="Derived conclusion or flag title.")

    # NEW: Nested explanation of how the conclusion was reached.
    proof_tree: Optional[Dict[str, Any]] = None


class RiskFlag(BaseModel):
    """A risk assessment flag raised by the symbolic rule engine."""

    flag_id: str = Field(description="Unique flag identifier.")
    clause_id: str = Field(description="Associated clause ID.")
    title: str = Field(description="Concise description of the identified risk.")
    explanation: str = Field(description="Plain-English explanation of why this term is risky.")
    severity: SeverityLevel = Field(description="Severity rating (low, medium, high, critical).")
    market_standard_source: str = Field(description="Citation to NVCA/market standard.")
    justification: JustificationRecord = Field(description="Audit proof-tree record.")


class DocumentRiskScore(BaseModel):
    """Overall calculated risk metrics for the term sheet."""

    overall_score: float = Field(
        ge=0.0, le=100.0, description="Normalized risk score from 0 (clean) to 100 (high risk)."
    )
    critical_count: int = Field(default=0, ge=0)
    high_count: int = Field(default=0, ge=0)
    medium_count: int = Field(default=0, ge=0)
    low_count: int = Field(default=0, ge=0)
    info_count: int = Field(default=0, ge=0)


class AuditTrace(BaseModel):
    """Full deterministic reasoning report produced by S4."""

    doc_id: str
    risk_score: DocumentRiskScore
    flags: List[RiskFlag] = Field(default_factory=list)
    rule_traces: List[JustificationRecord] = Field(default_factory=list)


# --- S5: Explain & End-to-End Pipeline Payload ---


class ClauseSummary(BaseModel):
    """Plain-English summary of an individual clause with consistency guarantees."""

    clause_id: str
    summary_text: str
    numbers_verified: bool = Field(
        default=True,
        description="True if an automated checker confirmed all numbers match the fact base.",
    )


class PipelinePayload(BaseModel):
    """The complete end-to-end container passed through stages S1 to S5."""

    metadata: DocumentMetadata
    clauses: List[ClauseSpan] = Field(default_factory=list)
    facts: Optional[ExtractedFacts] = None
    audit_trace: Optional[AuditTrace] = None
    summaries: List[ClauseSummary] = Field(default_factory=list)
    execution_errors: List[str] = Field(default_factory=list)
