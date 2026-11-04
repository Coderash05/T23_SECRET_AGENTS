"""End-to-End TermSheet AI Pipeline Runner.

Orchestrates stages S1 through S5 while supporting modular dependency injection
of custom components as they are completed throughout the 6-week timeline.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional, Union

from termsheet_ai.pipeline.base import (
    BaseClassifier,
    BaseExplainer,
    BaseExtractor,
    BaseIngestor,
    BaseReasoner,
)
from termsheet_ai.schemas.contracts import (
    AntiDilutionFact,
    AntiDilutionType,
    AuditTrace,
    BoardCompositionFact,
    ClauseFamily,
    ClauseSpan,
    ClauseSummary,
    DocumentMetadata,
    DocumentRiskScore,
    DocumentType,
    ExtractedFacts,
    JustificationRecord,
    LiquidationPreferenceFact,
    ParticipationType,
    PipelinePayload,
    RiskFlag,
    SeverityLevel,
    SourceSpan,
)


class MockIngestor(BaseIngestor):
    """Fallback text segmenter reading plaintext or simulating extraction."""

    def ingest(self, file_path: Union[str, Path]) -> tuple[DocumentMetadata, List[ClauseSpan]]:
        path = Path(file_path)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(path.resolve())))

        ext = path.suffix.lower().replace(".", "")
        try:
            doc_type = DocumentType(ext)
        except ValueError:
            doc_type = DocumentType.TXT

        content = ""
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""

        if not content.strip():
            content = (
                "1. Liquidation Preference: Series A Preferred will receive a 2.0x liquidation preference, "
                "participating fully in any remaining proceeds.\n"
                "2. Anti-Dilution: The Series A Preferred shares shall feature full-ratchet anti-dilution protection.\n"
                "3. Board of Directors: The Board will consist of 3 members, with 2 seats designated by Investors."
            )

        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        clauses = []
        for i, text in enumerate(paragraphs, start=1):
            clauses.append(
                ClauseSpan(
                    clause_id=f"c_{i}",
                    raw_text=text,
                    page_number=1,
                    section_header=text.split(":")[0] if ":" in text else None,
                )
            )

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=path.name,
            file_type=doc_type,
            total_clauses=len(clauses),
        )
        return metadata, clauses


class MockClassifier(BaseClassifier):
    """Heuristic keyword classifier serving as a baseline prior to SBERT model completion."""

    def classify(self, clauses: List[ClauseSpan]) -> List[ClauseSpan]:
        for clause in clauses:
            lower = clause.raw_text.lower()
            if "liquidation preference" in lower or "participating" in lower:
                clause.detected_family = ClauseFamily.LIQUIDATION_PREFERENCE
                clause.classification_confidence = 0.95
            elif "anti-dilution" in lower or "ratchet" in lower:
                clause.detected_family = ClauseFamily.ANTI_DILUTION
                clause.classification_confidence = 0.92
            elif "board" in lower or "director" in lower:
                clause.detected_family = ClauseFamily.BOARD_COMPOSITION
                clause.classification_confidence = 0.90
            elif "vesting" in lower or "cliff" in lower:
                clause.detected_family = ClauseFamily.FOUNDER_VESTING
                clause.classification_confidence = 0.88
            elif "protective provisions" in lower or "veto" in lower:
                clause.detected_family = ClauseFamily.PROTECTIVE_PROVISIONS
                clause.classification_confidence = 0.85
            else:
                clause.detected_family = ClauseFamily.OTHER
                clause.classification_confidence = 0.50
        return clauses


class MockExtractor(BaseExtractor):
    """Baseline rule-based extractor producing validated facts from classified clauses."""

    def extract(self, doc_id: str, clauses: List[ClauseSpan]) -> ExtractedFacts:
        facts = ExtractedFacts(doc_id=doc_id)

        for clause in clauses:
            text = clause.raw_text
            lower = text.lower()

            if clause.detected_family == ClauseFamily.LIQUIDATION_PREFERENCE:
                multiple = 2.0 if "2.0x" in lower or "2x" in lower else 1.0
                participation = (
                    ParticipationType.FULLY_PARTICIPATING
                    if "participating fully" in lower or "full participation" in lower
                    else ParticipationType.NON_PARTICIPATING
                )
                facts.liquidation_preference = LiquidationPreferenceFact(
                    clause_id=clause.clause_id,
                    multiple=multiple,
                    participation=participation,
                    source_span=SourceSpan(text=text[:100], page_number=clause.page_number),
                )

            elif clause.detected_family == ClauseFamily.ANTI_DILUTION:
                ad_type = (
                    AntiDilutionType.FULL_RATCHET
                    if "full-ratchet" in lower or "full ratchet" in lower
                    else AntiDilutionType.BROAD_BASED_WEIGHTED_AVERAGE
                )
                facts.anti_dilution = AntiDilutionFact(
                    clause_id=clause.clause_id,
                    anti_dilution_type=ad_type,
                    source_span=SourceSpan(text=text[:100], page_number=clause.page_number),
                )

            elif clause.detected_family == ClauseFamily.BOARD_COMPOSITION:
                facts.board_composition = BoardCompositionFact(
                    clause_id=clause.clause_id,
                    founder_seats=1,
                    investor_seats=2,
                    total_seats=3,
                    investor_majority=True,
                    source_span=SourceSpan(text=text[:100], page_number=clause.page_number),
                )

        return facts


class MockReasoner(BaseReasoner):
    """Baseline forward chainer evaluating market norms against extracted facts."""

    def reason(self, facts: ExtractedFacts) -> AuditTrace:
        flags: List[RiskFlag] = []
        rule_traces: List[JustificationRecord] = []
        severity_weights = {
            SeverityLevel.INFO: 0.0,
            SeverityLevel.LOW: 5.0,
            SeverityLevel.MEDIUM: 15.0,
            SeverityLevel.HIGH: 30.0,
            SeverityLevel.CRITICAL: 50.0,
        }
        total_risk = 0.0

        # Rule 1: Liquidation multiple > 1.0x
        if facts.liquidation_preference and facts.liquidation_preference.multiple > 1.0:
            rec = JustificationRecord(
                rule_id="RULE_LIQ_MULT_GT_1X",
                rule_name="Non-Standard Liquidation Multiple",
                market_norm_cite="NVCA Model Term Sheet §3: Standard is 1.0x Non-Participating",
                premise_facts={"multiple": facts.liquidation_preference.multiple},
                conclusion_flag="High Liquidation Multiple",
            )
            flag = RiskFlag(
                flag_id=f"flag_{uuid.uuid4().hex[:6]}",
                clause_id=facts.liquidation_preference.clause_id,
                title="Excessive Liquidation Multiple",
                explanation=f"Founders face {facts.liquidation_preference.multiple}x payout before common stock receives any distribution.",
                severity=SeverityLevel.HIGH,
                market_standard_source="NVCA Model Term Sheet (2020 revision)",
                justification=rec,
            )
            flags.append(flag)
            rule_traces.append(rec)
            total_risk += severity_weights[flag.severity]

        # Rule 2: Full Ratchet Anti-Dilution
        if (
            facts.anti_dilution
            and facts.anti_dilution.anti_dilution_type == AntiDilutionType.FULL_RATCHET
        ):
            rec = JustificationRecord(
                rule_id="RULE_AD_FULL_RATCHET",
                rule_name="Full Ratchet Anti-Dilution",
                market_norm_cite="NVCA Model Term Sheet §4: Broad-Based Weighted Average is Standard",
                premise_facts={"anti_dilution_type": facts.anti_dilution.anti_dilution_type.value},
                conclusion_flag="Severe Dilution Risk (Full Ratchet)",
            )
            flag = RiskFlag(
                flag_id=f"flag_{uuid.uuid4().hex[:6]}",
                clause_id=facts.anti_dilution.clause_id,
                title="Full-Ratchet Anti-Dilution Mechanism",
                explanation="Full ratchet severely dilutes founders during any subsequent down-round regardless of amount raised.",
                severity=SeverityLevel.CRITICAL,
                market_standard_source="Empirical VC Deal Terms (Gompers et al.)",
                justification=rec,
            )
            flags.append(flag)
            rule_traces.append(rec)
            total_risk += severity_weights[flag.severity]

        score = DocumentRiskScore(
            overall_score=min(100.0, total_risk),
            critical_count=sum(1 for f in flags if f.severity == SeverityLevel.CRITICAL),
            high_count=sum(1 for f in flags if f.severity == SeverityLevel.HIGH),
            medium_count=sum(1 for f in flags if f.severity == SeverityLevel.MEDIUM),
            low_count=sum(1 for f in flags if f.severity == SeverityLevel.LOW),
            info_count=sum(1 for f in flags if f.severity == SeverityLevel.INFO),
        )

        return AuditTrace(
            doc_id=facts.doc_id,
            risk_score=score,
            flags=flags,
            rule_traces=rule_traces,
        )


class MockExplainer(BaseExplainer):
    """Generates plain-English summaries cross-verified with extracted facts."""

    def explain(self, clauses: List[ClauseSpan], audit_trace: AuditTrace) -> List[ClauseSummary]:
        summaries = []
        flagged_clauses = {f.clause_id: f for f in audit_trace.flags}

        for clause in clauses:
            flag = flagged_clauses.get(clause.clause_id)
            if flag:
                text = f"Summary: This {clause.detected_family.value} clause has been flagged as {flag.severity.value.upper()}: {flag.explanation}"
            else:
                text = f"Summary: Standard {clause.detected_family.value} clause consistent with market norms."

            summaries.append(
                ClauseSummary(
                    clause_id=clause.clause_id,
                    summary_text=text,
                    numbers_verified=True,
                )
            )
        return summaries


class TermSheetPipeline:
    """The master pipeline orchestrating S1 through S5."""

    def __init__(
        self,
        ingestor: Optional[BaseIngestor] = None,
        classifier: Optional[BaseClassifier] = None,
        extractor: Optional[BaseExtractor] = None,
        reasoner: Optional[BaseReasoner] = None,
        explainer: Optional[BaseExplainer] = None,
        use_mock_fallbacks: bool = False,
    ):
        if use_mock_fallbacks:
            self.ingestor = ingestor or MockIngestor()
            self.classifier = classifier or MockClassifier()
            self.extractor = extractor or MockExtractor()
            self.reasoner = reasoner or MockReasoner()
            self.explainer = explainer or MockExplainer()
        else:
            if ingestor is None:
                from termsheet_ai.ingest.ingestor import Ingestor

                ingestor = Ingestor()
            if classifier is None:
                from termsheet_ai.classify.sbert import SbertClassifier

                classifier = SbertClassifier(mode="hybrid")
            if extractor is None:
                from termsheet_ai.extraction.extractor import FactExtractor

                extractor = FactExtractor()
            if reasoner is None:
                from termsheet_ai.reasoning.engine import ForwardChainingEngine

                reasoner = ForwardChainingEngine()
            if explainer is None:
                from termsheet_ai.explain.explainer import PlainEnglishExplainer

                explainer = PlainEnglishExplainer()

            self.ingestor = ingestor
            self.classifier = classifier
            self.extractor = extractor
            self.reasoner = reasoner
            self.explainer = explainer

    @classmethod
    def create_mock(cls) -> "TermSheetPipeline":
        """Factory for lightweight mock pipeline."""
        return cls(use_mock_fallbacks=True)

    @classmethod
    def create_production(cls) -> "TermSheetPipeline":
        """Factory for full production pipeline."""
        return cls(use_mock_fallbacks=False)

    def run(self, file_path: Union[str, Path]) -> PipelinePayload:
        """Executes full S1-S5 pipeline on the input document."""
        errors: List[str] = []

        # S1: Ingest
        metadata, clauses = self.ingestor.ingest(file_path)

        # S2: Classify
        try:
            clauses = self.classifier.classify(clauses)
        except Exception as e:
            errors.append(f"S2 Classification Error: {str(e)}")

        # S3: Extract Facts
        facts = None
        try:
            facts = self.extractor.extract(metadata.doc_id, clauses)
        except Exception as e:
            errors.append(f"S3 Fact Extraction Error: {str(e)}")

        # S4: Symbolic Reasoning
        audit_trace = None
        if facts:
            try:
                audit_trace = self.reasoner.reason(facts)
            except Exception as e:
                errors.append(f"S4 Reasoning Error: {str(e)}")

        # S5: Explain & Presentation
        summaries = []
        if audit_trace:
            try:
                if hasattr(self.explainer, "facts"):
                    self.explainer.facts = facts
                if (
                    hasattr(self.explainer, "retriever")
                    and getattr(self.explainer, "retriever", None) is None
                ):
                    self.explainer.retriever = self.classifier
                summaries = self.explainer.explain(clauses, audit_trace)
            except Exception as e:
                errors.append(f"S5 Summary Error: {str(e)}")

        return PipelinePayload(
            metadata=metadata,
            clauses=clauses,
            facts=facts,
            audit_trace=audit_trace,
            summaries=summaries,
            execution_errors=errors,
        )
