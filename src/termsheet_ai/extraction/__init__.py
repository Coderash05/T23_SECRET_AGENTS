"""Stage S3 Fact Extraction module for TermSheet AI."""

from termsheet_ai.extraction.extractor import FactExtractor
from termsheet_ai.extraction.span_validator import SpanValidator

__all__ = ["FactExtractor", "SpanValidator"]
