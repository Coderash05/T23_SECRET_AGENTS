"""Extraction module for TermSheet AI (Stage S3)."""

from termsheet_ai.extract.extractor import (
    FactExtractor,
    LLMCallable,
    create_fact_extractor,
)
from termsheet_ai.extract.llm_cache import make_cached_callable
from termsheet_ai.extract.llm_client import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
    make_gemini_callable,
)
from termsheet_ai.extract.span_validator import (
    SpanValidationResult,
    validate_source_span,
)

__all__ = [
    "FactExtractor",
    "LLMCallable",
    "create_fact_extractor",
    "SpanValidationResult",
    "validate_source_span",
    "make_gemini_callable",
    "make_cached_callable",
    "GeminiAPIError",
    "GeminiQuotaError",
    "GeminiModelNotFoundError",
    "GeminiAuthenticationError",
]
