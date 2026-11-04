"""End-to-end pipeline components for TermSheet AI."""

from typing import TYPE_CHECKING

from termsheet_ai.pipeline.base import (
    BaseClassifier,
    BaseExplainer,
    BaseExtractor,
    BaseIngestor,
    BaseReasoner,
)

if TYPE_CHECKING:
    from termsheet_ai.pipeline.runner import TermSheetPipeline

__all__ = [
    "BaseClassifier",
    "BaseExplainer",
    "BaseExtractor",
    "BaseIngestor",
    "BaseReasoner",
    "TermSheetPipeline",
]


def __getattr__(name: str):
    if name == "TermSheetPipeline":
        from termsheet_ai.pipeline.runner import TermSheetPipeline

        return TermSheetPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
