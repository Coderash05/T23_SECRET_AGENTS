"""Single place where the UI gets a PipelinePayload (mock file or the real pipeline)."""

import tempfile
from pathlib import Path
from typing import Optional

from termsheet_ai.pipeline.runner import MockIngestor, TermSheetPipeline
from termsheet_ai.schemas.contracts import PipelinePayload

MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_payload.json"


def load_mock_payload(path: Path = MOCK_PATH) -> PipelinePayload:
    """Load and validate the bundled demo result."""
    return PipelinePayload.model_validate_json(path.read_text(encoding="utf-8"))


def build_pipeline() -> TermSheetPipeline:
    """Swap real S1-S5 components in here as teammates finish them."""
    return TermSheetPipeline()


def uses_plaintext_ingestor(pipeline: TermSheetPipeline) -> bool:
    return isinstance(pipeline.ingestor, MockIngestor)


def run_pipeline_on_upload(
    filename: str, data: bytes, pipeline: Optional[TermSheetPipeline] = None
) -> PipelinePayload:
    """Write the upload to a temp file (keeping its name) and run the pipeline on it."""
    safe_name = Path(filename).name or "upload.txt"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / safe_name
        path.write_bytes(data)
        return (pipeline or build_pipeline()).run(path)
