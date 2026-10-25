"""Ingestion agent: parse + chunk + index a bid folder (wraps IndexingService; no LLM)."""
from __future__ import annotations

from pathlib import Path

from agents.base import Agent
from agents.contracts import IngestionInput, IngestionOutput
from agents.state import WorkflowState
from extraction.trace import ExtractionTrace
from services.uploads import validate_bid_id


class IngestionAgent(Agent):
    name = "ingestion"
    uses_llm = False
    input_type, output_type = IngestionInput, IngestionOutput

    def __init__(self, indexing):
        self.indexing = indexing

    def run(self, inp: IngestionInput, state: WorkflowState, trace: ExtractionTrace) -> IngestionOutput:
        folder = Path(inp.folder)
        bid_id = validate_bid_id(inp.bid_id or folder.resolve().name)
        report = self.indexing.index_folder(folder, bid_id=bid_id)  # incremental: unchanged files skipped
        return IngestionOutput(
            bid_id=bid_id, files=report.summary, chunks_added=report.total_chunks_added,
            failed_files=[f.file_name for f in report.files if f.status == "failed"],
        )
