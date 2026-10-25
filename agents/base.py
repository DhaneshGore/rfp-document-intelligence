"""Agent interface: a named unit with one responsibility and typed input/output."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.state import WorkflowState
from extraction.trace import ExtractionTrace


class Agent(ABC):
    name: str = "agent"
    uses_llm: bool = False  # honest labelling: True only if run() can call the model
    input_type: type = object
    output_type: type = object

    @abstractmethod
    def run(self, inp: Any, state: WorkflowState, trace: ExtractionTrace) -> Any:
        """Consume a typed input, read/extend the shared state, return a typed output."""
