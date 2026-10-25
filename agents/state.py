"""Shared workflow state owned by the Orchestrator and read / extended by every agent."""
from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agents.contracts import FieldVerdict, RetrievalOutput
from extraction.schemas import AddendumChange, BidExtraction, FieldResult
from schemas import Answer

StepStatus = Literal["pending", "running", "done", "failed", "skipped"]


class PlanStep(BaseModel):
    id: str  # e.g. "retrieve:identity"
    agent: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = "pending"
    error: Optional[str] = None
    latency_s: Optional[float] = None


class WorkflowState(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    mode: Literal["extract", "qa"]
    bid_id: Optional[str] = None
    folder: Optional[str] = None
    question: Optional[str] = None
    plan: list[PlanStep] = Field(default_factory=list)
    # evidence per retrieval label (latest call), and what the model saw per field
    evidence: dict[str, RetrievalOutput] = Field(default_factory=dict)
    shown: dict[str, set[str]] = Field(default_factory=dict)
    drafts: dict[str, FieldResult] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)
    verdicts: dict[str, FieldVerdict] = Field(default_factory=dict)
    retry_rounds: int = 0
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    final: Optional[BidExtraction] = None
    report_markdown: Optional[str] = None
    answer: Optional[Answer] = None
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    model: Optional[str] = None

    def add_step(self, id: str, agent: str, description: str = "", depends_on: Optional[list[str]] = None) -> PlanStep:
        step = PlanStep(id=id, agent=agent, description=description, depends_on=depends_on or [])
        self.plan.append(step)
        return step

    def step(self, id: str) -> PlanStep:
        return next(s for s in self.plan if s.id == id)

    def plan_summary(self) -> list[dict]:
        return [{"id": s.id, "agent": s.agent, "depends_on": s.depends_on, "status": s.status} for s in self.plan]
