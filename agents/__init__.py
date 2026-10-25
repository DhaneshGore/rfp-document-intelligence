"""Multi-agent workflow (assignment Section 7) built on the existing services.

Agents and what they wrap (LLM = makes model calls):
  IngestionAgent       IndexingService.index_folder                 no LLM
  RetrievalAgent       ExtractionService.gather_evidence /
                       QAService.gather_evidence (bid-filtered)     no LLM
  ExtractionAgent      ExtractionService.extract_fields              LLM, 1 call per field group
  ValidatorAgent       evidence/citation/format critic + retry plan  no LLM (deterministic)
  ReconciliationAgent  scoped requests + apply_reconciliation        no LLM itself; its two scoped
                                                                     extractions run on ExtractionAgent
  QAReportAgent        QAService.answer + report rendering           LLM for answers only
The Orchestrator owns the WorkflowState, the plan, dependencies, parallelism, bounded
retries, failures and the trace of every handoff.
"""
from agents.orchestrator import BidNotIndexedError, Orchestrator  # noqa: F401
from agents.state import WorkflowState  # noqa: F401
