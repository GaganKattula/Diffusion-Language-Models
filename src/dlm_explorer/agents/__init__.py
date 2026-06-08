"""The multi-agent auto-research loop.

Roles (each is a small, single-responsibility component):
  * Proposer  — proposes the next decoding configurations to try.
  * Runner    — executes one configuration: decode every eval example, score,
                aggregate into a Trial.
  * Analyst   — reads the history, computes the Pareto frontier, summarizes,
                and exposes "what's interesting so far".
  * Orchestrator — runs Proposer -> Runner -> Analyst over rounds, persisting
                every Trial to the ledger.

Proposers come in two flavours:
  * search-based (grid / random / local) — deterministic, no network, the
    default so the whole loop runs offline.
  * LLM-driven (agents/llm.py) — an optional "researcher" that reads the
    Analyst's summary and proposes the next batch. Gated behind a config flag +
    ANTHROPIC_API_KEY.
"""

from .runner import TrialRunner
from .analyst import Analyst, pareto_front
from .orchestrator import AutoResearchLoop

__all__ = ["TrialRunner", "Analyst", "pareto_front", "AutoResearchLoop"]
