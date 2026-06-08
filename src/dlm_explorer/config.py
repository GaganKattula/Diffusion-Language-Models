"""Experiment configuration: a YAML file describing a full search run.

A config wires together backend + task + search space + proposer + loop, so a run
is `dlm-explore run --config configs/xxx.yaml`. Everything is declarative and
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from .agents.proposer import PROPOSERS, SearchSpace
from .agents.analyst import Analyst
from .agents.orchestrator import AutoResearchLoop, LoopConfig
from .agents.runner import TrialRunner
from .backends import build_backend
from .eval import build_task
from .experiment.store import TrialStore


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    backend: Dict[str, Any] = field(default_factory=lambda: {"name": "mock"})
    task: Dict[str, Any] = field(default_factory=lambda: {"name": "copy"})
    search_space: Dict[str, Any] = field(default_factory=dict)
    proposer: Dict[str, Any] = field(default_factory=lambda: {"name": "grid"})
    loop: Dict[str, Any] = field(default_factory=dict)
    objective: str = "accuracy"
    cost: str = "model_calls"
    output_dir: str = "runs"

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    # --- builders ---
    def build_backend(self):
        spec = dict(self.backend)
        return build_backend(spec.pop("name"), **spec)

    def build_task(self):
        spec = dict(self.task)
        return build_task(spec.pop("name"), **spec)

    def build_space(self) -> SearchSpace:
        return SearchSpace(**self.search_space)

    def build_proposer(self, space: SearchSpace):
        spec = dict(self.proposer)
        name = spec.pop("name")
        if name == "llm":
            from .agents.llm import LLMProposer  # imported lazily (needs anthropic)
            return LLMProposer(analyst=Analyst(self.objective, self.cost), **spec)
        return PROPOSERS[name](space, **spec)

    def build_loop(self, on_round=None) -> AutoResearchLoop:
        backend = self.build_backend()
        task = self.build_task()
        space = self.build_space()
        proposer = self.build_proposer(space)
        runner = TrialRunner(backend, task)
        analyst = Analyst(self.objective, self.cost)
        store = TrialStore(Path(self.output_dir) / f"{self.name}.jsonl")
        return AutoResearchLoop(
            proposer, runner, analyst, store, LoopConfig(**self.loop), on_round=on_round
        )
