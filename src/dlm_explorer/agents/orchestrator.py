"""Orchestrator: the closed auto-research loop.

    for each round:
        configs = proposer.propose(history, batch_size)   # what to try next
        for cfg in configs:
            trial = runner.run(cfg)                        # decode + score
            store.append(trial)                            # persist
        summary = analyst.summarize(history)               # what we've learned

The proposer may be a deterministic search proposer or the LLM "researcher"
(both share the `propose(history, n)` interface), so the loop is identical
offline and with an LLM in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from ..experiment.store import TrialStore
from ..experiment.trial import Trial
from .analyst import Analyst, Summary
from .proposer import Proposer
from .runner import TrialRunner


@dataclass
class LoopConfig:
    rounds: int = 4
    batch_size: int = 6


class AutoResearchLoop:
    def __init__(
        self,
        proposer: Proposer,
        runner: TrialRunner,
        analyst: Analyst,
        store: TrialStore,
        config: LoopConfig | None = None,
        on_round: Optional[Callable[[int, Summary], None]] = None,
    ) -> None:
        self.proposer = proposer
        self.runner = runner
        self.analyst = analyst
        self.store = store
        self.config = config or LoopConfig()
        self.on_round = on_round

    def run(self) -> Summary:
        history: List[Trial] = self.store.load()  # resume if ledger exists
        seen = {t.trial_id for t in history}
        for r in range(self.config.rounds):
            configs = self.proposer.propose(history, self.config.batch_size)
            if not configs:
                break
            for cfg in configs:
                tid = Trial.make_id(cfg.to_dict())
                if tid in seen:
                    continue
                trial = self.runner.run(cfg)
                self.store.append(trial)
                history.append(trial)
                seen.add(tid)
            summary = self.analyst.summarize(history)
            if self.on_round:
                self.on_round(r, summary)
        return self.analyst.summarize(history)
