"""Proposers: decide which decoding configurations to try next.

A SearchSpace defines the grid of allowed values per DecodingParams field.
Proposers turn (search space, history) into the next batch of DecodingParams.

These are the *non-LLM* proposers — deterministic and offline. The LLM-driven
proposer lives in agents/llm.py and implements the same `propose` interface.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..decoding import DecodingParams
from ..experiment.trial import Trial


@dataclass
class SearchSpace:
    num_steps: Sequence[int] = (1, 2, 4, 8, 16, 32)
    unmask_order: Sequence[str] = ("confidence", "margin", "entropy", "random", "left_to_right")
    temperature: Sequence[float] = (0.0,)
    remask: Sequence[bool] = (False, True)
    remask_frac: Sequence[float] = (0.25,)
    remask_threshold: Sequence[float] = (0.5,)
    block_size: Sequence = (None,)
    seed: Sequence[int] = (0,)

    def fields(self) -> Dict[str, Sequence]:
        return {
            "num_steps": self.num_steps,
            "unmask_order": self.unmask_order,
            "temperature": self.temperature,
            "remask": self.remask,
            "remask_frac": self.remask_frac,
            "remask_threshold": self.remask_threshold,
            "block_size": self.block_size,
            "seed": self.seed,
        }

    def sample(self, rng: random.Random) -> DecodingParams:
        return DecodingParams(**{k: rng.choice(list(v)) for k, v in self.fields().items()})

    def grid(self) -> List[DecodingParams]:
        keys = list(self.fields())
        out = []
        for combo in itertools.product(*[list(v) for v in self.fields().values()]):
            out.append(DecodingParams(**dict(zip(keys, combo))))
        return out


class Proposer:
    def propose(self, history: List[Trial], n: int) -> List[DecodingParams]:
        raise NotImplementedError


class GridProposer(Proposer):
    """Enumerate the full grid once, in chunks of n per round."""

    def __init__(self, space: SearchSpace) -> None:
        self._queue = space.grid()
        self._i = 0

    def propose(self, history: List[Trial], n: int) -> List[DecodingParams]:
        batch = self._queue[self._i : self._i + n]
        self._i += len(batch)
        return batch


class RandomProposer(Proposer):
    def __init__(self, space: SearchSpace, seed: int = 0) -> None:
        self.space = space
        self.rng = random.Random(seed)

    def propose(self, history: List[Trial], n: int) -> List[DecodingParams]:
        seen = {t.trial_id for t in history}
        out, tries = [], 0
        while len(out) < n and tries < n * 50:
            tries += 1
            p = self.space.sample(self.rng)
            if Trial.make_id(p.to_dict()) not in seen:
                out.append(p)
                seen.add(Trial.make_id(p.to_dict()))
        return out


class LocalSearchProposer(Proposer):
    """Hill-climb: perturb one field of the best trial so far.

    A simple stand-in for "the agent focuses where the signal is". The LLM
    proposer is the richer version of this idea.
    """

    def __init__(self, space: SearchSpace, objective: str = "accuracy", seed: int = 0) -> None:
        self.space = space
        self.objective = objective
        self.rng = random.Random(seed)

    def propose(self, history: List[Trial], n: int) -> List[DecodingParams]:
        if not history:
            return [self.space.sample(self.rng) for _ in range(n)]
        best = max(history, key=lambda t: t.metrics.get(self.objective, 0.0))
        out, seen = [], {t.trial_id for t in history}
        fields = self.space.fields()
        tries = 0
        while len(out) < n and tries < n * 50:
            tries += 1
            d = dict(best.params)
            k = self.rng.choice(list(fields))
            d[k] = self.rng.choice(list(fields[k]))
            p = DecodingParams(**d)
            tid = Trial.make_id(p.to_dict())
            if tid not in seen:
                out.append(p)
                seen.add(tid)
        return out


PROPOSERS = {
    "grid": GridProposer,
    "random": RandomProposer,
    "local": LocalSearchProposer,
}
