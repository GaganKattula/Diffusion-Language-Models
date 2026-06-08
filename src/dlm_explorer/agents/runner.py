"""Runner: execute one decoding configuration and produce a Trial.

Runs the sampler over every example in the task, scores each, and aggregates.
The key metrics:
  * accuracy            — task quality (higher better)
  * model_calls         — NFE / compute per example (lower better) -> the cost
                          axis of the step<->quality Pareto frontier
  * self_corrections    — mean self-correction events per example (the
                          "diffusion can revise" signal)
"""

from __future__ import annotations

from statistics import mean
from typing import List

from ..decoding import DecodingParams, MaskedDiffusionSampler
from ..eval.tasks import Task
from ..experiment.trial import Trial


class TrialRunner:
    def __init__(self, backend, task: Task) -> None:
        self.backend = backend
        self.task = task
        self.sampler = MaskedDiffusionSampler(backend)
        self._examples = task.examples()

    def run(self, params: DecodingParams) -> Trial:
        accs, seq_accs, calls, corrections = [], [], [], []
        for ex in self._examples:
            res = self.sampler.generate(ex.prompt_ids, ex.gen_len, params)
            s = self.task.score(res.gen_tokens, ex)
            accs.append(s["accuracy"])
            seq_accs.append(s.get("sequence_accuracy", 0.0))
            calls.append(res.num_model_calls)
            corrections.append(res.num_self_corrections)

        metrics = {
            "accuracy": mean(accs),
            "sequence_accuracy": mean(seq_accs),
            "model_calls": mean(calls),
            "self_corrections": mean(corrections),
            "n_examples": len(self._examples),
        }
        pd = params.to_dict()
        return Trial(trial_id=Trial.make_id(pd), params=pd, metrics=metrics,
                     meta={"task": self.task.name, "backend": self.backend.describe()})
