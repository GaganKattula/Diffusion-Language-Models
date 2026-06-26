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

from itertools import groupby
from statistics import mean
from typing import List

from ..decoding import DecodingParams, MaskedDiffusionSampler
from ..decoding.batched import BatchedSampler
from ..eval.tasks import Task
from ..experiment.trial import Trial


class TrialRunner:
    def __init__(self, backend, task: Task, batch_size: int = 1) -> None:
        self.backend = backend
        self.task = task
        self.batch_size = max(1, batch_size)
        self.sampler = MaskedDiffusionSampler(backend)
        self.batched = BatchedSampler(backend) if self.batch_size > 1 else None
        self._examples = task.examples()

    def _decode_all(self, params: DecodingParams):
        """Return a list of (GenerationResult, Example), batched if enabled.

        Batched decoding groups examples by gen_len (a batch must share gen_len),
        then chunks each group into batches of `batch_size`.
        """
        if self.batched is None:
            return [(self.sampler.generate(ex.prompt_ids, ex.gen_len, params), ex)
                    for ex in self._examples]

        out = []
        by_len = sorted(range(len(self._examples)), key=lambda i: self._examples[i].gen_len)
        for gen_len, group in groupby(by_len, key=lambda i: self._examples[i].gen_len):
            idxs = list(group)
            for s in range(0, len(idxs), self.batch_size):
                chunk = idxs[s : s + self.batch_size]
                prompts = [self._examples[i].prompt_ids for i in chunk]
                results = self.batched.generate_batch(prompts, gen_len, params)
                for i, res in zip(chunk, results):
                    out.append((res, self._examples[i]))
        return out

    def run(self, params: DecodingParams) -> Trial:
        accs, seq_accs, calls, corrections = [], [], [], []
        for res, ex in self._decode_all(params):
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
