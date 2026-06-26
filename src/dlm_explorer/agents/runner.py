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

from ..decoding import DecodingParams
from ..decoding.batched import BatchedSampler
from ..eval.tasks import Task
from ..experiment.trial import Trial


class TrialRunner:
    def __init__(self, backend, task: Task, batch_size: int = 1) -> None:
        self.backend = backend
        self.task = task
        self.batch_size = max(1, batch_size)
        # Always use the batched path (even at batch_size=1) so the per-step
        # reduction runs through backend.predict_batch — on-device for LLaDA.
        # The serial MaskedDiffusionSampler does a full-vocab softmax+sort on the
        # CPU every step, which is fine for the tiny-vocab mock but pathological
        # for a 126k-vocab real model.
        self.batched = BatchedSampler(backend)
        self._examples = task.examples()

    def _decode_all(self, params: DecodingParams):
        """Return a list of (GenerationResult, Example).

        Batches are bucketed by (gen_len, prompt_len) so every sequence in a batch
        has the SAME length and NO padding is needed. A batched forward over
        equal-length, all-real-token sequences is identical to N single forwards
        (attention is within-sequence), which keeps batched == serial exactly on
        the real model. (Right-padded batching across unequal lengths corrupted
        LLaDA's bidirectional attention — see docs/05_roadmap.md.)
        """
        out = []

        def key(i):
            e = self._examples[i]
            return (e.gen_len, len(e.prompt_ids))

        order = sorted(range(len(self._examples)), key=key)
        for (gen_len, _plen), group in groupby(order, key=key):
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
