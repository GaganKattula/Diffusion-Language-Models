"""Evaluation tasks.

`CopyTask` is the offline task that pairs with the mock backend: the generation
region must reproduce the prompt token-for-token, so we have a deterministic
ground-truth to score against with no model weights. It is what makes the
mock-backed pipeline produce *meaningful* accuracy/Pareto numbers.

`Gsm8kTask` is a Phase-2 placeholder for a real benchmark on the LLaDA backend.
It intentionally raises with guidance rather than silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import TASKS


@dataclass
class Example:
    prompt_ids: List[int]
    target_ids: List[int]   # ground-truth for the generation region
    gen_len: int
    prompt_text: str = ""
    target_text: str = ""


class Task:
    name: str = "task"

    def examples(self) -> List[Example]:
        raise NotImplementedError

    def score(self, gen_tokens: List[int], example: Example) -> dict:
        """Return per-example metrics. Must include 'accuracy' in [0,1]."""
        raise NotImplementedError


@TASKS.register("copy")
class CopyTask(Task):
    name = "copy"

    def __init__(self, n_examples: int = 16, seq_len: int = 12, vocab_size: int = 66, seed: int = 0):
        self.n_examples = n_examples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self._first_content = 2  # keep in sync with backends.mock.FIRST_CONTENT_ID

    def examples(self) -> List[Example]:
        rng = np.random.default_rng(self.seed)
        out = []
        for _ in range(self.n_examples):
            ids = rng.integers(self._first_content, self.vocab_size, size=self.seq_len).tolist()
            out.append(Example(prompt_ids=ids, target_ids=ids, gen_len=self.seq_len))
        return out

    def score(self, gen_tokens: List[int], example: Example) -> dict:
        tgt = example.target_ids
        n = min(len(gen_tokens), len(tgt))
        correct = sum(1 for a, b in zip(gen_tokens[:n], tgt[:n]) if a == b)
        token_acc = correct / len(tgt) if tgt else 0.0
        seq_acc = 1.0 if gen_tokens[: len(tgt)] == tgt else 0.0
        return {"accuracy": token_acc, "sequence_accuracy": seq_acc}


@TASKS.register("gsm8k")
class Gsm8kTask(Task):
    name = "gsm8k"

    def __init__(self, n_examples: int = 100, split: str = "test", seed: int = 0):
        self.n_examples = n_examples
        self.split = split
        self.seed = seed

    def examples(self) -> List[Example]:  # pragma: no cover - Phase 2
        raise NotImplementedError(
            "Gsm8kTask is a Phase-2 placeholder for the real LLaDA backend.\n"
            "To implement: load GSM8K via `datasets`, tokenize each question with\n"
            "the LLaDA tokenizer, set gen_len to a fixed budget, and score by\n"
            "extracting the final numeric answer. See docs/05_roadmap.md."
        )

    def score(self, gen_tokens, example) -> dict:  # pragma: no cover - Phase 2
        raise NotImplementedError
