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

    def __init__(self, backend=None, n_examples: int = 16, seq_len: int = 12, vocab_size: int = 66, seed: int = 0):
        # `backend` is accepted for a uniform task-construction signature; the
        # copy task generates synthetic ids and does not need a tokenizer.
        self.backend = backend
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


import re  # noqa: E402


@TASKS.register("gsm8k")
class Gsm8kTask(Task):
    """GSM8K grade-school math word problems, generative eval.

    Diffusion LMs generate a FIXED-LENGTH region (no EOS-based stopping), so we
    give each question a fixed `gen_len` token budget, denoise it, decode, and
    extract the final integer. Scored by exact match against the gold answer.

    Few-shot: the first `n_shot` TRAIN examples are prepended (standard GSM8K
    protocol) so the base model learns the "reasoning ... #### answer" format.
    Needs a backend (for its tokenizer) and `datasets`.
    """

    name = "gsm8k"

    def __init__(self, backend=None, n_examples: int = 100, n_shot: int = 4,
                 gen_len: int = 200, split: str = "test", seed: int = 0):
        self.backend = backend
        self.n_examples = n_examples
        self.n_shot = n_shot
        self.gen_len = gen_len
        self.split = split
        self.seed = seed

    def _fewshot_prefix(self, train) -> str:
        shots = [f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}"
                 for i in range(self.n_shot)]
        return ("\n\n".join(shots) + "\n\n") if shots else ""

    def examples(self) -> List[Example]:
        if self.backend is None:
            raise ValueError("Gsm8kTask needs a backend (tokenizer). Pass backend=...")
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError("pip install datasets to use the gsm8k task.") from e

        ds = load_dataset("gsm8k", "main")
        train, test = ds["train"], ds[self.split]
        prefix = self._fewshot_prefix(train)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(test))[: self.n_examples]
        out = []
        for i in idx:
            q = test[int(i)]["question"]
            gold = self._extract_gold(test[int(i)]["answer"])
            prompt = f"{prefix}Question: {q}\nAnswer:"
            out.append(Example(
                prompt_ids=self.backend.encode(prompt),
                target_ids=[],
                gen_len=self.gen_len,
                prompt_text=prompt,
                target_text=gold,
            ))
        return out

    @staticmethod
    def _extract_gold(answer: str) -> str:
        # GSM8K gold answers end with "#### <number>".
        m = re.search(r"####\s*(-?[\d,]+)", answer)
        return m.group(1).replace(",", "") if m else ""

    @staticmethod
    def _extract_pred(text: str) -> str:
        # The model may run on into a hallucinated next "Question:"; cut at the
        # first one, then take the last number in this answer (handles "#### N"
        # and bare final answers).
        text = text.split("Question:")[0]
        nums = re.findall(r"-?\d[\d,]*", text)
        return nums[-1].replace(",", "") if nums else ""

    def score(self, gen_tokens: List[int], example: Example) -> dict:
        if self.backend is None:
            raise ValueError("Gsm8kTask.score needs a backend (detokenizer).")
        pred = self._extract_pred(self.backend.decode(gen_tokens))
        hit = 1.0 if pred != "" and pred == example.target_text else 0.0
        return {"accuracy": hit, "sequence_accuracy": hit}
