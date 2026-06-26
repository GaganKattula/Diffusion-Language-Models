"""The backend contract.

A masked diffusion LM, at its core, takes a partially-masked token sequence and
predicts a distribution over the vocabulary at every position. That is the whole
interface the decoding sampler needs:

    logits(input_ids, prompt_len) -> (L, V) array of per-position logits

Everything about *how* you turn those logits into text over multiple denoising
steps (step count, unmask order, remasking, blocks) lives in the sampler, not
here. Keeping that split is what lets one search the decoding space without
touching the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class StepPrediction:
    """Compact per-position summary of one batched forward pass.

    All arrays are (B, L). This is what the batched sampler needs from a single
    denoising step — computed by the backend (on-device for LLaDA) so the full
    (B, L, V) logits never have to leave the GPU. Values at pad columns are
    unused by the sampler.

    tokens:       argmax (or sampled) token id at each position
    confidence:   top-1 probability
    margin:       top-1 minus top-2 probability
    neg_entropy:  sum_v p*log p  ( = -entropy; higher => more confident)
    cur_prob:     probability assigned to the token currently at that position
                  (used by the remask / self-correction rule)
    """

    tokens: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    neg_entropy: np.ndarray
    cur_prob: np.ndarray


class MaskedDiffusionBackend(ABC):
    #: Token id that represents a masked position.
    mask_token_id: int
    #: Vocabulary size V (logits are returned over this dimension).
    vocab_size: int

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenize prompt text into token ids."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Detokenize token ids back into text."""

    @abstractmethod
    def logits(self, input_ids: np.ndarray, prompt_len: int) -> np.ndarray:
        """Predict per-position logits for a single sequence.

        Args:
            input_ids: 1-D int array of length L. Positions equal to
                ``mask_token_id`` are masked; others are committed (prompt or
                previously-unmasked generation tokens).
            prompt_len: number of leading positions that are the fixed prompt.
                Real models infer conditioning from the tokens themselves and can
                ignore this; the mock backend uses it to locate the generation
                region. It is passed for both so the contract is uniform.

        Returns:
            (L, V) float32 array of logits.
        """

    def predict_batch(
        self,
        input_ids: np.ndarray,
        prompt_lens: list[int],
        valid_lens: list[int],
        temperature: float = 0.0,
        seed: int = 0,
        step: int = 0,
    ) -> "StepPrediction":
        """Per-position prediction summary for a batch (default impl).

        Generic, correctness-first implementation built on the single-sequence
        `logits()`: it loops rows and computes the summaries in numpy. Backends
        with a real device (LLaDA) OVERRIDE this to run one batched forward and
        do the reduction on-device, returning only the small (B, L) arrays.

        input_ids:  (B, L) padded batch. Each row b is valid on [0, valid_lens[b])
                    (prompt + generation region) and padded after.
        """
        B, L = input_ids.shape
        tokens = np.zeros((B, L), dtype=np.int64)
        conf = np.zeros((B, L), dtype=np.float32)
        margin = np.zeros((B, L), dtype=np.float32)
        neg_ent = np.zeros((B, L), dtype=np.float32)
        cur = np.zeros((B, L), dtype=np.float32)
        for b in range(B):
            vlen = valid_lens[b]
            lg = self.logits(input_ids[b, :vlen], prompt_lens[b])  # (vlen, V)
            p = _softmax(lg)
            if temperature > 0:
                rng = np.random.default_rng((seed, step, b))
                pt = _softmax(lg / temperature)
                tok = np.array([rng.choice(p.shape[1], p=pt[i]) for i in range(vlen)])
            else:
                tok = lg.argmax(-1)
            top = np.sort(p, axis=-1)
            tokens[b, :vlen] = tok
            conf[b, :vlen] = top[:, -1]
            margin[b, :vlen] = top[:, -1] - top[:, -2]
            neg_ent[b, :vlen] = np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=-1)
            cur[b, :vlen] = p[np.arange(vlen), input_ids[b, :vlen]]
        return StepPrediction(tokens, conf, margin, neg_ent, cur)

    # Optional: backends may override to report a human-readable identity.
    def describe(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"
