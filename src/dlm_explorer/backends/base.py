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

import numpy as np


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

    # Optional: backends may override to report a human-readable identity.
    def describe(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"
