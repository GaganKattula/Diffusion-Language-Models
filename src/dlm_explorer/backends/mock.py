"""A deterministic, weight-free *simulator* of a masked diffusion LM.

THIS IS NOT A LANGUAGE MODEL. It exists so the entire search/agent pipeline can
run and be validated with no GPU, no network, and no model weights, while still
exhibiting the qualitative behaviours that make the decoding search meaningful:

  * More denoising steps + a good unmasking order beats a one-shot decode.
  * Committing high-confidence positions first seeds context for their
    neighbours, so confidence/margin ordering > random ordering.
  * Allowing remasking ("self-correction") lets early mistakes get fixed once
    surrounding context fills in — the structural advantage diffusion has over
    left-to-right autoregression.

The task it simulates is a COPY task: the generation region should reproduce the
prompt token-for-token. The backend knows the target (it can read the prompt),
and emits logits whose confidence in the correct token grows with the number of
already-committed correct neighbours. A position-dependent pseudo-random
"distractor" makes the zero-context argmax wrong often enough that iteration and
self-correction visibly help.

Tune BASE / CONTEXT_BONUS / NOISE_MAX to make the synthetic problem easier or
harder. The real signal comes from the LLaDA backend (Phase 2); this is the
test harness for the *machinery*.
"""

from __future__ import annotations

import numpy as np

from . import BACKENDS
from .base import MaskedDiffusionBackend

# Reserved ids.
PAD_ID = 0
MASK_ID = 1
FIRST_CONTENT_ID = 2


@BACKENDS.register("mock")
class MockCopyBackend(MaskedDiffusionBackend):
    def __init__(
        self,
        vocab_size: int = 66,
        base: float = 3.0,
        context_bonus: float = 2.5,
        noise_max: float = 5.0,
        seed: int = 0,
    ) -> None:
        if vocab_size <= FIRST_CONTENT_ID + 1:
            raise ValueError("vocab_size too small for any content tokens")
        self.vocab_size = vocab_size
        self.mask_token_id = MASK_ID
        self.base = base
        self.context_bonus = context_bonus
        self.noise_max = noise_max
        self.seed = seed
        self._n_content = vocab_size - FIRST_CONTENT_ID

    # --- (de)tokenization: whitespace words mapped to content-token ids ---
    def encode(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            ids.append(FIRST_CONTENT_ID + (abs(hash((tok, self.seed))) % self._n_content))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(
            "<mask>" if i == MASK_ID else ("<pad>" if i == PAD_ID else f"t{i}")
            for i in ids
        )

    def _noise(self, pos: int) -> float:
        # Deterministic per-position distractor strength in [0, noise_max).
        r = (abs(hash((pos, self.seed))) % 10_000) / 10_000.0
        return r * self.noise_max

    def _distractor(self, target: int, pos: int) -> int:
        off = 1 + (abs(hash((pos, "d", self.seed))) % (self._n_content - 1))
        d = FIRST_CONTENT_ID + ((target - FIRST_CONTENT_ID + off) % self._n_content)
        return d

    def logits(self, input_ids: np.ndarray, prompt_len: int) -> np.ndarray:
        L = len(input_ids)
        out = np.full((L, self.vocab_size), -1e4, dtype=np.float32)
        for p in range(L):
            if p < prompt_len:
                # Prompt positions: trivially confident in their own token.
                out[p, int(input_ids[p])] = 10.0
                continue
            j = p - prompt_len  # index within the generation region
            if j >= prompt_len:
                # Generation longer than prompt (no copy target): mild prior.
                out[p, FIRST_CONTENT_ID] = self.base
                continue
            target = int(input_ids[j])  # copy task: gen[j] should equal prompt[j]
            # Context bonus from already-committed, correct generation neighbours.
            ctx = 0
            for nb in (p - 1, p + 1):
                if prompt_len <= nb < prompt_len + prompt_len and nb < L:
                    nbj = nb - prompt_len
                    if nbj < prompt_len and input_ids[nb] != MASK_ID and int(input_ids[nb]) == int(input_ids[nbj]):
                        ctx += 1
            out[p, target] = self.base + self.context_bonus * ctx
            out[p, self._distractor(target, p)] = self._noise(p)
        return out

    def describe(self) -> str:
        return (
            f"MockCopyBackend(vocab_size={self.vocab_size}, base={self.base}, "
            f"context_bonus={self.context_bonus}, noise_max={self.noise_max})"
        )
