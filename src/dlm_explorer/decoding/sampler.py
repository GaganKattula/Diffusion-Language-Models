"""The masked-diffusion decoding loop.

This single, fully-configurable sampler covers the decoding design space we want
the agent to explore. A `DecodingParams` value fully determines the sampler's
behaviour, so "exploring the parameter space" == "proposing DecodingParams and
measuring the resulting Trial".

The loop (per generation block):

  1. Start with the generation region fully masked.
  2. For each of `num_steps` denoising steps:
       a. Ask the backend for per-position logits (this is the unit of compute —
          we count it as a "model call" / NFE, the real cost axis of the Pareto
          frontier).
       b. Score every masked position by an unmasking-order policy.
       c. Commit the top-k highest-priority masked positions (k set by a linear
          schedule so all positions are committed by the final step).
       d. Optionally REMASK already-committed positions whose confidence is now
          low (self-correction) — the structural move autoregression cannot make.
  3. Stop when nothing is masked.

`block_size` turns this into semi-autoregressive decoding: the generation region
is split into left-to-right blocks, each denoised in turn with earlier blocks
held fixed as context (this is the lever that recovers a KV-cache-like structure,
à la block diffusion / LLaDA's semi-AR mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

UNMASK_ORDERS = ("confidence", "margin", "entropy", "random", "left_to_right")


@dataclass(frozen=True)
class DecodingParams:
    """A point in the decoding search space.

    num_steps:      denoising steps per block. Fewer = faster, lower quality.
    unmask_order:   how to pick which masked positions to commit each step.
    temperature:    0.0 => argmax; >0 => sample from softmax(logits/T).
    remask:         enable self-correction (re-mask low-confidence commits).
    remask_frac:    fraction of committed positions eligible to be re-masked
                    each step (only those below `remask_threshold` are).
    remask_threshold: confidence below which a committed token may be re-masked.
    block_size:     None => one block (pure diffusion). N => semi-AR blocks of N.
    seed:           RNG seed for sampling / random order.
    """

    num_steps: int = 16
    unmask_order: str = "confidence"
    temperature: float = 0.0
    remask: bool = False
    remask_frac: float = 0.25
    remask_threshold: float = 0.5
    block_size: Optional[int] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.unmask_order not in UNMASK_ORDERS:
            raise ValueError(
                f"unmask_order={self.unmask_order!r} not in {UNMASK_ORDERS}"
            )
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    tokens: List[int]                 # full sequence (prompt + generation)
    gen_tokens: List[int]             # generation region only
    num_model_calls: int              # NFE — the compute/latency proxy
    num_self_corrections: int         # positions whose value changed via remask
    steps_used: int                   # total denoising steps actually run
    trajectory: List[int] = field(default_factory=list)  # masks remaining per step


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _priority(probs: np.ndarray, order: str, rng: np.random.Generator) -> np.ndarray:
    """Higher value => commit sooner. probs is (n_masked, V)."""
    if order == "random":
        return rng.random(probs.shape[0])
    if order == "left_to_right":
        # Earlier positions get higher priority; positions are already in order.
        return np.arange(probs.shape[0], 0, -1, dtype=np.float64)
    top = np.sort(probs, axis=-1)
    if order == "confidence":
        return top[:, -1]
    if order == "margin":
        return top[:, -1] - top[:, -2]
    if order == "entropy":
        ent = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0), axis=-1)
        return -ent  # lower entropy => higher priority
    raise ValueError(order)  # pragma: no cover


class MaskedDiffusionSampler:
    def __init__(self, backend) -> None:
        self.backend = backend

    def generate(
        self, prompt_ids: List[int], gen_len: int, params: DecodingParams
    ) -> GenerationResult:
        mask = self.backend.mask_token_id
        prompt_len = len(prompt_ids)
        seq = np.array(list(prompt_ids) + [mask] * gen_len, dtype=np.int64)

        block = params.block_size or gen_len
        rng = np.random.default_rng(params.seed)

        total_calls = 0
        total_corrections = 0
        steps_used = 0
        trajectory: List[int] = []

        for b_start in range(0, gen_len, block):
            b_end = min(b_start + block, gen_len)
            calls, corr, steps, traj = self._denoise_region(
                seq, prompt_len, b_start, b_end, params, rng
            )
            total_calls += calls
            total_corrections += corr
            steps_used += steps
            trajectory.extend(traj)

        gen = seq[prompt_len:].tolist()
        return GenerationResult(
            tokens=seq.tolist(),
            gen_tokens=gen,
            num_model_calls=total_calls,
            num_self_corrections=total_corrections,
            steps_used=steps_used,
            trajectory=trajectory,
        )

    def _denoise_region(
        self, seq, prompt_len, b_start, b_end, params: DecodingParams, rng
    ):
        """Denoise gen-region positions [b_start, b_end) in place. Returns
        (num_model_calls, num_self_corrections, steps_used, trajectory)."""
        mask = self.backend.mask_token_id
        abs_lo, abs_hi = prompt_len + b_start, prompt_len + b_end
        region = np.arange(abs_lo, abs_hi)

        calls = corrections = steps = 0
        trajectory: List[int] = []
        first_commit: dict[int, int] = {}  # position -> value at first commit

        def commit(p: int, value: int) -> None:
            nonlocal corrections
            if p in first_commit and first_commit[p] != value:
                corrections += 1  # this position changed after being committed once
            first_commit.setdefault(p, value)
            seq[p] = value

        for step in range(params.num_steps):
            masked = region[seq[region] == mask]
            committed = region[seq[region] != mask]
            if masked.size == 0 and not params.remask:
                break

            logits = self.backend.logits(seq, prompt_len)  # (L, V)
            calls += 1
            probs = _softmax(logits)

            # --- commit step: unmask the highest-priority masked positions ---
            if masked.size > 0:
                steps_left = params.num_steps - step
                k = int(np.ceil(masked.size / max(steps_left, 1)))
                prio = _priority(probs[masked], params.unmask_order, rng)
                chosen = masked[np.argsort(-prio)[:k]]
                for p in chosen:
                    commit(int(p), self._pick_token(logits[p], params, rng))

            # --- self-correction: re-mask committed positions the model now
            # disbelieves. We score by the probability assigned to the CURRENTLY
            # committed token (not max-prob): that value falls once neighbours fill
            # in and a better token emerges, which is the revision signal. ---
            if params.remask and committed.size > 0 and step < params.num_steps - 1:
                cur_tok_prob = probs[committed, seq[committed]]
                low_mask = cur_tok_prob < params.remask_threshold
                low = committed[low_mask]
                if low.size > 0:
                    n_re = max(1, int(np.floor(params.remask_frac * low.size)))
                    worst = low[np.argsort(cur_tok_prob[low_mask])[:n_re]]
                    seq[worst] = mask

            steps += 1
            trajectory.append(int((seq[region] == mask).sum()))

        # Final cleanup: if any masks remain (e.g. remask on last step), fill them.
        masked = region[seq[region] == mask]
        if masked.size > 0:
            logits = self.backend.logits(seq, prompt_len)
            calls += 1
            for p in masked:
                commit(int(p), self._pick_token(logits[p], params, rng))

        # `corrections` now counts generation positions whose committed value
        # changed after being committed at least once — i.e. genuine
        # self-corrections, only possible when remask is enabled.
        return calls, corrections, steps, trajectory

    def _pick_token(self, logit_row: np.ndarray, params: DecodingParams, rng) -> int:
        if params.temperature <= 0:
            return int(np.argmax(logit_row))
        p = _softmax(logit_row / params.temperature)
        return int(rng.choice(len(p), p=p))
