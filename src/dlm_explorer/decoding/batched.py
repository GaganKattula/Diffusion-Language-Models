"""Batched masked-diffusion decoding.

Decodes many examples in lockstep: one batched forward per denoising step
(via backend.predict_batch) instead of one forward per example per step. Since
diffusion has no KV-cache (every step is a full forward), batching examples is
the main wall-clock lever.

The per-row commit / remask / schedule logic mirrors the serial
MaskedDiffusionSampler exactly, so batched and serial decoding produce identical
results for deterministic configs — verified by tests/test_batched_equiv.py.

Right-padding is used so real-token positions are unchanged across the batch
(important for position-sensitive models); pad columns are attention-masked out
by the backend. Examples in one batch must share the same gen_len.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .sampler import DecodingParams, GenerationResult


def _priority(pred, b, masked, order, seed, step):
    """Higher => commit sooner. Mirrors sampler._priority using precomputed
    per-position summaries from StepPrediction."""
    if order == "random":
        rng = np.random.default_rng((seed, step, b))
        return rng.random(masked.size)
    if order == "left_to_right":
        return -masked.astype(np.float64)  # smaller column => higher priority
    if order == "confidence":
        return pred.confidence[b, masked]
    if order == "margin":
        return pred.margin[b, masked]
    if order == "entropy":
        return pred.neg_entropy[b, masked]
    raise ValueError(order)  # pragma: no cover


class BatchedSampler:
    def __init__(self, backend) -> None:
        self.backend = backend

    def generate_batch(
        self, prompts: List[List[int]], gen_len: int, params: DecodingParams
    ) -> List[GenerationResult]:
        mask = self.backend.mask_token_id
        pad = getattr(self.backend, "pad_token_id", 0)
        B = len(prompts)
        pls = [len(p) for p in prompts]
        valid_lens = [pl + gen_len for pl in pls]
        L = max(valid_lens)

        seq = np.full((B, L), pad, dtype=np.int64)
        for b, p in enumerate(prompts):
            seq[b, : pls[b]] = p
            seq[b, pls[b] : pls[b] + gen_len] = mask

        block = params.block_size or gen_len
        calls = 0
        steps_used = 0
        corrections = [0] * B
        first_commit = [dict() for _ in range(B)]

        def commit(b, c, value):
            c = int(c)
            if c in first_commit[b] and first_commit[b][c] != value:
                corrections[b] += 1
            first_commit[b].setdefault(c, value)
            seq[b, c] = value

        for blk in range(0, gen_len, block):
            bs, be = blk, min(blk + block, gen_len)
            for step in range(params.num_steps):
                # any masked positions left in this block across the batch?
                any_masked = any(
                    (seq[b, pls[b] + bs : pls[b] + be] == mask).any() for b in range(B)
                )
                if not any_masked and not params.remask:
                    break

                pred = self.backend.predict_batch(
                    seq, pls, valid_lens, params.temperature, params.seed, calls
                )
                calls += 1

                for b in range(B):
                    cols = np.arange(pls[b] + bs, pls[b] + be)
                    row = seq[b]
                    masked = cols[row[cols] == mask]
                    committed = cols[row[cols] != mask]

                    if masked.size > 0:
                        steps_left = params.num_steps - step
                        k = int(np.ceil(masked.size / max(steps_left, 1)))
                        prio = _priority(pred, b, masked, params.unmask_order, params.seed, step)
                        chosen = masked[np.argsort(-prio)[:k]]
                        for c in chosen:
                            commit(b, c, int(pred.tokens[b, c]))

                    if params.remask and committed.size > 0 and step < params.num_steps - 1:
                        cp = pred.cur_prob[b, committed]
                        low_mask = cp < params.remask_threshold
                        low = committed[low_mask]
                        if low.size > 0:
                            n_re = max(1, int(np.floor(params.remask_frac * low.size)))
                            worst = low[np.argsort(cp[low_mask])[:n_re]]
                            seq[b, worst] = mask
                steps_used += 1

            # cleanup: fill any masks still left in this block
            any_left = any(
                (seq[b, pls[b] + bs : pls[b] + be] == mask).any() for b in range(B)
            )
            if any_left:
                pred = self.backend.predict_batch(
                    seq, pls, valid_lens, params.temperature, params.seed, calls
                )
                calls += 1
                for b in range(B):
                    cols = np.arange(pls[b] + bs, pls[b] + be)
                    masked = cols[seq[b, cols] == mask]
                    for c in masked:
                        commit(b, c, int(pred.tokens[b, c]))

        results = []
        for b in range(B):
            gen = seq[b, pls[b] : pls[b] + gen_len].tolist()
            results.append(
                GenerationResult(
                    tokens=seq[b, : valid_lens[b]].tolist(),
                    gen_tokens=gen,
                    num_model_calls=calls,
                    num_self_corrections=corrections[b],
                    steps_used=steps_used,
                    trajectory=[],
                )
            )
        return results
