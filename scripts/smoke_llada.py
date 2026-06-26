#!/usr/bin/env python3
"""Phase-1 real-model plumbing smoke test for the LLaDA backend.

Run on a GPU box AFTER:  pip install -e ".[llada]"
    python scripts/smoke_llada.py
    python scripts/smoke_llada.py --model GSAI-ML/LLaDA-8B-Instruct

This does NOT measure quality. It verifies the plumbing the whole project
depends on, and prints what it finds so we can correct assumptions:
  1. the checkpoint loads, and what its mask_token_id / vocab_size actually are,
  2. backend.logits(seq, prompt_len) returns a (L, V) array (i.e. the model
     really exposes per-position .logits — the core contract),
  3. the MaskedDiffusionSampler denoising loop runs end-to-end on the real model
     and decodes to text.

If step 2 fails, that's the thing to fix in src/dlm_explorer/backends/llada.py
(model class / forward), and it's exactly why we run this first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlm_explorer.backends import build_backend  # noqa: E402
from dlm_explorer.decoding import DecodingParams, MaskedDiffusionSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen-len", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    print(f"[1/4] loading {args.model} (first run downloads the weights) ...", flush=True)
    backend = build_backend("llada", model_name=args.model)
    print("      backend:", backend.describe())
    print("      mask_token_id:", backend.mask_token_id, " vocab_size:", backend.vocab_size)

    print("[2/4] encoding prompt ...", flush=True)
    pids = backend.encode(args.prompt)
    print("      prompt ids:", pids)

    print("[3/4] direct logits shape check ...", flush=True)
    seq = np.array(list(pids) + [backend.mask_token_id] * args.gen_len)
    lg = backend.logits(seq, prompt_len=len(pids))
    expected = (len(seq), backend.vocab_size)
    print(f"      logits shape: {lg.shape}  expected: {expected}")
    assert lg.shape == expected, "logits shape mismatch — fix backends/llada.py"

    print(f"[4/4] sampler end-to-end ({args.steps} steps, confidence order) ...", flush=True)
    sampler = MaskedDiffusionSampler(backend)
    res = sampler.generate(
        pids, gen_len=args.gen_len,
        params=DecodingParams(num_steps=args.steps, unmask_order="confidence"),
    )
    print("      model_calls (NFE):", res.num_model_calls)
    print("      decoded generation:", repr(backend.decode(res.gen_tokens)))
    print("\nOK — plumbing verified. (Quality is not asserted here.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
