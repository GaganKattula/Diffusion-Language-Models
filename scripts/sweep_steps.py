#!/usr/bin/env python3
"""Phase-1 de-risk: hand-sweep num_steps on real GSM8K and confirm a clean
steps <-> accuracy curve before turning the agent loop loose.

Run on the GPU pod after the smoke test passes:
    python scripts/sweep_steps.py --n 20 --steps 16 32 64 128

If accuracy rises with steps (even modestly), the real substrate works and we
move to the full decoding search. This is intentionally a plain loop (no agent
loop) so the signal is easy to read.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlm_explorer.agents.runner import TrialRunner  # noqa: E402
from dlm_explorer.backends import build_backend  # noqa: E402
from dlm_explorer.decoding import DecodingParams  # noqa: E402
from dlm_explorer.eval import build_task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--n", type=int, default=20, help="num eval examples")
    ap.add_argument("--n-shot", type=int, default=4)
    ap.add_argument("--gen-len", type=int, default=200)
    ap.add_argument("--order", default="confidence")
    ap.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--batch", type=int, default=1, help="examples per batched forward")
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    backend = build_backend("llada", model_name=args.model)
    print("building GSM8K task ...", flush=True)
    task = build_task("gsm8k", backend=backend, n_examples=args.n,
                      n_shot=args.n_shot, gen_len=args.gen_len)
    runner = TrialRunner(backend, task, batch_size=args.batch)

    print(f"\nGSM8K  n={args.n}  gen_len={args.gen_len}  order={args.order}  batch={args.batch}")
    print(f"{'steps':>6} {'accuracy':>9} {'NFE':>5} {'sec':>7}")
    for s in args.steps:
        t0 = time.time()
        trial = runner.run(DecodingParams(num_steps=s, unmask_order=args.order))
        dt = time.time() - t0
        print(f"{s:>6} {trial.metrics['accuracy']:>9.3f} "
              f"{trial.metrics['model_calls']:>5.0f} {dt:>7.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
