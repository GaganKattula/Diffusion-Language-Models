"""Command-line entry point.

    dlm-explore validate                 # import + tiny mock smoke test (no run)
    dlm-explore plan   --config C.yaml   # print the resolved plan, run NOTHING
    dlm-explore run    --config C.yaml   # execute the auto-research loop

`plan` is deliberately side-effect-free so a real run is never kicked off by
accident — you inspect the plan, then explicitly `run`.
"""

from __future__ import annotations

import argparse
import sys

from .config import ExperimentConfig


def _cmd_validate(_args) -> int:
    from .backends import build_backend
    from .decoding import DecodingParams, MaskedDiffusionSampler

    backend = build_backend("mock")
    sampler = MaskedDiffusionSampler(backend)
    res = sampler.generate([2, 3, 4, 5], gen_len=4, params=DecodingParams(num_steps=8))
    print(f"OK: mock backend + sampler import and run. backend={backend.describe()}")
    print(f"    sample gen tokens={res.gen_tokens} model_calls={res.num_model_calls}")
    return 0


def _cmd_plan(args) -> int:
    cfg = ExperimentConfig.load(args.config)
    space = cfg.build_space()
    grid = space.grid()
    print(f"Experiment : {cfg.name}")
    print(f"Backend    : {cfg.backend}")
    print(f"Task       : {cfg.task}")
    print(f"Proposer   : {cfg.proposer}")
    print(f"Loop       : {cfg.loop}")
    print(f"Objective  : maximize {cfg.objective}, minimize {cfg.cost}")
    print(f"Search space: {len(grid)} grid points")
    print(f"Output     : {cfg.output_dir}/{cfg.name}.jsonl")
    print("\nNothing was executed. Run with:  dlm-explore run --config", args.config)
    return 0


def _cmd_run(args) -> int:
    cfg = ExperimentConfig.load(args.config)

    def on_round(r, summary):
        print(f"\n===== round {r} =====\n{summary.text}")

    loop = cfg.build_loop(on_round=on_round)
    final = loop.run()
    print("\n##### FINAL #####")
    print(final.text)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dlm-explore", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="import + tiny mock smoke test").set_defaults(fn=_cmd_validate)
    pp = sub.add_parser("plan", help="print the resolved plan, run nothing")
    pp.add_argument("--config", required=True)
    pp.set_defaults(fn=_cmd_plan)
    pr = sub.add_parser("run", help="execute the auto-research loop")
    pr.add_argument("--config", required=True)
    pr.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
