#!/usr/bin/env python3
"""Render the shareable figure from a trial ledger.

    python scripts/plot_results.py --ledger runs/llada_gsm8k.jsonl

Produces a two-panel PNG:
  (left)  step <-> quality Pareto frontier — every config scored by compute
          (model_calls / NFE, log x) vs accuracy, with the Pareto frontier
          traced and remask on/off colour-coded.
  (right) self-correction effect — accuracy with remask off vs on at matched
          (num_steps, order, block) settings, plus the mean gain.

Reuses the Analyst's frontier / self-correction logic so the figure matches the
numbers the agent loop reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dlm_explorer.agents.analyst import Analyst, pareto_front  # noqa: E402
from dlm_explorer.experiment.store import TrialStore  # noqa: E402

OFF_C, ON_C = "#4878a8", "#d9534f"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cost", default="model_calls")
    ap.add_argument("--objective", default="accuracy")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    trials = TrialStore(args.ledger).load()
    if not trials:
        print(f"no trials in {args.ledger}")
        return 1
    cost, obj = args.cost, args.objective

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel 1: Pareto frontier ---
    xs = [t.metrics[cost] for t in trials]
    ys = [t.metrics[obj] for t in trials]
    cols = [ON_C if t.params.get("remask") else OFF_C for t in trials]
    ax1.scatter(xs, ys, c=cols, alpha=0.65, s=45, edgecolor="white", linewidth=0.5, zorder=3)

    front = pareto_front(trials, cost, obj)
    ax1.plot([t.metrics[cost] for t in front], [t.metrics[obj] for t in front],
             "-o", color="black", lw=2, ms=6, zorder=4)
    for t in front:  # annotate frontier points with their step count
        ax1.annotate(f"{t.params.get('num_steps')}st",
                     (t.metrics[cost], t.metrics[obj]),
                     textcoords="offset points", xytext=(4, 6), fontsize=8)

    ax1.set_xscale("log")
    ax1.set_xlabel(f"{cost} (NFE, log scale) — lower is cheaper")
    ax1.set_ylabel(obj)
    ax1.set_title("Step ↔ quality Pareto frontier")
    ax1.grid(True, alpha=0.25)
    ax1.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OFF_C, label="remask off", ms=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ON_C, label="remask on", ms=8),
        Line2D([0], [0], color="black", marker="o", label="Pareto frontier"),
    ])

    # --- panel 2: self-correction matched pairs ---
    index: dict = {}
    for t in trials:
        k = (t.params.get("num_steps"), t.params.get("unmask_order"), t.params.get("block_size"))
        index.setdefault(k, {})[bool(t.params.get("remask"))] = t.metrics[obj]
    pairs = sorted((k, v[False], v[True]) for k, v in index.items() if True in v and False in v)

    if pairs:
        labels = [f"{k[0]}·{k[1][:4]}" for k, _, _ in pairs]
        off = [o for _, o, _ in pairs]
        on = [n for _, _, n in pairs]
        x = np.arange(len(pairs))
        w = 0.38
        ax2.bar(x - w / 2, off, w, label="remask off", color=OFF_C)
        ax2.bar(x + w / 2, on, w, label="remask on", color=ON_C)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        gain = float(np.mean([n - o for _, o, n in pairs]))
        ax2.set_title(f"Self-correction effect (mean Δ{obj} = {gain:+.3f})")
        ax2.set_ylabel(obj)
        ax2.set_xlabel("num_steps · unmask_order")
        ax2.grid(True, axis="y", alpha=0.25)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "no matched remask on/off pairs yet", ha="center", va="center")
        ax2.set_title("Self-correction effect")

    fig.suptitle(args.title or f"LLaDA decoding search — {Path(args.ledger).stem}", fontweight="bold")
    fig.tight_layout()

    out = args.out or str(Path(args.ledger).with_suffix(".png"))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print("\n" + Analyst(obj, cost).summarize(trials).text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
