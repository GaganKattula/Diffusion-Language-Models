"""Analyst: turn the trial history into findings.

Computes the step<->quality Pareto frontier (minimize model_calls, maximize
accuracy) and surfaces the headline results the project is after:
  * the Pareto-optimal decoding configs,
  * the best config at each compute budget,
  * the self-correction effect (does enabling remask help, at equal compute?).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..experiment.trial import Trial


def pareto_front(trials: List[Trial], cost="model_calls", quality="accuracy") -> List[Trial]:
    """Return trials not dominated by any other (lower cost AND higher quality)."""
    pts = [(t.metrics.get(cost, float("inf")), t.metrics.get(quality, 0.0), t) for t in trials]
    front = []
    for c, q, t in pts:
        dominated = any(
            (c2 <= c and q2 >= q) and (c2 < c or q2 > q) for c2, q2, _ in pts
        )
        if not dominated:
            front.append(t)
    front.sort(key=lambda t: t.metrics.get(cost, 0.0))
    return front


@dataclass
class Summary:
    n_trials: int
    best: Trial | None
    frontier: List[Trial]
    self_correction_effect: dict
    text: str


class Analyst:
    def __init__(self, objective: str = "accuracy", cost: str = "model_calls") -> None:
        self.objective = objective
        self.cost = cost

    def self_correction_effect(self, trials: List[Trial]) -> dict:
        """Compare remask=True vs remask=False at matched (num_steps, order)."""
        gains = []
        index = {}
        for t in trials:
            key = (t.params.get("num_steps"), t.params.get("unmask_order"))
            index.setdefault(key, {})[bool(t.params.get("remask"))] = t
        for key, by_remask in index.items():
            if True in by_remask and False in by_remask:
                on = by_remask[True].metrics.get(self.objective, 0.0)
                off = by_remask[False].metrics.get(self.objective, 0.0)
                gains.append(on - off)
        if not gains:
            return {"matched_pairs": 0, "mean_gain": None}
        return {"matched_pairs": len(gains), "mean_gain": sum(gains) / len(gains)}

    def summarize(self, trials: List[Trial]) -> Summary:
        if not trials:
            return Summary(0, None, [], {}, "No trials yet.")
        best = max(trials, key=lambda t: t.metrics.get(self.objective, 0.0))
        front = pareto_front(trials, self.cost, self.objective)
        sce = self.self_correction_effect(trials)

        lines = [f"Trials: {len(trials)}"]
        lines.append(
            f"Best {self.objective}={best.metrics.get(self.objective):.3f} "
            f"@ {self.cost}={best.metrics.get(self.cost):.1f}  params={best.params}"
        )
        lines.append(f"Pareto frontier ({len(front)} configs):")
        for t in front:
            lines.append(
                f"  {self.cost}={t.metrics.get(self.cost):.1f}  "
                f"{self.objective}={t.metrics.get(self.objective):.3f}  "
                f"steps={t.params.get('num_steps')} order={t.params.get('unmask_order')} "
                f"remask={t.params.get('remask')}"
            )
        if sce["matched_pairs"]:
            lines.append(
                f"Self-correction effect: mean {self.objective} gain "
                f"{sce['mean_gain']:+.3f} over {sce['matched_pairs']} matched pairs."
            )
        return Summary(len(trials), best, front, sce, "\n".join(lines))
