"""Append-only JSONL ledger of trials.

Every evaluated configuration is appended as one JSON line, so a run is fully
reproducible and inspectable after the fact, and a crashed run can be resumed by
reading back what was already tried.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .trial import Trial


class TrialStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trial: Trial) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(trial.to_dict(), sort_keys=True) + "\n")

    def load(self) -> List[Trial]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Trial.from_dict(json.loads(line)))
        return out

    def seen_ids(self) -> set[str]:
        return {t.trial_id for t in self.load()}
