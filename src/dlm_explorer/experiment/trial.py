"""A Trial is one evaluated decoding configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Trial:
    trial_id: str
    params: dict          # DecodingParams.to_dict()
    metrics: dict         # aggregated across the task's examples
    meta: dict = field(default_factory=dict)

    @staticmethod
    def make_id(params: dict) -> str:
        blob = json.dumps(params, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trial":
        return cls(trial_id=d["trial_id"], params=d["params"], metrics=d["metrics"], meta=d.get("meta", {}))
