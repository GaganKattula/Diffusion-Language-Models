"""Optional LLM-driven proposer/analyst — the "auto-researcher".

This is the agentic-research flavour of the loop: instead of a fixed search
heuristic, a Claude model reads the Analyst's summary of what's been tried and
proposes the next batch of decoding configurations, with a rationale.

It implements the same `propose(history, n)` interface as the search proposers in
proposer.py, so it is a drop-in swap inside the orchestrator.

GATED: requires `pip install anthropic` and ANTHROPIC_API_KEY. It is never used
by the default offline run. Defaulting to Sonnet (claude-sonnet-4-6) to conserve
tokens, since the proposer is called once per round; override via `model`.
"""

from __future__ import annotations

import json
import os
from typing import List

from ..decoding import DecodingParams, UNMASK_ORDERS
from ..experiment.trial import Trial
from .analyst import Analyst

DEFAULT_MODEL = "claude-sonnet-4-6"

_SYSTEM = """You are a research assistant searching the DECODING strategy space of
a masked diffusion language model. Each configuration you propose is evaluated by
inference only (no training). Your objective: find configurations on the
step<->quality Pareto frontier (maximize accuracy, minimize model_calls), and
probe whether self-correction (remask=true) helps at equal compute.

You will be given a summary of trials run so far. Propose the next batch of
distinct, sensible configurations to try. Vary num_steps across the compute range,
compare unmasking orders, and test remask on/off at matched num_steps so the
self-correction effect is measurable. Do not repeat configurations already tried."""

# JSON schema for a batch of decoding configurations (structured output).
_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "configs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "num_steps": {"type": "integer"},
                    "unmask_order": {"type": "string", "enum": list(UNMASK_ORDERS)},
                    "temperature": {"type": "number"},
                    "remask": {"type": "boolean"},
                    "remask_frac": {"type": "number"},
                    "remask_threshold": {"type": "number"},
                    "block_size": {"type": ["integer", "null"]},
                    "seed": {"type": "integer"},
                },
                "required": ["num_steps", "unmask_order", "temperature", "remask"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "configs"],
    "additionalProperties": False,
}


class LLMProposer:
    def __init__(self, model: str = DEFAULT_MODEL, analyst: Analyst | None = None) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLMProposer needs ANTHROPIC_API_KEY. Use a search proposer "
                "(grid/random/local) for offline runs."
            )
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic to use the LLM proposer.") from e
        self.client = anthropic.Anthropic()
        self.model = model
        self.analyst = analyst or Analyst()

    def propose(self, history: List[Trial], n: int) -> List[DecodingParams]:
        summary = self.analyst.summarize(history).text
        user = (
            f"Trials so far:\n{summary}\n\n"
            f"Propose {n} new distinct configurations as JSON matching the schema."
        )
        # Adaptive thinking + structured JSON output per the Anthropic Messages API.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        out = []
        for c in data["configs"][:n]:
            out.append(DecodingParams(**{k: c[k] for k in c if k in DecodingParams().to_dict()}))
        return out
