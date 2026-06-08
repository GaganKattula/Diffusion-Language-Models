# 04 — The multi-agent auto-research loop

The "auto-research" engine. Each role is a small, single-responsibility object;
together they form a closed propose → evaluate → learn loop over the decoding
space.

## Roles

| agent | input | output | file |
|---|---|---|---|
| **Proposer** | history + search space | next batch of `DecodingParams` | `agents/proposer.py`, `agents/llm.py` |
| **Runner** | one config | a `Trial` (decode every example, score, aggregate) | `agents/runner.py` |
| **Analyst** | full history | Pareto frontier + self-correction effect + summary | `agents/analyst.py` |
| **Orchestrator** | the three above | runs rounds, persists trials | `agents/orchestrator.py` |

```
for round in rounds:
    configs = proposer.propose(history, batch_size)   # decide what to try
    for cfg in configs:
        trial = runner.run(cfg)                        # inference-only eval
        store.append(trial); history.append(trial)     # persist (JSONL)
    summary = analyst.summarize(history)               # learn / report
```

## Two proposer flavours (same interface)

- **Search-based** (default, offline, deterministic):
  - `grid` — exhaustive sweep, in batches.
  - `random` — random sampling of the space, dedup against history.
  - `local` — hill-climb: perturb one field of the current best (focus where the
    signal is). A simple stand-in for "the agent concentrates effort".
- **LLM-driven** (`agents/llm.py`, optional): a Claude model reads the Analyst's
  text summary and proposes the next batch as structured JSON. This is the
  literal "auto-researcher". Defaults to `claude-sonnet-4-6` (token-frugal, one
  call per round). Gated behind `ANTHROPIC_API_KEY`; never used by offline runs.

Because both implement `propose(history, n)`, swapping the heuristic search for
the LLM researcher is a one-line config change (`proposer.name: llm`) — the loop
is identical.

## Why inference-only experiments
The original idea was multi-agent exploration over *training runs* — but each
such experiment costs hours of GPU and gives slow, expensive feedback, the wrong
regime for agents. Pointing the same loop at the **decoding** space makes each
experiment seconds of inference, so the agent gets hundreds of fast cycles on a
modest budget. Same vision, tractable economics.

## Reproducibility / resume
Every `Trial` is appended to a JSONL ledger keyed by a hash of its params. On
restart the loop loads the ledger and skips configs already evaluated, so a
crashed or paused run resumes cleanly and results are fully auditable.

## Cost note for the LLM researcher
The LLM proposer is called once per round (not per trial), and only consumes the
Analyst's compact text summary — so token cost is small and bounded. Sonnet is
the default; raise to Opus only if proposal quality demands it.
