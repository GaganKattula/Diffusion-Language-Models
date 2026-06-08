# 03 — Architecture

The design rule everything follows from:

> **The backend returns logits. The sampler owns the denoising loop.**

That split is why the decoding search is independent of the model — swap mock ↔
LLaDA and the exact same Proposer/Runner/Analyst loop applies.

```
                 ┌─────────────────────────────────────────────┐
                 │            AutoResearchLoop (agents)         │
                 │                                              │
   SearchSpace ──┤  Proposer.propose(history, n) ──► [configs]  │
                 │        │                                     │
                 │        ▼                                     │
                 │  Runner.run(config) ─ decode+score ─► Trial ─┼─► TrialStore (JSONL)
                 │        │     uses                            │
                 │        ▼                                     │
                 │  MaskedDiffusionSampler.generate(...)        │
                 │        │     calls                           │
                 │        ▼                                     │
                 │  Backend.logits(seq, prompt_len) ─► (L,V)    │
                 │        │                                     │
                 │        ▼                                     │
                 │  Analyst.summarize(history) ─► Pareto + SCE  │
                 └─────────────────────────────────────────────┘
```

## Modules

| module | responsibility |
|---|---|
| `backends/base.py` | `MaskedDiffusionBackend` ABC: `encode/decode/logits` |
| `backends/mock.py` | weight-free simulator (copy task) for offline validation |
| `backends/llada.py` | real LLaDA via HF; torch imported lazily; **gated on GPU** |
| `decoding/sampler.py` | `DecodingParams` (the search space) + the denoising loop |
| `eval/tasks.py` | `Task` interface; `CopyTask` (mock), `Gsm8kTask` (Phase 2) |
| `experiment/trial.py`, `store.py` | `Trial` record + append-only JSONL ledger (resume) |
| `agents/proposer.py` | `SearchSpace` + grid/random/local proposers |
| `agents/runner.py` | run one config over all examples → aggregated `Trial` |
| `agents/analyst.py` | `pareto_front` + self-correction effect + text summary |
| `agents/orchestrator.py` | the round loop tying it together |
| `agents/llm.py` | optional Claude-backed proposer (same `propose` interface) |
| `config.py` | YAML → wired `AutoResearchLoop` |
| `cli.py` | `validate` / `plan` / `run` |
| `registry.py` | name→class registries for backends/tasks/proposers |

## Key contracts

- **Backend**: `logits(input_ids: np.ndarray (L,), prompt_len: int) -> (L, V)`.
  `prompt_len` lets the mock locate the generation region; real models ignore it.
- **DecodingParams** is frozen and hashes to a stable `trial_id`, so the ledger
  dedupes and the loop resumes without re-running configs.
- **Cost metric** = `num_model_calls` (NFE), counted in the sampler. This is the
  x-axis of the Pareto frontier and the honest proxy for latency.

## Why a mock backend at all
It makes the whole pipeline runnable and *testable* with no GPU, no network, no
weights — so orchestration bugs are caught here, deterministically, before
spending GPU time. It is a **simulator of the machinery, not a language model**;
the real signal comes from LLaDA (Phase 2).
