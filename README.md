# Diffusion-Language-Models

An **agentic auto-research harness** that searches the *decoding* strategy space
of a masked diffusion language model (LLaDA) and surfaces:

1. the **step ↔ quality Pareto frontier** — decoding configs that beat the
   default sampler at equal or lower compute, and
2. the **self-correction effect** — where re-masking lets the model fix its own
   mistakes, the structural move autoregressive LLMs cannot make.

A multi-agent loop (Proposer → Runner → Analyst) proposes decoding
configurations, evaluates each by *inference only* (cheap, fast feedback), and
reads back what's working — "auto-research" over the inference design space
rather than expensive training runs.

> **New here? Read [`docs/00_recollection.md`](docs/00_recollection.md) first.**
> It is the single-page state-of-the-project summary: the thesis, what's built,
> what's validated, and what's pending.

## Status

| | |
|---|---|
| Pipeline (mock backend) | ✅ runs end-to-end, no GPU/network — validated by tests |
| Auto-research loop | ✅ Proposer/Runner/Analyst/Orchestrator + JSONL ledger |
| Real LLaDA backend | ✅ wired, ⛔ not yet run (needs a GPU — see Roadmap) |
| LLM "auto-researcher" | ✅ implemented (optional), defaults to `claude-sonnet-4-6` |
| Real benchmark task (GSM8K) | 🚧 placeholder — Phase 2 |

Nothing calls a real model until you explicitly run a `llada` config on a GPU.

## Quickstart (offline, no GPU)

```bash
pip install -e .            # core deps: numpy, pyyaml
pip install pytest && pytest        # 8 tests, ~0.3s
dlm-explore validate                # import + tiny smoke test
dlm-explore plan --config configs/mock_steps_pareto.yaml   # print plan, run nothing
dlm-explore run  --config configs/mock_steps_pareto.yaml   # run the loop on the mock
```

## Running the real model (Phase 2, needs a GPU)

```bash
pip install -e ".[llada]"          # torch, transformers, datasets, accelerate
# implement the GSM8K loader in src/dlm_explorer/eval/tasks.py (see docs/05_roadmap.md)
dlm-explore run --config configs/llada_gsm8k.yaml
```

## Layout

```
src/dlm_explorer/
  backends/    mock.py (offline simulator) · llada.py (real, gated) · base.py
  decoding/    sampler.py  — the configurable denoising loop = the search space
  eval/        tasks.py    — CopyTask (mock) · Gsm8kTask (Phase 2) · metrics
  agents/      proposer · runner · analyst · orchestrator · llm (auto-researcher)
  experiment/  trial · store (JSONL ledger)
  config.py · cli.py
configs/       mock_steps_pareto.yaml · llada_gsm8k.yaml
docs/          00_recollection (START HERE) · 01..05
tests/         offline, mock-backed
```

See [`docs/03_architecture.md`](docs/03_architecture.md) for how the pieces fit.
