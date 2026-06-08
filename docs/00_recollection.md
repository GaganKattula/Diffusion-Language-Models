# 00 — Recollection (START HERE)

This is the single-page state of the project. If you (or a fresh session) read
only one file, read this.

## The thesis

Diffusion counterparts to transformer LLMs already exist and scale (LLaDA-8B
rivals LLaMA3-8B). The frontier question is no longer "do they work" but **"how
should we decode them?"** — a large, underexplored design space. This project
points a **multi-agent auto-research loop** at that space:

> Search the decoding strategy of an open masked diffusion LM (LLaDA) to find
> (a) the **step↔quality Pareto frontier** that beats the default sampler, and
> (b) a **self-correction regime** where the model fixes its own mistakes —
> something an autoregressive model structurally cannot do.

Why this shape: each experiment is *inference only* (seconds), so the agent can
run hundreds of cycles on a few cloud GPUs — keeping the original "multi-agent
parameter exploration / auto-research" vision while making it tractable. The
narrative target is a crisp, shareable empirical result (X/LinkedIn), not a paper.

## The search space (what the agent explores)

A single configurable sampler (`decoding/sampler.py`) whose `DecodingParams`
*is* the parameter space:

| knob | meaning |
|---|---|
| `num_steps` | denoising steps per block — the compute/latency axis |
| `unmask_order` | confidence / margin / entropy / random / left_to_right |
| `remask` (+ frac, threshold) | self-correction: re-mask low-confidence commits |
| `block_size` | None = pure diffusion; N = semi-autoregressive blocks |
| `temperature` | 0 = argmax, >0 = sample |

The backend only returns per-position logits; the sampler owns the whole
denoising loop — so the same search runs on the mock and on real LLaDA unchanged.

## What is built and VALIDATED (offline, no GPU)

- **Mock backend** (`backends/mock.py`): a deterministic, weight-free *simulator*
  of a masked diffusion LM on a copy task. NOT a language model — it exists so
  the machinery can be proven without a GPU. It reproduces the qualitative
  effects we care about (more steps help; confidence order > random; remask
  recovers accuracy under weak orders).
- **Sampler** (`decoding/sampler.py`): the full denoising loop incl. semi-AR
  blocks and self-correction; counts `model_calls` (NFE) as the cost metric.
- **Eval** (`eval/tasks.py`): `CopyTask` with token/sequence accuracy.
- **Agents** (`agents/`): `Proposer` (grid/random/local + optional LLM),
  `Runner`, `Analyst` (Pareto frontier + self-correction effect), `Orchestrator`
  (the loop), with a JSONL ledger (`experiment/store.py`) that supports resume.
- **8 passing tests** (`tests/`) asserting the qualitative behaviours above.
- End-to-end run on the mock: 60 trials, a Pareto frontier (1 call→0.75 acc,
  2 calls→1.0 acc on the easy copy task), self-correction effect computed.

Run it: `pytest` then `dlm-explore run --config configs/mock_steps_pareto.yaml`.

## What is built but NOT yet run

- **Real LLaDA backend** (`backends/llada.py`): wired to load
  `GSAI-ML/LLaDA-8B-*` via HuggingFace and return logits. **Needs a GPU** — the
  dev sandbox has none (15 GB RAM, no CUDA), so it has not been executed. The
  search machinery is identical; only `--backend`/config changes.
- **LLM auto-researcher** (`agents/llm.py`): a Claude model reads the Analyst
  summary and proposes the next batch (structured-output JSON). Defaults to
  `claude-sonnet-4-6` to conserve tokens. Gated behind `ANTHROPIC_API_KEY`.

## What is still TODO (Phase 2)

1. Implement the real **GSM8K task** loader (currently a guarded placeholder).
2. Run the real LLaDA decoding search on a GPU box (see `configs/llada_gsm8k.yaml`).
3. Add plotting (the Pareto frontier figure is the shareable artifact).
4. Validate LLaDA's exact mask-token id / chat template against the checkpoint.

See `docs/05_roadmap.md` for the full plan and the exact commands.

## The honest blocker

We cannot run the real 8B model in the current environment — **no GPU**. Every
other piece is in place and tested. The moment a GPU + weights are available,
`dlm-explore run --config configs/llada_gsm8k.yaml` (after the GSM8K loader is
filled in) is the single command to start the real search.
