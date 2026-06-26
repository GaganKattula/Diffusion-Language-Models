# 05 — Roadmap

## Phase 0 — Machinery (DONE, validated offline)
- [x] Backend / sampler / eval / agents / ledger / CLI.
- [x] Mock backend reproduces the qualitative effects (steps, order, remask).
- [x] 8 passing tests; end-to-end mock run produces a Pareto frontier.

## Phase 1 — Real model substrate (LARGELY DONE on an A100-80GB pod)
- [x] `bash scripts/pod_setup.sh` installs deps (transformers pinned <5 — 5.x
      breaks LLaDA's custom modeling code) and runs the smoke test.
- [x] **Backend verified on the real model** (`scripts/smoke_llada.py`):
      LLaDA-8B-Base loads, **mask_token_id=126336 confirmed**, vocab_size=126464,
      `model(...).logits` returns `(L, V)`, and the sampler decoded
      *"The capital of France is"* → *" Paris."* end-to-end.
- [x] **GSM8K task implemented** (`eval/tasks.py::Gsm8kTask`): few-shot prompt,
      fixed `gen_len`, gold/pred numeric extraction (unit-tested offline).
- [ ] **Manual de-risk** (`scripts/sweep_steps.py`): hand-sweep `num_steps` on
      ~20 examples and confirm a clean steps↔accuracy curve. ← NEXT
- [ ] Throughput: add **example batching** (backend.logits over (B,L)) before the
      full sweep — diffusion has no KV-cache so each step is a full forward;
      batching examples is the main lever for wall-clock.

Then: `dlm-explore run --config configs/llada_gsm8k.yaml`.

## Phase 2 — The search & the result
- [ ] Run the decoding search (start with `proposer: local`, then `proposer: llm`).
- [ ] Produce the two headline artifacts:
  - the **step↔quality Pareto frontier** plot vs. the default sampler,
  - the **self-correction effect** (remask on/off at matched compute), ideally
    on a task where AR commits early errors (arithmetic / planning).
- [ ] Add a plotting module (`analyst` already yields the frontier; just render).
- [ ] Write up the crisp result for X/LinkedIn.

## Phase 3 — Stretch
- [ ] Semi-AR `block_size` sweep: latency/quality + KV-cache angle.
- [ ] Classifier-free guidance scale (wire `cfg_scale` into the sampler; currently
      omitted — documented as future in `DecodingParams`).
- [ ] Few-step distillation comparison.
- [ ] A second open diffusion LM (e.g. a small SEDD/MDLM checkpoint) to show the
      decoding findings transfer across models — and to enable a *real* (non-mock)
      smoke test on modest hardware.

## Known simplifications to revisit
- The mock backend is a copy-task simulator, intentionally easy (2 steps → 100%);
  it validates machinery, not science. Don't read research conclusions from it.
- `self_corrections` counts value changes after first commit; fine as a signal,
  but the real story should also report accuracy gain at matched compute.
- `cfg_scale` is declared but not implemented.
