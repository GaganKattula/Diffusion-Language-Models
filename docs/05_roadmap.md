# 05 — Roadmap

## Phase 0 — Machinery (DONE, validated offline)
- [x] Backend / sampler / eval / agents / ledger / CLI.
- [x] Mock backend reproduces the qualitative effects (steps, order, remask).
- [x] 8 passing tests; end-to-end mock run produces a Pareto frontier.

## Phase 1 — Real model substrate (NEXT, needs a GPU)
The blocker is hardware only — the dev sandbox has no GPU/CUDA and 15 GB RAM,
which cannot hold LLaDA-8B. On a GPU box:

1. `pip install -e ".[llada]"` (torch, transformers, datasets, accelerate).
2. **Verify the checkpoint specifics** in `backends/llada.py`:
   - mask token id (default 126336 — confirm against the model config),
   - whether to use `-Base` or `-Instruct` and the chat template,
   - that `AutoModel(...).logits` is the right forward (some LLaDA releases use a
     custom class via `trust_remote_code=True`).
3. **Implement the GSM8K task** in `eval/tasks.py` (`Gsm8kTask`):
   - load via `datasets.load_dataset("gsm8k", "main")`,
   - tokenize each question with the LLaDA tokenizer; set a fixed `gen_len`,
   - score by extracting the final numeric answer (regex on the decoded text).
4. **Manual de-risk first** (before the agent loop): hand-sweep `num_steps` only
   on ~100 examples and confirm a clean steps↔accuracy curve. If that plots, the
   substrate works.

Command once ready: `dlm-explore run --config configs/llada_gsm8k.yaml`.

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
