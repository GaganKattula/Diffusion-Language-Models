# Diffusion-LM Decoding Search — Project Breakdown & Learning Guide

A complete, self-contained account of this project: the idea, the science, the
engineering, the results, and the road ahead — written to be *learned from*, so
you can reproduce the reasoning and write your own version before publishing.

It is deliberately two-voiced:
- 🔬 **Researcher lens** — what question we asked, why it's interesting, what we
  found, and how confident we should be.
- 🛠️ **Engineer lens** — how the harness is built, the decisions and trade-offs,
  and the war stories (the bugs and infra failures, which are half the lesson).

> **How to use this doc.** Read Parts 1–3 for the story and the result. Read
> Part 4–5 for the system. Use Part 7 (roadmap) to decide what to do next, and
> Part 8 (reading) whenever a concept is unfamiliar. Part 9 is the pre-publish
> checklist — read it *before* you post anything.

---

## Part 1 — Executive summary

**Premise.** Diffusion language models (DLMs) are a non-autoregressive
alternative to transformer LLMs: instead of generating left-to-right, they start
from an all-`[MASK]` sequence and *iteratively unmask* it over several denoising
steps. Open 8B DLMs (LLaDA) now rival similarly-sized autoregressive LLMs. The
frontier question is no longer "do they work" but **"how should we decode
them?"** — a large, under-explored design space.

**What we built.** A multi-agent "auto-research" harness that searches the
*decoding* strategy space of LLaDA-8B by inference alone (cheap, seconds per
experiment), measuring accuracy on GSM8K as we vary: number of denoising steps,
the *order* in which positions are unmasked, whether we allow re-masking
("self-correction"), and semi-autoregressive block size.

**Headline result (n=100, GSM8K, LLaDA-8B-Base, 4-shot).**
> **How you unmask decides whether extra denoising compute is worth anything.**
> Confidence-ordered unmasking and random unmasking are *tied* at low step
> counts, then diverge sharply: at 128 steps, confidence reaches **0.52** while
> random plateaus at **0.20** — a **2.6×** gap at identical compute. Random
> ordering wastes the extra steps; confidence ordering converts them into
> accuracy.

**Second result (negative, and honest).** A naive self-correction rule
(re-mask committed tokens whose probability drops below a threshold) does **not**
reliably help: +0.03 at 64 steps but **−0.07 at 128 steps** (mean −0.02). The
promising +0.05 we saw in a small n=20 pilot was **noise** — the controlled
n=100 experiment killed it. Smarter remasking might help; this rule doesn't.

---

## Part 2 — 🔬 The idea and why it matters

### 2.1 The one-paragraph theory you need
A masked diffusion LM is trained like BERT taken to its logical extreme: mask a
random fraction of tokens (the fraction is the "time" / noise level) and predict
the originals. At generation time you run the reverse process — start fully
masked and progressively commit tokens over `T` denoising steps. Crucially,
**everything about *how* you commit is a free choice at inference**: how many
steps, which positions to commit first, whether to un-commit ("remask") a token
you later doubt, and whether to decode the sequence in one bidirectional block or
in left-to-right sub-blocks. Models ship with *a* default, almost never proven
optimal across tasks. That gap is the project.

### 2.2 Why this framing is the right bet
The original ambition — "multi-agent exploration of training runs" — is the
wrong economics: each experiment is a multi-hour GPU training job, slow and
expensive, the worst possible feedback loop for an agent. Re-pointing the same
propose→evaluate→analyze loop at the **decoding** space makes each experiment a
few seconds of *inference*, so an agent can run hundreds of cycles on a modest
budget. Same vision, tractable.

### 2.3 The scientific questions (what a paper/thread would claim)
1. **Does decoding order matter, and how much, at fixed compute?** (Answered:
   yes, hugely — see Part 3.)
2. **Is the step↔quality trade-off order-dependent?** (Answered: yes — order
   determines whether more steps help at all.)
3. **Does self-correction (a diffusion-only move) help?** (Answered for a naive
   rule: no. Open for smarter rules.)
4. **Where is the compute knee** — the cheapest config that captures most of the
   quality? (Answered: ~64 steps + confidence ≈ 0.34–0.40, ~2/3 of the 128-step
   result at half the compute.)

### 2.4 Lineage / related work (know these before you publish)
D3PM (Austin et al., 2021) → Diffusion-LM (Li et al., 2022) → SEDD (Lou et al.,
2024) → MDLM / MD4 (Sahoo et al., 2024; Shi et al., 2024) → LLaDA-8B (Nie et al.,
2025) → block diffusion / BD3-LM (Arriola et al., 2025). Most of that work is
about *training* these models; systematic study of *how to decode* them is
comparatively thin — which is the opening this project uses. (Full reading list:
Part 8.)

---

## Part 3 — 🔬 What we found (results + interpretation)

All numbers: LLaDA-8B-Base, GSM8K test, 4-shot, gen_len=200, single-block,
temperature 0 (argmax), seed 0. "steps" = denoising steps = NFE (forward passes)
per example. Accuracy = exact match on the extracted final integer.

### 3.1 The de-risk curve (confidence order only)
First we confirmed the substrate behaves sanely — accuracy rises with steps:

| steps | accuracy |
|------:|---------:|
| 16 | 0.05 |
| 32 | 0.225 |
| 64 | 0.35 |
| 128 | 0.425 |

Monotonic, diminishing returns, knee ~32–64. (These early numbers were slightly
understated by a batching bug later fixed — see Part 4.6.) This alone validated
the core thesis: **decoding compute trades off against quality**.

### 3.2 Finding 1 — unmask order gates compute (the headline, n=100)

| steps | **confidence** | **random** |
|------:|---------------:|-----------:|
| 16 | 0.04 | 0.07 |
| 32 | 0.21 | 0.19 |
| 64 | **0.34** | 0.17 |
| 128 | **0.52** | 0.20 |

Read it as two curves (see `runs/llada_confirm.png`): **tied below ~24 steps,
then they diverge.** Confidence keeps climbing; random plateaus around 0.20.

**Interpretation.** Extra denoising steps only help if each step commits the
*right* tokens. Confidence ordering commits the model's most-certain positions
first, which seeds context that sharpens the rest — a virtuous cascade. Random
ordering commits arbitrary positions, so more steps just re-shuffle noise. **The
value of compute is unlocked by the ordering policy**, not by the step count
alone. This is the non-obvious, shareable claim, and at n=100 the 0.52-vs-0.20
gap (32 examples) is far outside sampling noise.

### 3.3 Finding 2 — naive self-correction doesn't help (negative, n=100)
Confidence order, matched remask off/on:

| steps | remask off | remask on | Δ |
|------:|-----------:|----------:|----:|
| 64 | 0.34 | 0.37 | **+0.03** |
| 128 | 0.52 | 0.45 | **−0.07** |

Mean Δ = −0.02. The rule (remask committed tokens whose current probability
< 0.5, up to 25% of them per step) helps marginally at 64 steps but *hurts* at
128. **Why plausibly:** with enough steps + a good order the model already
commits well; re-masking near-final, well-committed tokens mostly disrupts good
predictions. The +0.03 at 64 is within noise (3 examples at n=100).

**The methodological lesson (important).** Our n=20 pilot showed remask helping
the best config (+0.05). The controlled n=100 run *falsified* it. Small-n
"wins" on the frontier are frequently noise; the dedicated matched-pair
experiment is what settles it. Report this honestly — a clean negative result is
worth more than an over-claimed positive.

### 3.4 The compute knee
`64 steps + confidence` (0.34) buys ~65% of the best result (0.52 @ 128) at half
the compute. If you cared about latency you'd sit near here. The Pareto frontier
(minimize steps, maximize accuracy) is: 16/random 0.07 → 32/confidence 0.21 →
64/confidence 0.34 → 128/confidence 0.52.

### 3.5 ⚠️ Threats to validity — READ BEFORE PUBLISHING
Be precise about what you claim. Honest caveats:
- **Confidence intervals.** n=100 → a binomial 95% CI is roughly ±8–10 points.
  0.52 vs 0.20 is unambiguous; **0.34 vs 0.37 is not** (overlapping CIs). Quote
  CIs, or bump n to 500+ for the numbers you headline.
- **One model, one task, one prompt, one seed.** LLaDA-8B-Base, GSM8K, 4-shot,
  seed 0. Do **not** claim generality. "On LLaDA-8B/GSM8K" is the honest scope.
- **"Beats the default" needs care.** LLaDA's *actual* recommended decoding is a
  semi-autoregressive, low-confidence-remasking scheme. We implement a *related
  but not identical* sampler. Our robust claim is **confidence ≫ random ordering
  at equal compute**, which is real regardless. Avoid implying we beat LLaDA's
  tuned decoder unless you replicate and compare it.
- **Answer extraction is heuristic** (last number, cut at a hallucinated next
  "Question:"). It can mis-grade; spot-check a sample before quoting exact
  accuracies.
- **Fixed gen_len=200, single block.** Semi-AR (`block_size`) was excluded from
  the headline run to control cost; it's a separate experiment (Part 7).
- **Absolute accuracies are modest** (base model, naive decoding, small n). The
  *comparisons* are the result, not the absolute 0.52.
- **Prior art exists — frame novelty carefully.** Confidence-based iterative
  decoding is the core idea of **MaskGIT** (2022); confidence-threshold parallel
  decoding is in **Fast-dLLM** (2025); remasking / self-correction with
  inference-time scaling is **ReMDM** (2025). So "confidence beats random" and
  "remasking as a knob" are *not new mechanisms*. Our honest contribution is the
  **controlled quantification on LLaDA-8B / GSM8K** — the diverging step↔quality
  curves by order, and a **negative result** for a naive remasking rule. Read
  ReMDM before making any self-correction claim (they show remasking *can* help
  with a principled schedule — our result says a naive threshold rule doesn't,
  which is consistent, not contradictory). See `docs/08_reading.md` §5.

---

## Part 4 — 🛠️ How it's built (architecture + war stories)

### 4.1 The one design rule
> **The backend returns logits. The sampler owns the denoising loop.**

`backends/base.py` defines a `MaskedDiffusionBackend` whose only real job is
`logits(input_ids, prompt_len) -> (L, V)`. Everything about steps/order/remask/
blocks lives in the sampler. Consequence: the *identical* search runs on the
mock and on real LLaDA — you swap the backend, nothing else. This decoupling is
what makes the decoding space searchable independent of the model.

### 4.2 The mock backend — why a fake model is essential
`backends/mock.py` is a deterministic, weight-free *simulator* of a masked
diffusion LM on a copy task (the generation region must reproduce the prompt). It
is **not** a language model. Its purpose: make the whole pipeline runnable and
*testable* with zero GPU/network, while reproducing the qualitative effects that
matter (more steps help; confidence order beats random; remask fixes early
mistakes under a weak order). Every orchestration bug is caught here,
deterministically, before spending a cent on GPU.

### 4.3 The sampler — the search space as code
`decoding/sampler.py` implements one configurable loop whose `DecodingParams`
*is* the search space: `num_steps`, `unmask_order` ∈ {confidence, margin,
entropy, random, left_to_right}, `remask` (+ threshold/frac), `block_size`,
`temperature`. Each step: get logits → score masked positions by the order
policy → commit the top-k (schedule commits everything by the last step) →
optionally remask low-probability commits. `block_size` splits the region into
left-to-right sub-blocks (semi-AR). The cost metric is `model_calls` (NFE) — the
honest x-axis of every Pareto plot.

### 4.4 Evaluation
`eval/tasks.py`: `CopyTask` (mock, synthetic, deterministic ground truth) and
`Gsm8kTask` (real; few-shot prompt, fixed gen_len, regex answer extraction).
Tasks take a backend so they can tokenize with the real tokenizer.

### 4.5 The auto-research loop
`agents/`: `Proposer` (grid / random / local hill-climb, plus an optional
Claude-driven "LLM researcher"), `Runner` (decode all examples → aggregate a
`Trial`), `Analyst` (Pareto frontier + self-correction effect + text summary),
`Orchestrator` (rounds, with a per-config progress heartbeat). Every `Trial` is
appended to a JSONL ledger (`experiment/store.py`) that supports resume — which
is exactly what saved us when the pod died mid-run.

### 4.6 War stories (the engineering half of the lesson)
These are where most of the real learning is:

1. **`transformers` 5.x broke LLaDA's custom model.** The community modeling
   code targets the 4.x loader; 5.x raised `AttributeError: 'LLaDAModelLM' has no
   attribute 'all_tied_weights_keys'`. **Fix:** pin `transformers<5` (4.49.0).
   *Lesson:* pin versions for `trust_remote_code` models; the model card's tested
   stack is load-bearing.

2. **The serial sampler "hung."** It did a full **126,464-vocab softmax + sort on
   the CPU every step** — billions of ops/step. **Fix:** move the reduction
   on-device — `backend.predict_batch` computes softmax/top-2/entropy/gather on
   the GPU and returns only small `(B, L)` summaries. This was the single biggest
   speedup. *Lesson:* never move a full `(L, V)` logit tensor to host if you only
   need argmax + a confidence scalar.

3. **Batching corrupted accuracy.** Right-padding unequal-length prompts changed
   LLaDA's outputs (0.375 vs 0.250 on the same 8 examples) — its bidirectional
   attention didn't honor the pad mask. **Fix:** bucket batches by `(gen_len,
   prompt_len)` so every batch is equal-length with *no padding*; a batched
   forward over equal-length all-real sequences equals N single forwards.
   *Lesson:* verify batched==serial *accuracy* on the real model; padding is a
   classic silent corruptor. (We also proved batched==serial bit-for-bit on the
   mock via an equivalence test — but the mock can't catch a model that ignores
   the attention mask.)

4. **Batching barely helped wall-clock.** At ~900-token sequences the forward is
   already compute-bound, so batching examples gave little speedup. *Lesson:* know
   whether you're memory- or compute-bound before optimizing; the on-device
   reduction (war story 2) was the real win, not batch size.

5. **The pod died mid-run (out of funds).** No data lost: the JSONL ledger is
   append-as-you-go and lived on a **persistent network volume** (weights cache
   too). Recovery = new pod in the same datacenter + attach volume. *Lesson:*
   append-only logs + persistent storage make long runs crash-safe; design for
   the run dying.

---

## Part 5 — 🛠️ Reproducing everything

```bash
# offline (no GPU): prove the machinery
pip install -e . && pytest                         # 80+ tests incl. batched-equivalence
dlm-explore run --config configs/mock_steps_pareto.yaml

# on a GPU pod (A100/L40S, >=24GB), weights ~16GB:
export HF_HOME=/workspace/hf                        # persist the cache on the volume
pip install -e . && pip install "transformers==4.49.0" accelerate datasets matplotlib
python scripts/smoke_llada.py                       # verify the backend loads + runs
python scripts/sweep_steps.py --n 20 --steps 16 32 64 128   # steps<->accuracy de-risk
bash scripts/run_publishable.sh                     # the two headline runs, back-to-back
python scripts/plot_results.py --ledger runs/llada_confirm.jsonl --color-by unmask_order
```
Configs: `llada_gsm8k.yaml` (broad hill-climb search), `llada_confirm.yaml`
(order-vs-compute headline), `llada_selfcorrect.yaml` (matched remask pairs).

---

## Part 6 — 🛠️🔬 The multi-agent loop, in depth
`AutoResearchLoop` runs, per round: `Proposer.propose(history, k)` → for each
config `Runner.run(cfg)` (decode+score+aggregate) → append to ledger →
`Analyst.summarize(history)`. Proposers share one interface, so swapping the
grid/hill-climb search for the Claude "researcher" (`agents/llm.py`, defaults to
Sonnet, structured-JSON output) is a one-line config change. This is the
"auto-research" engine: it proposes decoding configs, evaluates them by
inference, and reads back the Pareto frontier — the same loop you'd point at any
cheap-to-evaluate design space.

---

## Part 7 — 🔬🛠️ Roadmap ahead

### 7.1 Solidify the headline (do this before publishing)
- [ ] **Re-run at n=500** for the confidence-vs-random curve → tight CIs on the
      numbers you'll quote. (🛠️ just bump `n_examples`; ~2–3 h GPU.)
- [ ] **Add margin & entropy** to the order comparison at n=500 (is "confidence"
      special, or is any *informative* order enough?). Entropy underperformed at
      64 but recovered at 128 — worth a clean look.
- [ ] **Hand-verify answer extraction** on ~30 samples; report extraction error.
- [ ] **Bootstrap confidence intervals** in the Analyst and on the plots.

### 7.2 New research questions
- [ ] **Semi-autoregressive blocks** (`block_size` ∈ {32,64,128,None}): the
      latency/quality/KV-cache angle. Note our sampler runs `num_steps` *per
      block*, so account for the NFE multiplier (docs/05_roadmap.md).
- [ ] **Smarter self-correction.** The naive threshold rule failed; try:
      early-only remasking, entropy-triggered remasking, or remasking budget that
      decays with step. This is where the "diffusion can revise" story could still
      be won — or cleanly closed.
- [ ] **Classifier-free guidance** (`cfg_scale`, currently stubbed) for masked
      diffusion.
- [ ] **Transfer:** does "confidence gates compute" hold on a second open DLM
      (a small SEDD/MDLM) and a second task (HumanEval-infilling, where diffusion
      should structurally shine)? Transfer is what turns a finding into a claim.
- [ ] **Compare to LLaDA's official decoder** (semi-AR + low-confidence remask)
      so any "beats default" claim is grounded.

### 7.3 Engineering hardening
- [ ] Bootstrap CIs + significance tests in `Analyst`.
- [ ] Batched inference that survives unequal lengths (correct position_ids /
      a mask format LLaDA honors) if larger effective batches are ever needed.
- [ ] Cache the few-shot **prefix** across denoising steps if a correct
      prefix-KV scheme exists for the bidirectional model (big potential speedup).
- [ ] Wire the LLM proposer for a real agentic run and compare its sample
      efficiency to grid/hill-climb.
- [ ] A `RESULTS.md` that regenerates every figure from committed ledgers.

### 7.4 Suggested publishing arc
1. n=500 confidence-vs-random figure + one-paragraph explanation (the hook).
2. The honest self-correction negative + the methodology note (credibility).
3. The compute-knee / Pareto framing (practical takeaway).
4. Link the repo; note scope and caveats (Part 3.5) up front.

---

## Part 8 — Prerequisite reading

The full, leveled reading list — a **research track** (the ideas: diffusion
fundamentals → discrete/masked diffusion → LLaDA/block diffusion → decoding →
evaluation → Pareto framing → automation) and an **engineering track** (packaging,
NumPy vectorization, PyTorch/HF inference, diffusion-inference specifics, experiment
infra, testing, cloud-GPU workflow, matplotlib) — lives in
**[`docs/08_reading.md`](08_reading.md)**. Each entry says what to extract and its
difficulty. The single most on-point reads for *this* project: **MaskGIT** (the
origin of confidence-based iterative decoding), **ReMDM** (remasking /
self-correction with inference-time scaling), and **LLaDA** (the model itself).

---

## Part 9 — Pre-publish checklist
- [ ] Every quoted number has an n and a CI (or is a large, unambiguous gap).
- [ ] Scope stated up front: model, task, prompt, seed.
- [ ] No "beats the default / SOTA" claim unless you replicated the baseline.
- [ ] The negative result (self-correction) is included, not buried — it *builds*
      credibility.
- [ ] Figures are self-explanatory (title says the finding; axes labeled; log
      scale noted).
- [ ] Repo link + a one-command repro.
- [ ] A sentence on limitations and what would falsify the claim.

---

## Part 10 — Glossary
- **DLM** — diffusion language model.
- **Masked / absorbing diffusion** — forward process masks tokens; reverse
  process unmasks them. The kind LLaDA uses.
- **NFE** — number of function evaluations = model forward passes; our compute
  axis. One denoising step = one NFE (× blocks for semi-AR).
- **Unmask order** — the policy choosing which masked positions to commit each
  step (confidence/margin/entropy/random/left_to_right).
- **Remask / self-correction** — un-committing a token you later doubt; a move
  autoregression cannot make.
- **Semi-autoregressive (block) decoding** — decode the region in left-to-right
  sub-blocks, each internally diffused.
- **Pareto frontier** — configs not dominated on both cost (NFE) and quality
  (accuracy); the efficient trade-off curve.
- **Ledger** — the append-only JSONL of trials; the run's source of truth.
</content>
