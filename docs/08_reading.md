# Prerequisite Reading

Two tracks — **research** (the ideas) and **engineering** (the build). Consult
when a concept is unfamiliar. Each entry says what to extract and its difficulty
(**[B]** beginner / **[I]** intermediate / **[A]** advanced). Do not treat any
arXiv id as gospel — if one looks off, search the title.

---

# Research track — diffusion LMs & decoding

Assumes strong general ML/DL but no prior diffusion. Read §1 first; §2–3 are the
technical core; §4–7 are what you need to reason about a concrete diffusion-LM
project; §8 is optional automation context.

## 1. Diffusion fundamentals — DDPM, forward/reverse process, ELBO, score/SDE view
Everything downstream specializes this: a fixed noising process, a learned
denoising process, and an ELBO that turns "learn to denoise" into a loss.
- **Lilian Weng, "What are Diffusion Models?" (blog, 2021→2024)** — **[B]** — *Best start.* Consistent notation, forward process, reparameterized loss, guidance.
- **Calvin Luo, "Understanding Diffusion Models: A Unified Perspective," 2022, arXiv:2208.11970** — **[I]** — ELBO from scratch; DDPM as a hierarchical VAE; the three equivalent targets (data/noise/score).
- **Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," 2020, arXiv:2006.11239** — **[I]** — The original; the simplified noise-prediction loss + train/sample pseudocode.
- **Song et al., "Score-Based Generative Modeling through SDEs," 2021, arXiv:2011.13456** — **[A]** — Unifies DDPM & score matching; read just enough to recognize "score"/"reverse-time" when SEDD reuses it.

## 2. Discrete diffusion — D3PM, transition matrices, the absorbing state
Text is categorical; diffusion is defined by transition *matrices* over the
vocab. D3PM introduces the **absorbing-state** (→`[MASK]`) case everything builds on.
- **Austin, Johnson, Ho, Tarlow, van den Berg, "Structured Denoising Diffusion Models in Discrete State-Spaces" (D3PM), 2021, arXiv:2107.03006** — **[A]** — *Best start (no easier option).* The transition-matrix formulation; uniform vs absorbing vs Gaussian variants; how absorbing degenerates toward a BERT-like loss.
- **Hunter Heidenreich, "D3PM" blog note (search title)** — **[B]** — Code-first, lighter-notation companion.

## 3. Masked / absorbing diffusion for text — SEDD, MDLM, MD4; the any-order-AR intuition
The absorbing-state ELBO collapses to a **weighted cross-entropy at masked
positions** — training ≈ BERT MLM with a time-dependent mask rate. Sampling ≈
pick a random generation *order* then fill it (an "any-order autoregressive model").
- **Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (MDLM), 2024, arXiv:2406.07524** — **[I]** — *Best start.* SUBS parameterization; the ELBO reducing to a weighted average of masked-LM losses.
- **Shi et al., "Simplified and Generalized Masked Diffusion for Discrete Data" (MD4), 2024, arXiv:2406.04329** — **[I]** — Independent concurrent derivation; cross-check on MDLM.
- **Lou, Meng, Ermon, "Discrete Diffusion by Estimating Ratios of the Data Distribution" (SEDD), 2024 (ICML best paper), arXiv:2310.16834** — **[A]** — The score-based route; "concrete score"; matches quality with ~32× fewer steps.
- **Hoogeboom et al., "Autoregressive Diffusion Models" (ARDM), 2021, arXiv:2110.02037** — **[A]** — The formal "order-agnostic AR" masked diffusion generalizes; read only for the precise theorem behind the slogan.

## 4. Scaling diffusion LMs — LLaDA, block diffusion, commercial systems
Where diffusion is validated as an AR alternative at real scale, plus the hybrid
that keeps AR's KV-cache tricks.
- **Nie et al., "Large Language Diffusion Models" (LLaDA), 2025, arXiv:2502.09992** — **[I]** — *Best start.* The 8B model we use; scaling the MDLM objective; fixed-length generation without EOS.
- **Arriola et al., "Block Diffusion" (BD3-LM), 2025 (ICLR oral), arXiv:2503.09573** — **[A]** — Diffuse within blocks, generate blocks AR; quality/efficiency vs block size; recovers KV-cache + unbounded length.
- **Inception Labs, "Mercury," 2025, arXiv:2506.17298** — **[I]** — First commercial-scale diffusion LLM; throughput numbers; existence proof.
- **Google DeepMind, "Gemini Diffusion" (blog, 2025; search title)** — **[B]** — High-level framing only.

## 5. Decoding / sampling for masked diffusion — steps, order, remasking, blocks, guidance
Where projects get stuck: the training objective doesn't specify how many steps,
which positions to unmask first, whether to correct decided tokens, or block
structure. **These are exactly this project's knobs.**
- **Chang et al., "MaskGIT: Masked Generative Image Transformer," 2022, arXiv:2202.04200** — **[B]** — *Best start (image, but foundational).* Confidence-based iterative decoding (keep most-confident predictions, remask rest) — the ancestor of our confidence/margin/entropy unmask orders.
- **Wang, Schiff, Sahoo, Kuleshov, "Remasking Discrete Diffusion Models with Inference-Time Scaling" (ReMDM), 2025, arXiv:2503.00307** — **[A]** — Directly on **self-correction**: a principled remasking backward process trading steps for quality. **Read this before claiming anything about our remasking result.**
- **Wu et al., "Fast-dLLM: Training-free Acceleration of Diffusion LLM…," 2025, arXiv:2505.22618** — **[I]** — Confidence-threshold parallel decoding + approximate block KV-cache; NFE/throughput levers.
- **Ho, Salimans, "Classifier-Free Diffusion Guidance," 2022, arXiv:2207.12598** — **[I]** — The guidance-scale trick text-diffusion CFG transplants (our stubbed `cfg_scale`).

## 6. Evaluation — GSM8K, few-shot/CoT, exact-match, fixed-length generation
Mostly inherited AR-LLM conventions, plus the diffusion wrinkle: no EOS → fixed
generation window.
- **Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K), 2021, arXiv:2110.14168** — **[B]** — *Best start.* The dataset; exact-match on the final numeric string (why our extraction is regex-on-last-number).
- **Wei et al., "Chain-of-Thought Prompting…," 2022, arXiv:2201.11903** — **[B]** — The few-shot CoT prompt format.
- **Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020, arXiv:2005.14165** — **[I]** — Few-shot in-context learning framing.
- *Fixed-length/no-EOS:* re-read LLaDA §4 and BD3-LM (§4 above) — both address it directly.

## 7. The compute–quality / Pareto framing
Diffusion exposes an explicit steps↔quality knob; framing results as a Pareto
frontier is the standard argument. **This is our core framing.**
- **ReMDM (arXiv:2503.00307)** — **[A]** — *Best start.* Explicitly plots "pushing the Pareto frontier" as steps increase, incl. how remasking reshapes it.
- **SEDD (arXiv:2310.16834)** — **[A]** — Matching AR quality with far fewer NFEs — a concrete anchor.
- **Fast-dLLM (arXiv:2505.22618)** — **[I]** — Throughput-vs-accuracy curves; a template for your own Pareto plot.
- **Hoffmann et al., "Chinchilla," 2022, arXiv:2203.15556** — **[I]** — The canonical "compute-optimal frontier" mental model (not diffusion-specific).

## 8. (lighter) Automated / agentic ML research
Only if you build the propose→run→analyze loop.
- **Snoek, Larochelle, Adams, "Practical Bayesian Optimization of ML Algorithms," 2012, arXiv:1206.2944** — **[I]** — *Best start for the "search" half.* GP over hyperparameters to pick the next experiment — the ancestor of every propose→evaluate loop.
- **Lu et al., "The AI Scientist," 2024, arXiv:2408.06292** — **[I]** — End-to-end idea→code→run→write pipeline; read for the architecture, not for evaluation quality (contested).
- **"Agent Laboratory: Using LLM Agents as Research Assistants," 2025, arXiv:2501.04227** — **[I]** — Human-in-the-loop variant; how they scope agent autonomy per phase.

---

# Engineering track — building the harness

## 1. Python project structure & packaging — **[B]**
`src/` layout + `pyproject.toml` + editable install make `dlm-explore` a real CLI
while iterating; extras keep it laptop-installable; a name→class registry powers
`backend: mock|llada` without an if/elif ladder.
- **Python Packaging User Guide — "src layout vs flat layout"** (packaging.python.org) — why `src/` prevents importing the uninstalled copy; how `pip install -e .` resolves.
- **setuptools docs / packaging.python.org "Writing your pyproject.toml"** — `[project.optional-dependencies]` extras and `[project.scripts]` entry points.
- **pip docs — "editable installs"** — why source edits show up without reinstalling (debugging "why isn't my change showing up").
- Registry pattern: search **"Python plugin registry decorator `@register`"** — the `_REGISTRY: dict[str,Callable]` + decorator idiom.

## 2. NumPy vectorization for model outputs — **[B→I]**
A step yields `(B, L, ~126k)` logits; a naive per-step CPU softmax+full-sort is
the easiest way to turn a GPU job CPU-bound (a real bug we hit).
- **NumPy docs — `numpy.argsort` vs `numpy.argpartition`** — `argpartition` is O(n) for top-k; you rarely need a full sort over 126k logits.
- **NumPy docs — "basics.broadcasting"** — vectorized softmax `exp(x - x.max(axis=-1,keepdims=True))` avoids Python loops.
- **VanderPlas, *Python Data Science Handbook*, "Computation on NumPy Arrays: Universal Functions"** (jakevdp.github.io) — *why* vectorized ufuncs beat loops; the mental model behind the CPU-softmax slowdown.

## 3. PyTorch + HuggingFace transformers inference — **[I]**
Load 8B correctly (dtype, `trust_remote_code`, eval) and keep every reduction
on-device; the difference between seconds/step and a stall on a `(B,L,V)` host transfer.
- **transformers docs — "Load with an AutoClass" / `from_pretrained` reference** — `torch_dtype=bfloat16`, `trust_remote_code=True` (custom `modeling_*.py`, exactly LLaDA), `device_map`.
- **PyTorch docs — `torch.no_grad` / `inference_mode`** — disable autograd for inference; cuts memory.
- **PyTorch docs — `torch.topk`, `torch.softmax`, `torch.gather`** — reduce on the CUDA tensor; only `.cpu()` the small result, never the raw `(B,L,V)`.
- **transformers — "Efficient inference on a single GPU"** — dtype choice; avoid host transfers.

## 4. Diffusion-LM inference specifics — **[A]**
The part that breaks AR-serving intuition: masked diffusion re-runs a full
bidirectional forward every step (no KV-cache), so cost ≈ `num_steps × forward`.
- **LLaDA paper + repo (`ML-GSAI/LLaDA`)** — the mask-predict-remask sampling loop; why no causal cache.
- **Sasha Rush, "Diffusion Language Models" notes (GitHub `srush`) / MD4** — mask-predict step structure; bidirectional attention forecloses caching.
- **"Transformer Inference Arithmetic" (kipp.ly)** — the roofline framing; at batch 1 a full re-forward is memory-bandwidth-bound, so batching is the lever to become compute-bound.
- **transformers docs — "Padding and truncation" + Glossary `attention_mask`** — right-padding keeps real-token positions fixed; `attention_mask` ignores pad — but **a custom `forward()` can silently ignore it** (our batching bug), only caught by bit-for-bit tests.

## 5. Experiment infrastructure — **[I]**
Append-only, hash-keyed ledgers + declarative configs let you re-launch a sweep
and pay only for un-run trials; fixed seeds make "did my change help?" meaningful.
- **JSON Lines (jsonlines.org)** — one object per line, streaming-appendable, crash-safe (partial last line discarded).
- **Python `hashlib` docs** — hash a canonical `json.dumps(cfg, sort_keys=True)` → stable trial id for idempotent "skip if already run".
- **PyYAML docs — `safe_load`** (never plain `load`) for configs.
- **PyTorch — "Reproducibility"** — the seed checklist + the caveat that CUDA kernels aren't fully deterministic (set expectations before chasing "non-repro" bugs).

## 6. Testing scientific/ML code — **[I]**
The two silent-failure modes: batched ≠ serial (padding/mask bugs), and burning
GPU-hours on a bug a fake backend catches for free.
- **pytest docs — "Fixtures" and "Parametrizing tests"** — swap in a mock backend; parametrize over configs.
- **PyTorch — `torch.testing.assert_close` / `allclose`** — exact vs tolerance equality; decide whether batched==serial must be bit-exact or `atol/rtol`-bounded.
- **Martin Fowler, "Test Double"** — fake vs mock vs stub; why `mock.py` is a first-class "no-GPU" mode, not just test scaffolding.

## 7. Cloud GPU workflow — **[B→I]**
8B weights are tens of GB — on ephemeral disk they re-download every restart; long
sweeps must survive SSH drops.
- **RunPod docs — "Storage / Network Volumes"** — container disk (wiped) vs volume disk (per-pod) vs network volume (persists, but datacenter-pinned — new pod must be same region).
- **HF docs — env vars, `HF_HOME`** — point it at the persistent volume *before* the first `from_pretrained` so weights don't re-download.
- **tmux cheat-sheet + `man nohup`** — `tmux new -s run` / detach `Ctrl-b d` / `tmux attach`; or `nohup job > log 2>&1 &` — keep sweeps alive after disconnect.
- **RunPod docs — "Connect with SSH"** — add your public key so `ssh` skips passwords.

## 8. matplotlib for figures — **[B]**
Must run headless on a GPU box; the standard shapes are one-liners.
- **matplotlib docs — "what is a backend" / `matplotlib.use('Agg')`** — Agg (set before importing pyplot) to save PNGs with no display.
- **matplotlib — "pyplot tutorial" + `Axes.set_xscale`** — `plot`/`scatter` + `ax.set_xscale("log")` for step-count axes.
- **matplotlib — "Creating multiple subplots"** — `fig, axes = plt.subplots(...)` small-multiples + `fig.savefig(..., dpi=150, bbox_inches="tight")`.
