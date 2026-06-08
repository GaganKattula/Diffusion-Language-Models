# 01 — Background: masked diffusion language models

Just enough theory to understand the code. Four ideas.

## 1. Continuous diffusion (DDPM), one line
A forward process adds noise to data; a model learns the reverse (denoising)
process. Generation = start from noise, denoise iteratively. Text is discrete,
so the continuous formulation (Diffusion-LM, 2022) is awkward and didn't scale.

## 2. Discrete diffusion (D3PM, 2021)
Tokens hop between discrete states via transition matrices. The special case
that won is the **absorbing/masked** process.

## 3. Masked diffusion (MDLM / MD4, 2024) — the one we implement against
The forward process simply **masks** tokens at an increasing rate; "time" is the
mask ratio. The model is trained to predict the original token at masked
positions (a weighted cross-entropy). Mental model:

> **Masked diffusion ≈ BERT-style masking turned into a principled generative
> model — an "any-order autoregressive" model.**

LLaDA (2025) scales this to 8B and rivals LLaMA3-8B. It is open
(`GSAI-ML/LLaDA-8B-Base` / `-Instruct`), which is why it's our target.

## 4. Sampling = the part this project searches
Generation starts from an all-`[MASK]` region and **iteratively unmasks**:

```
repeat for num_steps:
    logits = model(current_sequence)        # predict every masked position
    pick which masked positions to commit   # <- unmask ORDER policy
    commit their tokens (argmax / sample)    # <- temperature
    optionally re-mask low-confidence commits# <- self-correction (remask)
```

Everything in that loop is a **free design choice**, and the defaults shipped
with models like LLaDA are almost certainly not optimal across tasks. The knobs:

- **number of steps** — fewer is faster but lower quality (the Pareto tradeoff).
  Each step is one model forward pass = one "NFE" (number of function
  evaluations), our compute/latency proxy.
- **unmasking order** — commit the most confident positions first? by margin?
  by entropy? left-to-right? randomly? Order changes how context accrues.
- **remasking / self-correction** — a committed token can be un-committed if the
  model later disbelieves it. Autoregressive models can't take back a token;
  diffusion can. This is the structural-advantage story.
- **block size (semi-AR)** — decode the region in left-to-right blocks, each
  internally diffused. Interpolates AR↔diffusion and recovers a KV-cache-like
  structure (block diffusion / LLaDA's semi-AR mode).

## Lineage (for citing the gap)
D3PM (2021) → Diffusion-LM (2022) → SEDD (2024) → MDLM / MD4 (2024) →
LLaDA-8B (2025) → block diffusion / BD3-LM (2025). Most of this work is about
*training* the models; systematic study of *how to decode* them is comparatively
thin — that gap is the project's opening.
