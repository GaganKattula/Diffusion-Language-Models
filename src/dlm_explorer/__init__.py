"""dlm_explorer — an agentic auto-research harness for diffusion-LM decoding.

This package searches the *decoding* design space of a masked diffusion language
model (number of denoising steps, unmasking order, remasking / self-correction,
semi-autoregressive block size, temperature) and surfaces the step <-> quality
Pareto frontier plus self-correction effects.

The whole pipeline is runnable end-to-end against a deterministic *mock* backend
(no GPU, no network, no weights). The real LLaDA backend is wired but gated; see
`dlm_explorer.backends.llada`.

Start here for orientation: docs/00_recollection.md
"""

__version__ = "0.1.0"
