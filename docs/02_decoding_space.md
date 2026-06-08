# 02 — The decoding search space

This is the object the agent searches. It is fully captured by `DecodingParams`
(`decoding/sampler.py`); a config sweep is just a set of values per field.

| field | values explored | hypothesis |
|---|---|---|
| `num_steps` | 1 … 256 | the compute axis; find the knee of the curve |
| `unmask_order` | confidence, margin, entropy, random, left_to_right | confidence/margin beat random; left_to_right is the AR-like baseline |
| `remask` | false / true | self-correction recovers accuracy lost to early commits |
| `remask_frac`, `remask_threshold` | e.g. 0.25 / 0.5 | how aggressively to revise |
| `block_size` | None, 32, 64 | semi-AR; latency/quality + KV-cache structure |
| `temperature` | 0.0 (argmax), >0 | determinism vs. diversity |

## Metrics the Runner records per config
- `accuracy` (task quality) — **maximize**
- `model_calls` (NFE per example) — **minimize** (compute/latency proxy)
- `self_corrections` — mean positions revised via remask (the revision signal)

## The two headline results we're after
1. **Step↔quality Pareto frontier** (`Analyst.pareto_front`): the set of configs
   not dominated by any other. The story: *"the default sampler is wasteful —
   here's a config that matches its quality at a fraction of the steps."*
2. **Self-correction effect** (`Analyst.self_correction_effect`): compare
   `remask=True` vs `False` at matched `(num_steps, order)`. The story: *"the one
   thing diffusion does that GPT structurally can't — take back a mistake."*

## How the unmasking-order policy is computed
Per masked position, from the softmax `probs`:
- `confidence` = top-1 probability
- `margin` = top-1 − top-2 probability
- `entropy` = −Σ p log p (we commit *low* entropy first)
- `random` = uniform
- `left_to_right` = positional priority

Each step commits the top-k highest-priority masked positions, with k set by a
linear schedule so everything is committed by the final step.

## How self-correction is scored
A committed position is re-masked when the probability the model *currently*
assigns to its **committed token** falls below `remask_threshold` — i.e. the
model now disbelieves what it wrote, typically because neighbours filled in and a
better token emerged. A "self-correction" is counted when a position's committed
value actually changes after having been committed at least once.
