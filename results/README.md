# Results (committed, permanent)

Final result artifacts from the LLaDA-8B-Base / GSM8K decoding search (n=100,
4-shot, single-block, temperature 0). Committed to git so they survive the pod /
volume being deleted. See `docs/PROJECT_BREAKDOWN.md` Part 3 for interpretation.

| file | what |
|---|---|
| `llada_confirm.jsonl` | order-vs-compute run: confidence & random × {16,32,64,128} steps |
| `llada_confirm.png` | the headline figure — confidence vs random diverge as steps grow |
| `llada_selfcorrect.jsonl` | matched remask off/on at 64 & 128 steps (confidence) |
| `llada_selfcorrect.png` | self-correction figure (naive remask: +0.03 @64, −0.07 @128) |

Headline: at 128 steps, confidence **0.52** vs random **0.20** (2.6×) at identical
compute; the curves are tied below ~24 steps then diverge — extra denoising
compute only pays off under a good unmask order.

Regenerate figures from the ledgers:
```bash
python scripts/plot_results.py --ledger results/llada_confirm.jsonl --color-by unmask_order
python scripts/plot_results.py --ledger results/llada_selfcorrect.jsonl
```
