#!/usr/bin/env bash
# Run both publishable searches in series, then plot both. Launch once, walk away.
#   bash scripts/run_publishable.sh
# Best run under tmux/nohup so it survives an SSH drop (see below). ~2.25 h total.
#
# Both SEARCHES run first (the expensive part); plotting is best-effort at the end,
# so a plotting hiccup can never waste a completed search. The .jsonl ledgers are
# the real artifacts — figures can always be regenerated from them.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
pip install -q matplotlib 2>/dev/null || true

echo "===== [1/2] confirm: unmask order vs compute (n=100, ~75 min) ====="
dlm-explore run --config configs/llada_confirm.yaml || { echo "confirm run FAILED"; exit 1; }

echo "===== [2/2] self-correction: remask off vs on (n=100, ~60 min) ====="
dlm-explore run --config configs/llada_selfcorrect.yaml || { echo "selfcorrect run FAILED"; exit 1; }

echo "===== plotting ====="
python scripts/plot_results.py --ledger runs/llada_confirm.jsonl --color-by unmask_order \
  --title "LLaDA-8B-Base GSM8K n=100 - unmask order vs compute" || echo "confirm plot skipped"
python scripts/plot_results.py --ledger runs/llada_selfcorrect.jsonl \
  --title "LLaDA-8B-Base GSM8K - self-correction (remask off vs on)" || echo "selfcorrect plot skipped"

echo "===== DONE ====="
echo "ledgers: runs/llada_confirm.jsonl  runs/llada_selfcorrect.jsonl"
echo "figures: runs/llada_confirm.png    runs/llada_selfcorrect.png"
