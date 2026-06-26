#!/usr/bin/env bash
# One-shot pod setup + plumbing smoke test for the LLaDA backend.
#
# Run from the repo root after cloning, on a GPU pod (RunPod A100/L40S/...):
#     bash scripts/pod_setup.sh
#
# Optional overrides:
#     MODEL=GSAI-ML/LLaDA-8B-Instruct bash scripts/pod_setup.sh
#     HF_HOME=/workspace/hf bash scripts/pod_setup.sh        # cache off small disk
#     SKIP_SMOKE=1 bash scripts/pod_setup.sh                 # install only
#
# torch is assumed preinstalled by the pod image; we do NOT reinstall it.
set -euo pipefail

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Base}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

# Run from the repo root regardless of where this is invoked from.
cd "$(dirname "$0")/.."
echo "==> repo: $(pwd)"
echo "==> branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "==> HF_HOME: $HF_HOME"
mkdir -p "$HF_HOME"

echo
echo "===== [1/5] hardware ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "WARNING: nvidia-smi not found — is this a GPU pod?"
echo "-- disk --"; df -h "$HF_HOME" . 2>/dev/null | sort -u

echo
echo "===== [2/5] torch / CUDA check ====="
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: torch is not installed. Use a RunPod PyTorch template, or:"
  echo "       pip install torch --index-url https://download.pytorch.org/whl/cu121"
  exit 1
fi
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: torch.cuda.is_available() is False — no usable GPU.")
PY

echo
echo "===== [3/5] install package + inference deps (torch left as-is) ====="
pip install -q -e .
pip install -q -U transformers accelerate
echo "installed: $(python -c 'import transformers,accelerate; print("transformers", transformers.__version__, "accelerate", accelerate.__version__)')"

echo
echo "===== [4/5] offline self-check (mock backend, no GPU/network) ====="
python -m dlm_explorer.cli validate

echo
echo "===== [5/5] LLaDA plumbing smoke test ====="
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "SKIP_SMOKE=1 set — stopping before the model download."
  exit 0
fi
echo "Model: $MODEL  (first run downloads ~16 GB into $HF_HOME)"
python scripts/smoke_llada.py --model "$MODEL"

echo
echo "==> DONE. Paste the [1/4]..[4/4] output above (especially mask_token_id,"
echo "    vocab_size, logits shape, decoded generation, and any traceback)."
