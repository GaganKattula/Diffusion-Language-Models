#!/usr/bin/env bash
# Pull results off the pod BEFORE stopping/deleting it. Run on YOUR LOCAL machine.
#   bash scripts/backup_from_pod.sh <POD_IP> <POD_PORT> [SSH_KEY]
#
# Copies only runs/ (ledgers, figures, run.log) — a few hundred KB. It does NOT
# copy the ~16 GB weights cache (that lives at /workspace/hf, is re-downloadable,
# and is not under the repo).
#
# NOTE: STOPPING a pod keeps the network volume — your data survives. You only
# lose it if you DELETE the volume. The results are also committed to git under
# results/, so the findings are safe regardless. This script is for grabbing the
# original run.log / raw ledgers locally.
set -euo pipefail
HOST="${1:?usage: backup_from_pod.sh <POD_IP> <POD_PORT> [SSH_KEY]}"
PORT="${2:?usage: backup_from_pod.sh <POD_IP> <POD_PORT> [SSH_KEY]}"
KEY="${3:-$HOME/.ssh/id_runpod}"
REMOTE=/workspace/Diffusion-Language-Models
DEST="pod_backup"

mkdir -p "$DEST"
echo "Copying $HOST:$PORT:$REMOTE/runs  ->  $DEST/runs ..."
scp -P "$PORT" -i "$KEY" -r "root@${HOST}:${REMOTE}/runs" "$DEST/" \
  && echo "OK -> $DEST/runs/  (ledgers, figures, run.log)" \
  || echo "scp failed — check IP/port/key from the pod's Connect panel"
