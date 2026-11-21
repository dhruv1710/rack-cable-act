#!/usr/bin/env bash
# Train ACT on the generated rack-cable dataset.
#
#   models/act/train_act.sh                       # defaults below (Apple GPU)
#   STEPS=50000 BATCH=16 models/act/train_act.sh
#   DEVICE=cuda models/act/train_act.sh           # on a cloud GPU box
#   RESUME=1 models/act/train_act.sh              # continue an interrupted run
#
# Notes
#  * Data is 50 Hz. chunk_size=200 -> the policy predicts 4 s of actions per query.
#  * n_action_steps=50 -> re-plans every 1 s (closed-loop) instead of running a whole chunk blind;
#    insertion needs feedback. Try 100/200 or temporal ensembling later.
#  * Checkpoints every SAVE_FREQ steps under $OUT/checkpoints/<step>/pretrained_model
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-out/rack_cable/datasets/rack_act_p2}
REPO=${REPO:-local/rack_act_p2}
OUT=${OUT:-out/rack_cable/act/train/act_p2}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-8}
DEVICE=${DEVICE:-mps}
WORKERS=${WORKERS:-4}
SAVE_FREQ=${SAVE_FREQ:-2500}
CHUNK=${CHUNK:-200}
NACT=${NACT:-50}

if [[ "${RESUME:-0}" == "1" ]]; then
  exec .venv/bin/lerobot-train --resume=true \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
fi

if [[ -e "$OUT" ]]; then
  echo "error: $OUT already exists. Use RESUME=1 to continue it, or OUT=<new dir> / rm -rf it." >&2
  exit 1
fi

exec .venv/bin/lerobot-train \
  --dataset.repo_id="$REPO" \
  --dataset.root="$DATASET" \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --policy.chunk_size="$CHUNK" \
  --policy.n_action_steps="$NACT" \
  --output_dir="$OUT" \
  --job_name="$(basename "$OUT")" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --num_workers="$WORKERS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=100
