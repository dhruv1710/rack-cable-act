#!/usr/bin/env bash
# Collect ACT demonstrations on a CPU-only Ubuntu 22.04/24.04 box (e.g. AWS c7i.2xlarge).
# NOT TESTED on a cloud machine yet -- software (OSMesa) rendering speed is unmeasured,
# so this runs a 2-episode timing test first and stops unless you pass GO=1.
#
# From your Mac, copy the code up (from the repo root):
#   rsync -av --exclude .venv --exclude out --exclude third_party --exclude papers ./ ubuntu@HOST:~/groot-2/
# On the box:
#   cd ~/groot-2 && bash cloud/collect.sh              # setup + 2-episode timing test
#   GO=1 EPISODES=50 WORKERS=6 bash cloud/collect.sh   # real run (setup is skipped if done)
# Copy the result back:
#   rsync -av ubuntu@HOST:~/groot-2/out/rack_cable/datasets/rack_act_p2 ./out/rack_cable/datasets/
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES=${EPISODES:-50}
WORKERS=${WORKERS:-6}
PORT=${PORT:-2}
OUT=${OUT:-out/rack_cable/datasets/rack_act_p2}

if [[ ! -x .venv/bin/python ]]; then
  sudo apt-get update -y
  sudo apt-get install -y git ffmpeg libosmesa6-dev libgl1-mesa-dev libglfw3 curl build-essential linux-libc-dev python3-dev python3.12-dev
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python mujoco==4.0.1 numpy gymnasium imageio imageio-ffmpeg \
      'lerobot[training,core-scripts]==0.7.0'
fi
[[ -d third_party/mujoco_menagerie ]] || {
  mkdir -p third_party
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
}

export MUJOCO_GL=osmesa          # CPU software rendering (no GPU on this box)

if [[ "${GO:-0}" != "1" ]]; then
  echo "== timing test: 2 episodes, 2 workers (check 'attempts' times below) =="
  time .venv/bin/python tasks/rack_cable/gen_dataset.py --episodes 2 --workers 2 --port "$PORT" \
      --out out/rack_cable/datasets/timing_test --repo-id local/timing_test --overwrite
  echo "Per-episode seconds above x EPISODES / WORKERS = collection time. Rerun with GO=1 to collect."
  exit 0
fi

nohup .venv/bin/python tasks/rack_cable/gen_dataset.py --episodes "$EPISODES" --workers "$WORKERS" --port "$PORT" \
    --out "$OUT" --repo-id "local/$(basename "$OUT")" --overwrite > out_gen.log 2>&1 &
echo "started in background; follow with: tail -f out_gen.log"
