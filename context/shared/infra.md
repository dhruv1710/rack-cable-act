# Cloud and model infrastructure

Last verified: 2026-11-23. Code: `models/pi/modal_baseline.py`, `envs/pi_run.py`, `models/act/`.

## Modal (GPU)
- Runs used one L4. Image builds needed: uv-managed Python 3.12, `--no-dev`, exclude gym-aloha / mujoco / dm-control and
  similar, run the venv's python directly (not `uv run`), a long `UV_HTTP_TIMEOUT`, and a fix to the script's
  `parents[2]` path inside the container.
- The simulator can run on the GPU host too; the dev laptop runs these envs at roughly 12-71 steps/s with rendering.

## openpi (pi0.5)
- Checkpoint `pi05_droid`. Output is joint velocity, integrated to absolute joint targets (0.2 rad per step). Policy runs
  at 15 Hz, a new chunk every 8 actions, about 0.44 s per call.
- Zero-shot results: see `projects/datacentre_envs/pi05_baseline.md`.

## LeRobot / ACT
- ACT trained on scripted demonstrations. The rack-cable run: 50 demos, 54,879 frames at 50 Hz, 17,500 steps, batch 8, one
  L4, about $0.995 total (figures from the published page; (unverified) in this session).
- Evaluation fix that mattered: temporal ensembling at inference (`--temporal-ensemble 0.01`), no retraining.
