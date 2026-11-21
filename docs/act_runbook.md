# ACT on the rack-cable task: runbook

Task: Franka picks a cable plug off a table and seats it in port 2 of a patch panel (MuJoCo).
Cable start pose is randomized each episode (+-4 cm sideways, up to 1.5 cm back, +-14 deg yaw).

All commands run from the repo root with the project venv (`.venv`, Python 3.12, MuJoCo 4.0.1, LeRobot 0.7.0).

## 1. Generate demonstrations
    .venv/bin/python tasks/rack_cable/gen_dataset.py --episodes 100 --workers 8 --port 2 \
        --out out/rack_cable/datasets/rack_act_p2 --repo-id local/rack_act_p2
- Runs the scripted expert (`scripted_insert.py`) on seeds 1000+, keeps only successful episodes.
- 50 Hz, ~1600 frames/episode. Observation: `observation.images.overview` + `.wrist` (192x256),
  `observation.state` (7 joints + gripper). Action: 7 absolute joint targets + gripper (1=open).
- Recorded actions are exactly what `env.step()` executes: replaying them open-loop on the same seed
  reproduces the insertion (verified), so there is no train/eval action mismatch.
- Videos are H.264 with `streaming_encoding` (the AV1 default was ~10x slower).

## 2. Train
    models/act/train_act.sh                 # 20k steps, batch 8, Apple GPU (mps)
    DEVICE=cuda STEPS=100000 BATCH=16 models/act/train_act.sh   # on a cloud GPU
    RESUME=1 models/act/train_act.sh        # continue an interrupted run
Checkpoints: `out/rack_cable/act/train/act_p2/checkpoints/<step>/pretrained_model` (and `last`).

## 3. Evaluate in simulation
    .venv/bin/python models/act/eval_act.py \
        --checkpoint out/rack_cable/act/train/act_p2/checkpoints/last/pretrained_model \
        --episodes 20 --video-dir out/rack_cable/act/evals/act
Seeds start at 50000 (disjoint from training). Success = port latch engaged within 45 s.

## Things to know
- The port has a **latch** (weld constraint) that engages automatically once the tip is within 2 mm of the
  axis and 2.5 mm of fully seated. Without it the hanging cable pulls the plug back out. Success = latched.
- Socket clearance is 1.5 mm/side with a 30 deg lead-in funnel (real RJ45 is much tighter).
- ACT has no language input: this policy only does port 2. Other ports need their own dataset/policy.
- The expert uses privileged state (plug pose); ACT only sees the two cameras and joint state.
- Expert reference: 24/24 successes on random seeds, ~32 s per episode.

## Results and the recipe that works (2026-11-21)
Trained on Modal L4 (`models/act/modal_train.py`, 50 fixed episodes, batch 8, ~17.5k steps, ~$0.85), evaluated locally on unseen seeds 50000-50019.

| eval setting (step-17,500 checkpoint) | success |
|---|---|
| default (`n_action_steps=50`) | 4/20 |
| `n_action_steps` 10 / 25 / 100 | 0/8 each |
| **temporal ensembling** (`--temporal-ensemble 0.01`) | **20/20** (95% lower bound ~83%) |

    .venv/bin/python models/act/eval_act.py --checkpoint out/rack_cable/act/checkpoints/best/pretrained_model \
        --episodes 20 --temporal-ensemble 0.01
- Sim only; same randomization as training. Not evidence for real hardware.
- The expert must not idle: a ~10 s dead stretch at the hover pose (fixed timeout) made the policy stall there. Fixed in `scripted_insert.py` (loose tol + short timeouts on the rise/yaw phase).
- `gen_dataset.py` needs `encoder_queue_maxsize=2000` or the video encoder silently drops frames under load.
- CPU-only cloud generation is ~5x slower than the Mac (software rendering); generate locally.
- Checkpoints trained on CUDA need the device override in `eval_act.py` (already added).
