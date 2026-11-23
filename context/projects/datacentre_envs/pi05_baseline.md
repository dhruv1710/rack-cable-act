# pi0.5 zero-shot baseline

Last verified: 2026-11-23. Code: `models/pi/modal_baseline.py`, `envs/pi_run.py`. Setup details: `shared/infra.md`.

- Checkpoint `pi05_droid`, no fine-tuning, one L4. 3 episodes per task (seeds 0-2), full time limit, single-arm tasks only.
- Result: 0/27. Mean progress was mostly the starting layout: a motionless arm scores 0.52 on relative placement.
  Progress: cable routing 0.25, fuse holder 0.25, relative placement 0.57, others 0.00.
- Prompts were the environments' own instructions, not simplified. The velocity-to-target scale (0.2 rad/step) is untested.
- Six environments changed after the baseline ran, so it used earlier versions: cable routing (straight 0.40 m cable, clips
  on one line), parts sorting (rolling friction), bottle binning (bottles overlapped), plate racking (source and rack on the
  same side), fuse holder (lids and holder spacing), handoff (20 cm bar).
- The page shows seed-0 baseline videos at 3x speed.
