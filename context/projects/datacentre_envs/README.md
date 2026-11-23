# Data-centre manipulation benchmark (11 tasks)

Last verified: 2026-11-23. Code: `envs/` (tasks in `envs/tasks/`, scripted policies in `envs/experts.py`), docs:
`docs/envs.md`, page: `public/envs/index.html` (live at `/envs/`).

11 MuJoCo tasks, one Franka Panda model (two for handoff, a mobile base for the patrol), one Gymnasium interface.
Reference solutions are scripted (privileged state + IK). A zero-shot pi0.5 baseline scores 0/27 (`pi05_baseline.md`).

## Status (scripted policy, 2026-11-23)
| Task | Result | Note |
|---|---|---|
| cable_insertion | 10/10 | |
| cable_routing | 8/8 | only on the simplified layout (see tasks.md) |
| fuse_holder | 12/12 | |
| plate_racking | 10/10 | |
| object_handoff | 10/10 | |
| thermal_patrol | 6/6 | thermal image only |
| pouring_beads | 21/24 | |
| bottle_binning | 10/12 | |
| parts_sorting | 16/24 | |
| relative_placement | 16/24 | |
| drive_replacement | 3/9 | seeds 4, 5, 6 |

Files: `tasks.md` (per-task details the code does not say), `knowledge.md` (failed approaches, open problems),
`pi05_baseline.md`. Tests: `.venv/bin/python -m envs.smoke`, `envs.tests.mechanisms`, `envs.tests.grasp_check`,
`envs.tests.reach`; lint `uvx ruff check envs`; record `envs.record_scripted`. Last full regression: smoke 11/11,
mechanisms 17 pass, grasp 16/16, lint clean.
