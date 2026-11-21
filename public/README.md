# Franka plug-into-patch-panel: ACT policy (MuJoCo sim)

A Franka Panda picks up a cable plug and inserts it into a server-rack patch-panel port.
Policy: ACT (LeRobot 0.7.0), trained on 50 scripted-expert demos (54,879 frames, 50 Hz,
overview + wrist cameras 192x256, 7 joint targets + gripper).

## Results (unseen seeds, simulation only)
| Setup | Success |
|---|---|
| Default ACT chunk execution (n_action_steps=50) | 4/20 |
| Re-plan every 10/25/100 steps | 0/8 each |
| **Temporal ensembling (coeff 0.01)** | **20/20** (52/52 across all runs, ~93% lower 95% bound) |
| Temporal ensembling, cable start range 1.25x wider | 10/10 |

Checkpoint: 17,500 steps, batch 8, trained on a Modal L4 for under $1.
Sim-to-real is untested.

## Contents
- `rack_cable/videos/`: example rollouts (overview | wrist) and a default-chunking failure
- `rack_cable/model/`: processor configs. The 198 MB `model.safetensors` is not in the repo (over GitHub's 100 MB limit); ask the author for it.
- `rack_cable/front.png`, `closeup.png`: scene views

## Run it
```
.venv/bin/python models/act/eval_act.py --checkpoint public/rack_cable/model --temporal-ensemble 0.01 --episodes 10
```
