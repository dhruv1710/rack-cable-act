"""Run the scripted policies with rendering on, record an mp4 per episode, report success rates.

    .venv/bin/python -m envs.record_scripted pouring_beads 5
    .venv/bin/python -m envs.record_scripted all 5

The scripted policies read privileged simulator state (object poses) and use inverse kinematics. They are reference
solutions that show each task is solvable, not learned policies. Videos go to out/scripted/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

import envs

from .experts import (
    bottle_binning,
    drive_replacement,
    fuse_holder,
    object_handoff,
    parts_sorting,
    plate_racking,
    pouring_beads,
    thermal_patrol,
)
from .tests.pick_place import solve_relative_placement

OUT = Path("out/scripted")
IMG_HW = (288, 384)
STRIDE = 2  # keep every 2nd 50 Hz frame -> 25 fps recording, played back at 2x

POLICIES = {
    "relative_placement": lambda env: solve_relative_placement(env)[1],
    "object_handoff": object_handoff,
    "pouring_beads": pouring_beads,
    "parts_sorting": parts_sorting,
    "bottle_binning": bottle_binning,
    "thermal_patrol": thermal_patrol,
    "plate_racking": lambda env: plate_racking(env)[0],
    "fuse_holder": fuse_holder,
    "drive_replacement": drive_replacement,
}


def frame_of(task: str, obs: dict) -> np.ndarray:
    """Video frame: the overview camera, plus the thermal image beside it for the patrol task."""
    img = obs["observation.images.overview"]
    if task == "thermal_patrol":
        th = Image.fromarray(obs["observation.images.thermal"]).resize((img.shape[0] * 4 // 3, img.shape[0]))
        img = np.concatenate([img, np.asarray(th)], axis=1)
    return img


def run(task: str, n: int, record_all: bool = False, seeds=None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    wins, rows = 0, []
    for seed in seeds if seeds is not None else range(n):
        env = envs.make(task, img_hw=IMG_HW)
        env.reset(seed=seed)
        frames, step = [], env.step
        count = [0]

        def wrapped(a, _step=step, _f=frames, _c=count):
            out = _step(a)
            _c[0] += 1
            if _c[0] % STRIDE == 0:
                _f.append(frame_of(task, out[0]))
            return out

        env.step = wrapped
        t0 = time.time()
        ev = POLICIES[task](env)
        ok = bool(ev["success"])
        wins += ok
        sim_s = count[0] / 50
        rows.append(
            {"task": task, "seed": seed, "success": ok, "sim_seconds": round(sim_s, 1), "instruction": env.instruction}
        )
        tag = "ok" if ok else "fail"
        print(f"{task} seed {seed}: {tag}  sim {sim_s:.0f}s  wall {time.time() - t0:.0f}s", flush=True)
        if ok or record_all:
            imageio.mimsave(OUT / f"{task}_seed{seed}_{tag}.mp4", frames, fps=25 * 2, macro_block_size=1)
        env.close()
    print(f"{task}: {wins}/{n} scripted successes")
    with open(OUT / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"task": task, "wins": wins, "n": n}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for t in POLICIES if which == "all" else [which]:
        run(t, n)
