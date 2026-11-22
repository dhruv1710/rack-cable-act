"""Smoke test for every task: build, reset, step (stability), render a contact sheet, oracle success.

.venv/bin/python -m envs.smoke [task ...] [--out out/envs]
"""

import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np

import envs


def run(name, out):
    t0 = time.time()
    env = envs.make(name)
    build_s = time.time() - t0
    obs, info = env.reset(seed=1)
    assert not info.get("success"), f"{name}: already successful at reset"
    zero = env.noop_action()
    t0 = time.time()
    for _ in range(100):
        obs, _r, _term, _trunc, info = env.step(zero)
    sps = 100 / (time.time() - t0)
    assert np.isfinite(env.d.qpos).all() and np.isfinite(env.d.qvel).all(), f"{name}: NaN state"
    assert not info["is_success"], f"{name}: succeeded with no-op policy"
    imgs = [obs[k] for k in obs if k.startswith("observation.images")]
    h = max(i.shape[0] for i in imgs)
    sheet = np.concatenate([np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs], 1)
    Path(out).mkdir(parents=True, exist_ok=True)
    iio.imwrite(Path(out) / f"{name}.png", sheet)
    # oracle: solved state must register as success
    env.reset(seed=2)
    env.oracle_solve()
    for _ in range(60):
        env.physics()
    ev = env.evaluate()
    status = "OK " if ev["success"] else "FAIL(oracle)"
    print(
        f"{status} {name:20s} build {build_s:4.1f}s  {sps:5.0f} step/s  nbody={env.m.nbody} ngeom={env.m.ngeom} "
        f"progress@oracle={ev.get('progress', 0):.2f}  instr='{env.instruction}'"
    )
    env.close()
    return ev["success"]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = "out/envs"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        args = [a for a in args if a != out]
    names = args or envs.list_tasks()
    ok = all(run(n, out) for n in names)
    sys.exit(0 if ok else 1)
