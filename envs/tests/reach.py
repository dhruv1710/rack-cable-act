"""IK reachability of every task's key poses (a sanity check that the layout fits the Panda's workspace)."""

import sys

import numpy as np

import envs

from ..common.arms import solve_ik


def main(names):
    bad = 0
    for n in names:
        env = envs.make(n, render=False)
        env.reset(seed=1)
        tg = env.reach_targets()
        if not tg:
            print(f"--  {n}: no arm targets")
            continue
        for label, ai, pos, R in tg:
            _q, ok = solve_ik(env.m, env.d, env.arms[ai], pos, R, iters=300, restarts=30)
            tag = "ok " if ok else "FAIL"
            bad += not ok
            print(f"{tag} {n:20s} {label:28s} pos={np.round(pos, 3)}")
        env.close()
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:] or envs.list_tasks()) else 0)
