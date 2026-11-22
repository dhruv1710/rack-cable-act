"""Scripted pick-and-place demonstrations (IK + joint interpolation) used as physical sanity checks.

.venv/bin/python -m envs.tests.pick_place relative_placement [n_episodes]
"""

import sys

import numpy as np

import envs

from ..common.arms import down_rot
from ..tasks.relative_placement import CATALOG, REL
from .scripted import Driver


def yaw_of(env, body):
    R = env.d.xmat[body].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def solve_relative_placement(env):
    d = env.d
    drv = Driver(env)
    tgt, ref, rel = env.target, env.ref, env.rel
    rt, ht, _ = CATALOG[tgt]
    rr, hr, _ = CATALOG[ref]
    p = d.xpos[env.bid[tgt]].copy()
    yaw = yaw_of(env, env.bid[tgt])
    # fingers should close across the object's local y (box / mug handle on local +x)
    gyaw = yaw + np.pi / 2 if tgt in ("cereal box", "mug", "toy block") else 0.0
    R = down_rot(gyaw)
    zc = max(p[2], 0.035)
    if tgt == "bowl":  # too wide to pinch across: grip the near rim wall
        gyaw, R, zc = np.pi / 2, down_rot(np.pi / 2), 0.040
        p = p + np.array([-0.066, 0.0, 0.0])
    hover = 0.20
    ok = []
    ok.append(drv.move(np.array([p[0], p[1], hover]), R, 1.5, grip=1.0)[0])
    ok.append(drv.move(np.array([p[0], p[1], zc]), R, 1.2)[0])
    drv.grasp(0.7)
    ok.append(drv.move(np.array([p[0], p[1], hover]), R, 1.0)[0])
    pr = d.xpos[env.bid[ref]].copy()
    if rel == "on top of":
        goal = np.array([pr[0], pr[1], pr[2] + hr / 2 + ht / 2 + 0.006])
    else:
        ax = np.array(REL[rel])
        goal = np.array([pr[0] + ax[0] * (rt + rr + 0.025), pr[1] + ax[1] * (rt + rr + 0.025), ht / 2 + 0.006])
    if tgt == "bowl":
        goal = goal + np.array([-0.066, 0.0, 0.0])
        goal[2] = 0.044
    ok.append(drv.move(np.array([goal[0], goal[1], hover + max(0, goal[2] - 0.05)]), R, 1.6)[0])
    ok.append(drv.move(goal, R, 1.2)[0])
    drv.release(0.6)
    drv.move(np.array([goal[0], goal[1], goal[2] + 0.18]), R, 1.0)
    drv.hold(0.8)
    return all(ok), env.evaluate()


def main(task, n=5):
    fn = {"relative_placement": solve_relative_placement}[task]
    wins = 0
    for seed in range(n):
        env = envs.make(task, render=False)
        env.reset(seed=seed)
        info0 = env.instruction
        ik_ok, ev = fn(env)
        wins += ev["success"]
        print(
            f"seed {seed}: {'SUCCESS' if ev['success'] else 'fail   '} ik_ok={ik_ok}  '{info0}'  "
            f"progress={ev['progress']:.2f}"
        )
        env.close()
    print(f"{task}: {wins}/{n} scripted successes")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
