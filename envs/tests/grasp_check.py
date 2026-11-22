"""Scripted grasp-and-lift for the key object of every task: confirms the object is physically graspable by the
Panda gripper (size, friction, approach clearance). Success = object rises >= 5 cm with the gripper and stays up.

    .venv/bin/python -m envs.tests.grasp_check [task ...]
"""

import sys

import numpy as np

import envs

from ..common.arms import down_rot, forward_rot
from .scripted import Driver


def lift_test(env, body, pre, grasp, R, lift=0.12, arm=0, approach_s=1.6, label="", stage=None):
    drv = Driver(env, arm)
    d = env.d
    z0 = d.xpos[body][2]
    stage = np.asarray(stage) if stage is not None else None
    if stage is not None:  # safe staging pose, clear of the objects, before the straight-line approach
        drv.move(stage, R, 1.8, grip=1.0)
        ok1, _ = drv.move_linear(pre, R, 1.2)
    else:
        ok1, _ = drv.move(pre, R, approach_s, grip=1.0)
    ok2, _ = drv.move_linear(grasp, R, 1.6)
    drv.grasp(0.8)
    drv.move_linear(grasp + np.array([0, 0, lift]), R, 1.4)
    drv.hold(0.6)
    rise = d.xpos[body][2] - z0
    return rise >= 0.05, rise, ok1 and ok2


def case_bottle(env, standing):
    d = env.d
    for b, _n in zip(env.bid, env.bottle_names):
        up = abs(d.xmat[b].reshape(3, 3)[2, 2])
        if (up > 0.9) == standing:
            p = d.xpos[b].copy()
            if standing:
                return (
                    b,
                    np.array([p[0], p[1], 0.32]),
                    np.array([p[0], p[1], 0.168]),
                    down_rot(0.0),
                )  # pinch the neck: palm must clear the cap
            ax = d.xmat[b].reshape(3, 3)[:, 2]
            yaw = float(np.arctan2(ax[1], ax[0]))
            return b, p + np.array([0, 0, 0.12]), np.array([p[0], p[1], p[2]]), down_rot(yaw)
    return None


def run(name):
    out = []
    env = envs.make(name, render=False)
    env.reset(seed=0)
    d, _m = env.d, env.m
    if name == "bottle_binning":
        for standing in (True, False):
            c = case_bottle(env, standing)
            if c:
                b, pre, g, R = c
                out.append((f"bottle ({'standing' if standing else 'lying'})", lift_test(env, b, pre, g, R)))
                env.reset(seed=0)
    elif name == "pouring_beads":
        b = env.cup
        p = d.xpos[b].copy()
        R = down_rot(np.pi / 2)
        out.append(
            ("cup (full of beads)", lift_test(env, b, p + np.array([0, 0, 0.12]), np.array([p[0], p[1], 0.055]), R))
        )
    elif name == "parts_sorting":
        for n in ("nut0", "bolt0", "standoff0", "standoff1", "standoff2"):
            b = env.part_bid[n]
            p = d.xpos[b].copy()
            Rm = d.xmat[b].reshape(3, 3)
            ax = Rm[:, 2] if n.startswith(("bolt", "standoff")) else None
            yaw = 0.0
            if ax is not None and abs(ax[2]) < 0.5:  # lying: close across the axis
                yaw = float(np.arctan2(ax[1], ax[0]))
            elif ax is None:
                yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
            gz = max(p[2], 0.009) if not n.startswith("bolt") else p[2]
            out.append(
                (n, lift_test(env, b, p + np.array([0, 0, 0.10]), np.array([p[0], p[1], gz]), down_rot(yaw), lift=0.10))
            )
            env.reset(seed=0)
    elif name == "fuse_holder":
        b = env.fuse_bid["fuse_new0"]
        p = d.xpos[b].copy()
        out.append(
            (
                "fresh fuse in caddy",
                lift_test(
                    env, b, p + np.array([0, 0, 0.12]), np.array([p[0], p[1], p[2] + 0.01]), down_rot(0.0), lift=0.10
                ),
            )
        )
    elif name == "object_handoff":
        b = env.bar
        p = d.xpos[b].copy()
        yaw = float(np.arctan2(d.xmat[b].reshape(3, 3)[1, 0], d.xmat[b].reshape(3, 3)[0, 0]))
        out.append(
            (
                "bar (left arm)",
                lift_test(env, b, p + np.array([0, 0, 0.14]), np.array([p[0], p[1], 0.075]), down_rot(yaw), arm=0),
            )
        )
    elif name == "plate_racking":
        b = env.plate
        p = d.xpos[b].copy()
        rim = np.array([p[0] - 0.1 + 0.012, p[1], p[2]])
        R = forward_rot(0.0, np.pi / 2)
        out.append(
            (
                "plate rim (hand horizontal)",
                lift_test(
                    env, b, rim + np.array([-0.08, 0, 0.0]), rim, R, lift=0.08, stage=rim + np.array([-0.16, 0, 0.12])
                ),
            )
        )
    elif name == "drive_replacement":
        b = env.bid["drive_new"]
        p = d.xpos[b].copy()
        R = forward_rot(0.0, np.pi / 2)
        out.append(
            (
                "spare drive (hand horizontal)",
                lift_test(
                    env,
                    b,
                    p + np.array([-0.12, 0, 0.0]),
                    p + np.array([-0.03, 0, 0.0]),
                    R,
                    lift=0.08,
                    stage=p + np.array([-0.20, 0, 0.12]),
                ),
            )
        )
    elif name == "relative_placement":
        for n in ("soda can", "mug", "apple", "toy block"):
            b = env.bid[n]
            p = d.xpos[b].copy()
            Rm = d.xmat[b].reshape(3, 3)
            yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0])) + (np.pi / 2 if n in ("mug", "toy block") else 0)
            out.append(
                (
                    n,
                    lift_test(
                        env, b, p + np.array([0, 0, 0.12]), np.array([p[0], p[1], max(p[2], 0.035)]), down_rot(yaw)
                    ),
                )
            )
            env.reset(seed=0)
    env.close()
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "bottle_binning",
        "pouring_beads",
        "parts_sorting",
        "fuse_holder",
        "object_handoff",
        "plate_racking",
        "drive_replacement",
        "relative_placement",
    ]
    fails = 0
    for n in names:
        for label, (ok, rise, ik) in run(n):
            fails += not ok
            print(
                f"{'PASS' if ok else 'FAIL'}  {n:18s} {label:32s} rise={rise * 100:5.1f} cm  ik={'ok' if ik else 'NO'}"
            )
    sys.exit(1 if fails else 0)
