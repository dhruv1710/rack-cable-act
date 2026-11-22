"""Scripted reference policies for tasks beyond the ones in envs/tests/.

Status: bottle_binning is partial (about 3 of 8 seeds). It is not used for the published videos.

Each reads privileged simulator state (object poses), solves inverse kinematics and drives the arm through
``Driver``. They exist to show each task is solvable, to produce demonstration videos and, later, demonstration data.
"""

from __future__ import annotations

import numpy as np

from .common.arms import down_rot, solve_ik
from .tasks.bottle_binning import CRATE_C
from .tests.scripted import Driver


def reachable_R(env, drv, positions, yaw):
    """The hand is symmetric under a 180 degree turn: return the first yaw (of yaw, yaw+pi, yaw-pi) for which IK
    reaches every position, or None. Executing an unsolved IK result throws objects around, so callers skip instead."""
    for y in (yaw, yaw + np.pi, yaw - np.pi):
        R = down_rot(y)
        q0 = drv.joint_state()
        if all(solve_ik(env.m, env.d, drv.arm, pos, R, q_init=q0, iters=300, restarts=40)[1] for pos in positions):
            return R
    return None


CRATE_SLOTS = [(-0.07, -0.05), (0.07, -0.05), (-0.07, 0.05), (0.07, 0.05)]


def _on_belt(env, b):
    p = env.d.xpos[b]
    return p[2] > 0.0 and abs(p[1] - CRATE_C[1]) > 0.2 and abs(p[0]) < 1.0


def _clearance(d, b, pool):
    """Distance from bottle b to its nearest neighbour in pool."""
    return min((np.linalg.norm(d.xpos[b][:2] - d.xpos[o][:2]) for o in pool if o != b), default=9.0)


def bottle_binning(env):
    """Repeatedly pick the most isolated bottle left on the belt (standing: pinch the neck, lying: grasp the body)
    and drop it in its own crate slot. The open hand is only ~1 cm wider than a bottle on each side, so picking in
    order of clearance avoids shoving the neighbours."""
    d = env.d
    drv = Driver(env)
    for k in range(len(env.bid)):
        left = [b for b in env.bid if _on_belt(env, b)]
        if not left:
            break

        b = max(left, key=lambda c, pool=left: _clearance(d, c, pool))
        p = d.xpos[b].copy()
        up = abs(d.xmat[b].reshape(3, 3)[2, 2])
        if up > 0.9:  # standing
            yaw, pre, grasp = 0.0, np.array([p[0], p[1], 0.32]), np.array([p[0], p[1], 0.168])
        else:  # lying: close across the body at its centre, hand yawed to the bottle axis
            ax = d.xmat[b].reshape(3, 3)[:, 2]
            yaw = float(np.arctan2(ax[1], ax[0]))
            pre, grasp = p + np.array([0, 0, 0.12]), p.copy()
        sx, sy = CRATE_SLOTS[k % len(CRATE_SLOTS)]
        tx, ty = CRATE_C[0] + sx, CRATE_C[1] + sy
        R = reachable_R(env, drv, [pre, grasp, np.array([tx, ty, 0.40])], yaw)
        if R is None:
            continue
        drv.move(pre, R, 1.8, grip=1.0)
        drv.hold(0.5)
        drv.move_linear(grasp, R, 2.2)
        drv.hold(0.5)
        drv.grasp(0.8)
        drv.move_linear(grasp + np.array([0, 0, 0.22]), R, 1.2)
        drv.move(np.array([tx, ty, 0.40]), R, 2.0)
        drv.move_linear(np.array([tx, ty, 0.36]), R, 0.8)
        drv.release(0.7)
        drv.move_linear(np.array([tx, ty, 0.42]), R, 0.8)
    drv.hold(2.0)
    return env.evaluate()
