"""Scripted bimanual handoff: left arm picks the bar by its top, right arm takes it by the lower part, left lets go,
right places it on the pad. Verifies the env's contact-based handoff tracking against real two-arm contact.

Known limit: tuned on seed 0 only. On other seeds the random bar yaw leaves the bar rotated at the handoff point and
the right hand's lower grasp cannot reach the pad height (a horizontal hand needs the grasp point >= 12 cm up).
"""

import sys

import numpy as np

import envs

from ..common.arms import down_rot, forward_rot
from ..tasks.object_handoff import BAR_HALF, PAD_C
from .scripted import Driver


def handoff(env, verbose=True):
    d = env.d
    L, Rr = Driver(env, 0), Driver(env, 1)
    bar = env.bar
    p = d.xpos[bar].copy()
    yaw = float(np.arctan2(d.xmat[bar].reshape(3, 3)[1, 0], d.xmat[bar].reshape(3, 3)[0, 0]))
    RL = down_rot(yaw)
    log = []

    def note(tag):
        log.append((tag, env.stage))
        if verbose:
            print(f"  {tag:34s} stage={env.stage} bar={np.round(d.xpos[bar], 3)}")

    # 1. left picks the bar by its upper part
    L.move(np.array([p[0], p[1], 0.32]), RL, 1.8, grip=1.0)
    L.move_linear(np.array([p[0], p[1], 0.12]), RL, 1.4)
    L.grasp(0.8)
    note("left grasp")
    L.move_linear(np.array([p[0], p[1], 0.30]), RL, 1.2)
    L.move_linear(np.array([0.42, 0.0, 0.34]), RL, 1.8)
    note("left at handoff point")
    # 2. right approaches horizontally (pointing -x) and takes the lower part
    b = d.xpos[bar].copy()
    RR = forward_rot(np.pi, 0.0)
    grasp_r = np.array([b[0], b[1], b[2] - 0.03])
    Rr.move(grasp_r + np.array([0.20, 0.0, 0.08]), RR, 2.2, grip=1.0)
    Rr.move_linear(grasp_r + np.array([0.08, 0.0, 0.0]), RR, 1.0)
    Rr.move_linear(grasp_r, RR, 1.2)
    Rr.grasp(0.8)
    note("right grasp (both hold)")
    # 3. left releases and retreats upward
    L.release(0.6)
    note("left released")
    L.move_linear(np.array([0.40, 0.0, 0.55]), RL, 1.0)
    off = Rr.arm.tcp_pos(d) - d.xpos[bar]
    # 4. right carries the bar to the pad and sets it down
    goal = np.array([PAD_C[0], PAD_C[1], BAR_HALF[2] + 0.004]) + off
    Rr.move_linear(np.array([goal[0], goal[1], goal[2] + 0.12]), RR, 2.0)
    Rr.move_linear(goal, RR, 1.2)
    Rr.release(0.8)
    note("right released on pad")
    Rr.move_linear(goal + np.array([0.10, 0, 0.05]), RR, 1.0)
    Rr.hold(1.0)
    return env.evaluate(), log


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    wins = 0
    for seed in range(n):
        env = envs.make("object_handoff", render=False)
        env.reset(seed=seed)
        ev, log = handoff(env)
        wins += ev["success"]
        print(f"seed {seed}: success={ev['success']} stage={ev['stage']} progress={ev['progress']:.2f}")
    print(f"{wins}/{n} scripted handoffs succeeded")
