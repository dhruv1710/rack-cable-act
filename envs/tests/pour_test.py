"""Scripted pour: grasp the cup, carry it over the bowl, tilt, return, release. Reports the env's own metrics."""

import sys

import numpy as np

import envs

from ..common.arms import down_rot
from ..tasks.pouring_beads import rot_y
from .scripted import Driver


def pour(env, tilt_max=2.35, verbose=True, back=0.03, z=0.15):
    d = env.d
    drv = Driver(env)
    R = down_rot(0.0)  # fingers close along y; tilt about world y pours toward +x (away from the robot)
    c = d.xpos[env.cup].copy()
    drv.move(np.array([c[0], c[1], 0.30]), R, 1.8, grip=1.0)
    drv.move_linear(np.array([c[0], c[1], 0.055]), R, 1.4)
    drv.grasp(0.8)
    drv.move_linear(np.array([c[0], c[1], 0.20]), R, 1.2)
    bowl = env.bowl_c
    over = np.array([bowl[0] - back, bowl[1], z])  # low and close: the lip ends ~3 cm above the bowl floor
    drv.move_linear(np.array([over[0], over[1], 0.20]), R, 1.6)
    drv.move_linear(over, R, 1.0)
    steps = 8
    for k in range(1, steps + 1):  # slow tilt about the TCP
        a = tilt_max * k / steps
        drv.move_linear(over, rot_y(a) @ R, 0.5)
    for _ in range(20):  # wait until the cup is empty (<= 2 beads left), max ~10 s
        drv.hold(0.5)
        if env.beads_in_cup() <= 2:
            break
    drv.hold(1.0)
    n_in, n_sp = env._counts()
    if verbose:
        print(f"  after pour: in bowl {n_in}/100, spilled {n_sp}")
    Rtilt = rot_y(tilt_max) @ R
    drv.move_linear(np.array([over[0] - 0.02, over[1], 0.27]), Rtilt, 1.5)  # rise clear of the bowl while still tipped
    for k in range(steps, 0, -1):  # untilt in the air
        drv.move_linear(np.array([over[0] - 0.02, over[1], 0.27]), rot_y(tilt_max * (k - 1) / steps) @ R, 0.4)
    drv.move_linear(np.array([c[0], c[1], 0.20]), R, 1.5)
    drv.move_linear(np.array([c[0], c[1], 0.058]), R, 1.2)
    drv.release(0.6)
    drv.move_linear(np.array([c[0], c[1], 0.25]), R, 1.0)
    drv.hold(1.0)
    return env.evaluate()


if __name__ == "__main__":
    ev = None
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    wins = 0
    for seed in range(n):
        env = envs.make("pouring_beads", render=False)
        env.reset(seed=seed)
        ev = pour(env)
        wins += ev["success"]
        print(f"seed {seed}: success={ev['success']} in_bowl={ev['beads_in_bowl']} spilled={ev['beads_spilled']}")
    print(f"{wins}/{n} scripted pours succeeded")
