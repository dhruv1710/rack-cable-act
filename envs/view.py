"""Open tasks in the interactive MuJoCo viewer (macOS: run with mjpython).

    mjpython envs/view.py                    # all tasks, one after another (close a window -> next task)
    mjpython envs/view.py fuse_holder        # one task (also: mjpython -m envs.view fuse_holder)

Moving the arms: open the viewer's right-hand panel -> "Control". Each Panda has 7 joint sliders (target angle, rad)
and a gripper slider (255 = open, negative = squeeze). The arms hold whatever you set (no scripted control).
thermal_patrol: arrow keys drive (up/down = forward/back, left/right = turn), [ and ] pan the head, ; and ' tilt it,
SPACE flags an anomaly. The camera follows the robot, and hot servers are tinted (orange = hot) so you can see them.
Backspace resets the physics.
"""

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]  # envs/ itself must not shadow packages
sys.path.insert(0, str(_HERE.parent))  # repo root, so `mjpython envs/view.py` works

import mujoco
import mujoco.viewer
import numpy as np

import envs

# GLFW key codes
K_RIGHT, K_LEFT, K_DOWN, K_UP, K_SPACE = 262, 263, 264, 265, 32
K_LB, K_RB, K_SEMI, K_APOS = 91, 93, 59, 39


def aim_free_camera(cam, pos, look):
    pos, look = np.asarray(pos, float), np.asarray(look, float)
    v = pos - look
    dist = float(np.linalg.norm(v))
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
    cam.elevation = float(-np.degrees(np.arcsin(v[2] / dist)))


def tint_hot_servers(env):
    """Viewer-only x-ray: colour each server front by temperature so heat is visible in the 3D view."""
    m = env.m
    for gid, t in zip(env.srv_gid, env.srv_temp, strict=True):
        f = float(np.clip((t - 25.0) / 45.0, 0.0, 1.0))
        m.geom_matid[gid] = -1
        m.geom_rgba[gid] = (0.18 + 0.82 * f, 0.18 + 0.30 * f * (1 - f) * 2, 0.20 * (1 - f), 1.0)


def view(name):
    env = envs.make(name, render=False)
    env.reset(seed=0)
    m, d = env.m, env.d
    print(f"[{name}] {env.instruction}", flush=True)
    thermal = name == "thermal_patrol"
    legacy = not hasattr(env, "on_substep")  # the original rack_cable env has its own latch hook
    drive = {"v": 0.0, "w": 0.0, "pan": 0.0, "tilt": 0.0, "flag": 0.0}

    def on_key(key):
        if not thermal:
            return
        step_v, step_w = 0.5, 1.0
        if key == K_UP:
            drive["v"] = step_v if drive["v"] <= 0 else 0.0
        elif key == K_DOWN:
            drive["v"] = -step_v if drive["v"] >= 0 else 0.0
        elif key == K_LEFT:
            drive["w"] = step_w if drive["w"] <= 0 else 0.0
        elif key == K_RIGHT:
            drive["w"] = -step_w if drive["w"] >= 0 else 0.0
        elif key == K_LB:
            drive["pan"] = min(1.7, drive["pan"] + 0.2)
        elif key == K_RB:
            drive["pan"] = max(-1.7, drive["pan"] - 0.2)
        elif key == K_SEMI:
            drive["tilt"] = min(0.9, drive["tilt"] + 0.1)
        elif key == K_APOS:
            drive["tilt"] = max(-0.9, drive["tilt"] - 0.1)
        elif key == K_SPACE:
            drive["flag"] = 1.0

    if thermal:
        tint_hot_servers(env)

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        if thermal:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = m.body("robot").id
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.2, 180.0, -22.0
        else:
            aim_free_camera(v.cam, *env.overview)
        n_ctrl_steps = 10
        while v.is_running():
            t0 = time.time()
            if thermal:  # keyboard drives the robot; flag is a one-shot
                env.apply_action(np.array([drive["v"], drive["w"], drive["pan"], drive["tilt"], drive["flag"]]))
                if drive["flag"]:
                    print("flagged:", env.evaluate(), flush=True)
                    drive["flag"] = 0.0
            for _ in range(n_ctrl_steps):  # sliders set d.ctrl; we only advance physics and the task's latches
                mujoco.mj_step(m, d)
                if legacy:
                    env._maybe_latch()
                else:
                    env.on_substep()
            v.sync()
            time.sleep(max(0.0, m.opt.timestep * n_ctrl_steps - (time.time() - t0)))
    env.close()


if __name__ == "__main__":
    for n in sys.argv[1:] or envs.list_tasks():
        view(n)
