"""Gymnasium environment: Franka plugs a cable into a patch-panel port.

Observation  : {"observation.state": float32[8]   (7 joint angles, gripper opening 0..1),
                "observation.images.overview": uint8[H,W,3],
                "observation.images.wrist":    uint8[H,W,3]}
Action       : float32[8]  (7 absolute joint-position targets in rad, gripper 0=closed..1=open)
Control rate : 50 Hz (1 control step of 20 ms per env step; physics 2 ms)
Success      : the port latch has engaged (plug tip seated and aligned) -- see `_maybe_latch`.

The port latch is a weld constraint that switches on once the tip is within `LATCH_TOL`
of fully seated; it models the retention latch a real port has (otherwise the hanging cable
pulls the plug back out as soon as the gripper lets go).
"""
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent))
import build_scene as bs

CTRL_EVERY = 10                 # physics steps per control step (0.002 s each -> 50 Hz control)
CTRL_PER_STEP = 1               # control steps per env step -> 50 Hz (the scripted expert was tuned at this rate)
FPS = round(1.0 / (CTRL_PER_STEP * CTRL_EVERY * 0.002))
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
LATCH_YZ_TOL = 0.002            # tip must be within 2 mm of the port axis ...
LATCH_X_TOL = 0.0025            # ... and within 2.5 mm of fully seated
CAMERAS = {"observation.images.overview": "closeup", "observation.images.wrist": "wrist"}
STATE_NAMES = [f"joint{i}" for i in range(1, 8)] + ["gripper"]


class RackCableEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, port: int = 2, img_hw=(192, 256), randomize: bool = True,
                 render: bool = True, max_seconds: float = 45.0, rand_scale: float = 1.0):
        self.port, self.randomize, self.max_seconds = port, randomize, max_seconds
        self.rand_scale = rand_scale    # 1.0 = training distribution; >1 widens the cable start pose range
        self.img_hw = tuple(img_hw)
        self.m, _ = bs.build_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_qadr = [m.jnt_qposadr[m.joint(f"panda_joint{i}").id] for i in range(1, 8)]
        self.arm_jid = [m.joint(f"panda_joint{i}").id for i in range(1, 8)]
        self.finger_qadr = m.jnt_qposadr[m.joint("panda_finger_joint1").id]
        self.cable_qadr = m.jnt_qposadr[m.joint("cable_free").id]
        self.tcp = m.site("tcp").id
        self.tip = m.site("plug_tip_site").id
        self.port_site = m.site(f"port{port}").id
        self.plug_body = m.body(f"cable{bs.N_SEG - 1}").id
        self.eq_latch = m.equality("latch").id
        self.latched = False
        self._renderer = mujoco.Renderer(m, *self.img_hw) if render else None
        lo, hi = m.jnt_range[self.arm_jid, 0], m.jnt_range[self.arm_jid, 1]
        self.action_space = spaces.Box(np.r_[lo, 0.0].astype(np.float32), np.r_[hi, 1.0].astype(np.float32))
        obs_spaces = {"observation.state": spaces.Box(-np.inf, np.inf, (8,), np.float32)}
        if render:
            for k in CAMERAS:
                obs_spaces[k] = spaces.Box(0, 255, (*self.img_hw, 3), np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)
        self.q_des = np.zeros(7)
        self.grip = GRIP_OPEN
        self._steps = 0
        self.reset(seed=0)

    # ---- helpers ----
    @property
    def instruction(self) -> str:
        return f"plug the cable into port {self.port}"

    def tip_pos(self):
        return self.d.site_xpos[self.tip].copy()

    def port_pos(self):
        return self.d.site_xpos[self.port_site].copy()

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def tip_error(self):
        """Tip position relative to the fully-seated target (m)."""
        return self.tip_pos() - self.port_pos()

    def get_state(self):
        q = self.d.qpos[self.arm_qadr]
        grip = np.clip(self.d.qpos[self.finger_qadr] / 0.04, 0.0, 1.0)
        return np.r_[q, grip].astype(np.float32)

    def get_obs(self):
        obs = {"observation.state": self.get_state()}
        if self._renderer is not None:
            for key, cam in CAMERAS.items():
                self._renderer.update_scene(self.d, camera=cam)
                obs[key] = self._renderer.render().copy()
        return obs

    # ---- latch ----
    def _maybe_latch(self):
        if self.latched:
            return
        e = self.tip_error()
        if abs(e[1]) < LATCH_YZ_TOL and abs(e[2]) < LATCH_YZ_TOL and e[0] > -LATCH_X_TOL:
            self._engage_latch()

    def _engage_latch(self):
        """Freeze the plug at its current pose with the weld constraint."""
        m, d = self.m, self.d
        b = self.plug_body
        p, q = d.xpos[b].copy(), d.xquat[b].copy()
        qi = np.zeros(4); mujoco.mju_negQuat(qi, q)
        pi = np.zeros(3); mujoco.mju_rotVecQuat(pi, -p, qi)
        best = None
        for rel in (np.concatenate([p, q]), np.concatenate([pi, qi])):  # try both weld conventions
            m.eq_data[self.eq_latch, :] = 0.0
            m.eq_data[self.eq_latch, 3:10] = rel
            m.eq_data[self.eq_latch, 10] = 1.0
            d.eq_active[self.eq_latch] = 1
            mujoco.mj_forward(m, d)
            sel = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == self.eq_latch)
            err = np.abs(d.efc_pos[sel]).max() if sel.any() else np.inf
            if best is None or err < best[0]:
                best = (err, rel.copy())
        m.eq_data[self.eq_latch, 3:10] = best[1]
        d.eq_active[self.eq_latch] = 1
        mujoco.mj_forward(m, d)
        self.latched = True

    # ---- low-level control (used by the scripted expert) ----
    def control_step(self, q_des, grip_raw):
        self.d.ctrl[:7] = q_des
        self.d.ctrl[7] = grip_raw
        for _ in range(CTRL_EVERY):
            mujoco.mj_step(self.m, self.d)
        self._maybe_latch()

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self.latched = False
        d.eq_active[self.eq_latch] = 0
        if self.randomize:
            rng = self.np_random
            k = self.rand_scale
            dx = rng.uniform(-0.015, 0.0) * k   # slide back from the table edge
            dy = rng.uniform(-0.04, 0.04) * k
            yaw = rng.uniform(-0.25, 0.25) * k  # ~ +-14 deg about the vertical
            a = self.cable_qadr
            d.qpos[a:a + 3] += np.array([dx, dy, 0.0])
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.q_des = d.qpos[self.arm_qadr].copy()
        self.grip = GRIP_OPEN
        mujoco.mj_forward(m, d)
        for _ in range(int(1.0 / (CTRL_EVERY * m.opt.timestep))):   # let the cable settle
            self.control_step(self.q_des, self.grip)
        self._steps = 0
        return self.get_obs(), {"instruction": self.instruction, "port": self.port}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        lo, hi = self.action_space.low, self.action_space.high
        a = np.clip(a, lo, hi)
        self.q_des = a[:7]
        self.grip = float(a[7]) * GRIP_OPEN
        for _ in range(CTRL_PER_STEP):
            self.control_step(self.q_des, self.grip)
        self._steps += 1
        t = self._steps * CTRL_PER_STEP * CTRL_EVERY * self.m.opt.timestep
        terminated = bool(self.latched)
        truncated = (not terminated) and t >= self.max_seconds
        e = self.tip_error()
        info = {"is_success": terminated, "tip_error_mm": e * 1000.0, "time": t}
        return self.get_obs(), float(terminated), terminated, truncated, info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
