"""Base Gymnasium environment shared by all tasks.

Observation : {"observation.state": float32[8*n_arms], "observation.images.<cam>": uint8[H,W,3]}
Action      : float32[8*n_arms]  per arm: 7 absolute joint-position targets (rad) + gripper (0 closed..1 open)
Control     : 50 Hz (10 physics steps of 2 ms)
Reward      : 1.0 on success else 0.0; info carries `progress` in [0,1] and task metrics.

Subclasses implement build() / bind() / randomize() / evaluate() (see TaskEnv docstrings).
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .arms import GRIP_OPEN, Q_HOME, Arm, ArmCfg, attach_panda, grip_ctrl
from .world import new_spec

CTRL_EVERY = 10


class TaskEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    name = "task"
    arm_cfgs = (ArmCfg(),)
    world_kw: dict = {}
    max_seconds = 60.0
    success_hold = 0.3  # evaluate()['success'] must hold this long (s) before the episode terminates
    settle_seconds = 1.0
    overview = ((0.95, -1.05, 0.75), (0.5, 0.0, 0.05))  # camera position, look-at point

    def __init__(self, img_hw=(192, 256), render=True, randomize=True, max_seconds=None):
        self.img_hw, self.randomize = tuple(img_hw), randomize
        if max_seconds is not None:
            self.max_seconds = max_seconds
        spec = new_spec(name=self.name, **self.world_kw)
        for cfg in self.arm_cfgs:
            attach_panda(spec, cfg)
        spec.worldbody.add_body(name="look", pos=self.overview[1])
        spec.worldbody.add_camera(
            name="overview",
            pos=self.overview[0],
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
            targetbody="look",
            fovy=55,
        )
        self.build(spec)
        self.spec = spec
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.arms = [Arm(self.m, c) for c in self.arm_cfgs]
        self.camera_map = {"overview": "overview"}
        for a in self.arms:
            self.camera_map[a.cfg.obs_name] = a.cam
        self.camera_map.update(self.extra_cameras())
        self.bind()
        self._renderer = mujoco.Renderer(self.m, *self.img_hw) if render else None
        self.action_space = self.make_action_space()
        ob = {"observation.state": spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)}
        if render:
            for k in self.camera_map:
                ob[f"observation.images.{k}"] = spaces.Box(0, 255, (*self.img_hw, 3), np.uint8)
        self.observation_space = spaces.Dict(ob)
        self.q_des = self.home_action()
        self._t, self._hold_t, self._best = 0.0, 0.0, 0.0
        self.reset(seed=0)

    # ------------------------------------------------------------------ hooks
    def build(self, spec):  # add task bodies/geoms to the MjSpec (before compile)
        pass

    def bind(self):  # cache ids after compile
        pass

    def extra_cameras(self) -> dict:
        return {}

    def randomize_state(self, rng):  # place objects (called after arms are at home, before settling)
        pass

    def after_settle(self):  # record reference poses etc.
        pass

    def on_substep(self):  # called after every physics step (latches, ...)
        pass

    def reach_targets(
        self,
    ):  # [(label, arm_index, pos, R_tcp)] poses a policy must be able to reach (IK-checked in tests)
        return []

    def oracle_solve(self):  # put the world in a solved state (used by tests); then call self.physics()
        raise NotImplementedError

    def evaluate(self) -> dict:  # must return {"success": bool, "progress": float, ...}
        return {"success": False, "progress": 0.0}

    @property
    def instruction(self) -> str:
        return ""

    @property
    def state_dim(self):
        return 8 * len(self.arms)

    def make_action_space(self):
        lo = np.concatenate([np.r_[a.lo, 0.0] for a in self.arms]).astype(np.float32)
        hi = np.concatenate([np.r_[a.hi, 1.0] for a in self.arms]).astype(np.float32)
        return spaces.Box(lo, hi)

    def home_action(self):
        return np.concatenate([np.r_[Q_HOME, 1.0] for _ in self.arms]) if self.arms else np.zeros(0)

    def noop_action(self):
        return self.q_des.astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def contacts_between(self, geoms_a, geoms_b) -> bool:
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in geoms_a and c.geom2 in geoms_b) or (c.geom2 in geoms_a and c.geom1 in geoms_b):
                return True
        return False

    def body_geoms(self, body_name) -> set:
        b = self.m.body(body_name).id
        return {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b}

    def subtree_geoms(self, body_name) -> set:
        root = self.m.body(body_name).id
        bodies = {root}
        for b in range(self.m.nbody):
            p = b
            while p != 0 and p not in bodies:
                p = self.m.body_parentid[p]
            if p in bodies:
                bodies.add(b)
        return {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in bodies}

    # ------------------------------------------------------------------ obs
    def get_state(self):
        return np.concatenate([a.state(self.d) for a in self.arms])

    def get_obs(self):
        obs = {"observation.state": self.get_state()}
        if self._renderer is not None:
            for k, cam in self.camera_map.items():
                self._renderer.update_scene(self.d, camera=cam)
                obs[f"observation.images.{k}"] = self._renderer.render().copy()
        return obs

    def render(self, camera="overview"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, *self.img_hw)
        self._renderer.update_scene(self.d, camera=self.camera_map.get(camera, camera))
        return self._renderer.render().copy()

    # ------------------------------------------------------------------ control
    def apply_action(self, a):
        for i, arm in enumerate(self.arms):
            seg = a[8 * i : 8 * i + 8]
            self.d.ctrl[arm.act] = seg[:7]
            self.d.ctrl[arm.grip_act] = grip_ctrl(seg[7])

    def physics(self, n=CTRL_EVERY):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.on_substep()

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for arm in self.arms:
            d.qpos[arm.qadr] = Q_HOME
            d.ctrl[arm.act] = Q_HOME
            d.ctrl[arm.grip_act] = GRIP_OPEN
            d.qpos[arm.finger_qadr] = 0.04
            d.qpos[arm.finger_qadr + 1] = 0.04
        self.q_des = self.home_action()
        self.reset_dynamic_model()
        self.randomize_state(self.np_random if self.randomize else np.random.default_rng(0))
        mujoco.mj_forward(m, d)
        for _ in range(int(self.settle_seconds / (CTRL_EVERY * m.opt.timestep))):
            self.physics()
        self.after_settle()
        self._t, self._hold_t, self._best = 0.0, 0.0, 0.0
        info = {"instruction": self.instruction}
        info.update(self.evaluate())
        return self.get_obs(), info

    def reset_dynamic_model(self):  # restore model-parameter changes from a previous episode
        pass

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), self.action_space.low, self.action_space.high)
        self.q_des = a
        self.apply_action(a)
        self.physics()
        self._t += CTRL_EVERY * self.m.opt.timestep
        info = self.evaluate()
        self._best = max(self._best, info.get("progress", 0.0))
        self._hold_t = self._hold_t + CTRL_EVERY * self.m.opt.timestep if info["success"] else 0.0
        terminated = self._hold_t >= self.success_hold
        truncated = (not terminated) and self._t >= self.max_seconds
        info.update(is_success=bool(terminated), time=self._t, instruction=self.instruction)
        return self.get_obs(), float(terminated), bool(terminated), bool(truncated), info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
