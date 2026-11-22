"""Scripted end-effector control helpers for tests/demos (IK -> joint targets, step the env)."""

import numpy as np

from ..common.arms import solve_ik


class Driver:
    def __init__(self, env, arm=0):
        self.env, self.arm = env, env.arms[arm]
        self.ai = arm
        self.grip = 1.0

    def _act(self, q, grip=None, other=None):
        env = self.env
        a = env.q_des.copy()
        a[8 * self.ai : 8 * self.ai + 7] = q
        a[8 * self.ai + 7] = self.grip if grip is None else grip
        return a

    def joint_state(self):
        return self.env.d.qpos[self.arm.qadr].copy()

    def move(self, pos, R, seconds=1.5, grip=None, q_seed=None):
        """IK to (pos, R) then interpolate joint targets over `seconds`. Returns IK convergence."""
        env = self.env
        q0 = self.joint_state()
        q1, ok = solve_ik(
            env.m, env.d, self.arm, pos, R, q_init=q_seed if q_seed is not None else q0, iters=300, restarts=40
        )
        if grip is not None:
            self.grip = grip
        n = max(2, int(seconds * 50))
        for i in range(1, n + 1):
            q = q0 + (q1 - q0) * (i / n)
            _obs, _r, _term, _trunc, info = env.step(self._act(q))
        return ok, info

    def hold(self, seconds=0.5, grip=None):
        if grip is not None:
            self.grip = grip
        q = self.env.q_des[8 * self.ai : 8 * self.ai + 7]
        for _ in range(int(seconds * 50)):
            _obs, _r, _term, _trunc, info = self.env.step(self._act(q))
        return info

    def grasp(self, seconds=0.8):
        return self.hold(seconds, grip=0.0)

    def release(self, seconds=0.5):
        return self.hold(seconds, grip=1.0)

    def move_linear(self, pos, R, seconds=1.5, grip=None, n_way=10):
        """Straight-line Cartesian move: IK at each waypoint (warm-started), tracked at 50 Hz."""
        env = self.env
        if grip is not None:
            self.grip = grip
        start = self.arm.tcp_pos(env.d)
        q_prev = self.joint_state()
        qs, ok_all = [], True
        for i in range(1, n_way + 1):
            wp = start + (np.asarray(pos) - start) * (i / n_way)
            q, ok = solve_ik(env.m, env.d, self.arm, wp, R, q_init=q_prev, iters=200)
            ok_all &= ok
            qs.append(q)
            q_prev = q
        steps = max(n_way, int(seconds * 50))
        info = None
        for t in range(1, steps + 1):
            f = t / steps * n_way
            i = min(int(f), n_way - 1)
            qa = qs[i - 1] if i > 0 else qs[0]
            qb = qs[i]
            q = (
                qa + (qb - qa) * (f - i)
                if i > 0
                else qs[0] * min(f, 1.0) + self.env.q_des[8 * self.ai : 8 * self.ai + 7] * (1 - min(f, 1.0))
            )
            _, _, _, _, info = env.step(self._act(q))
        return ok_all, info
