"""Scripted pick-and-insert: Franka grabs the plug from the table and seats it in a port.

Usage:
    .venv/bin/python tasks/rack_cable/scripted_insert.py --port 2 --video out/rack_cable/insert.mp4

Phases: open -> above plug -> descend -> grasp -> lift -> carry to pre-insert ->
        push in (closed loop on the plug tip) -> release -> retract -> success check.
Uses privileged sim state (plug pose) -- fine for a scripted demo / data generator.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_scene as bs
from env import CTRL_EVERY, CTRL_PER_STEP, GRIP_CLOSED, GRIP_OPEN, RackCableEnv

# hand pointing down, fingers closing along world y (plug is long along x)
R_DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


def rot_vec(R_cur, R_tgt):
    """Orientation error as a rotation vector (world frame)."""
    return 0.5 * sum(np.cross(R_cur[:, i], R_tgt[:, i]) for i in range(3))


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Demo:
    """Scripted expert acting on a RackCableEnv.

    The expert runs at the ENV rate (25 Hz): one IK update per env step, and the resulting
    action is held for the CTRL_PER_STEP control steps it covers. So the recorded
    (observation, action) pairs are exactly what `env.step()` would execute.
    `record_cb(q_des, grip01)` (optional) is called once per env step, before the action is applied.
    """

    def __init__(self, env: RackCableEnv, video: str | None = None, record_cb=None):
        self.env = env
        self.m, self.d = env.m, env.d
        m = self.m
        self.arm_dofs = [m.jnt_dofadr[m.joint(f"panda_joint{i}").id] for i in range(1, 8)]
        self.arm_qadr = env.arm_qadr
        self.tcp, self.tip = env.tcp, env.tip
        self.grip_site = m.site("plug_grip_site").id
        self.plug_body = env.plug_body
        self.port = env.port
        self.port_site = env.port_site
        self.q_des = env.q_des.copy()
        self.q_home = self.q_des.copy()
        self.grip = GRIP_OPEN
        self.frames, self.renderer, self.cams = [], None, ("front", "closeup", "wrist")
        if video:
            self.renderer = mujoco.Renderer(m, 360, 480)
        self.t_frame = 0.0
        self.debug = False
        self.trace = None
        self._t_trace = 0.0
        self.record_cb = record_cb
        self.dt = CTRL_PER_STEP * CTRL_EVERY * self.m.opt.timestep   # 0.04 s per expert step
        self.R_tgt = R_DOWN.copy()

    # --- sim helpers ---
    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def tip_pos(self):
        return self.d.site_xpos[self.tip].copy()

    def plug_axis(self):
        return self.d.xmat[self.plug_body].reshape(3, 3)[:, 0].copy()

    def port_pos(self):
        return self.d.site_xpos[self.port_site].copy()

    def step(self):
        q, g = self.q_des.copy(), self.grip
        if self.record_cb is not None:
            self.record_cb(q, g / GRIP_OPEN)
        for _ in range(CTRL_PER_STEP):
            self.env.control_step(q, g)
        if self.trace and self.d.time - self._t_trace >= 0.4 and self.d.time > self.trace[0] and self.d.time < self.trace[1]:
            self._t_trace = self.d.time
            fj = self.m.jnt_qposadr[self.m.joint("panda_finger_joint1").id]
            q = self.d.qpos[self.arm_qadr]
            print(f"    t={self.d.time:5.2f} tcp {self.tcp_pos().round(3)} plug-grip {(self.d.site_xpos[self.grip_site] - self.tcp_pos()).round(3)} "
                  f"finger_q {self.d.qpos[fj]:.4f} |qdes-q| {np.abs(self.q_des - q).max():.3f} qdes7 {self.q_des[6]:.2f} q1 {q[0]:.2f}")
        if self.renderer is not None and self.d.time - self.t_frame >= 1 / 20:
            self.t_frame = self.d.time
            imgs = []
            for c in self.cams:
                self.renderer.update_scene(self.d, camera=c)
                imgs.append(self.renderer.render().copy())
            self.frames.append(np.concatenate(imgs, axis=1))

    def settle(self, seconds):
        for _ in range(int(seconds / self.dt)):
            self.step()

    # --- IK ---
    def ik_step(self, p_tgt, gain=6.0, rot_gain=4.0, max_dq=0.05):
        m, d = self.m, self.d
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
        J = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])
        R_cur = d.site_xmat[self.tcp].reshape(3, 3)
        err = np.concatenate([gain * (p_tgt - self.tcp_pos()), rot_gain * rot_vec(R_cur, self.R_tgt)])
        lam = 0.05
        JJt = J @ J.T + lam**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # weak null-space pull toward home keeps the elbow sensible. Must use the EXACT null
        # projector: the damped one leaks into task space near stretched poses and was
        # dragging the TCP back toward home during the push.
        N = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
        dq += N @ (0.1 * (self.q_home - self.q_des))
        dq = np.clip(dq * self.dt, -max_dq, max_dq)
        jid = [m.joint(f"panda_joint{i}").id for i in range(1, 8)]
        q_now = d.qpos[self.arm_qadr]
        q_new = np.clip(self.q_des + dq, q_now - 0.15, q_now + 0.15)   # anti-windup
        self.q_des = np.clip(q_new, m.jnt_range[jid, 0], m.jnt_range[jid, 1])

    def level_plug(self, k=0.25):
        """Nudge the hand orientation target so the plug axis converges to +x."""
        w = np.cross(self.plug_axis(), np.array([1.0, 0, 0]))
        self.R_tgt = rodrigues(k * w) @ self.R_tgt

    def align_plug(self, timeout=8.0, tol=0.012, k=0.3):
        """Rotate the hand about the TCP until the plug axis points along +x (yaw and pitch).

        Must actually spend time rotating: a position-tolerance wait exits immediately because
        the TCP is already in place.
        """
        p0 = self.tcp_pos(); t0 = self.d.time
        while self.d.time - t0 < timeout:
            w = np.cross(self.plug_axis(), np.array([1.0, 0, 0]))
            if np.linalg.norm(w) < tol:
                break
            if np.linalg.norm(w) > 0.3:
                w *= 0.3 / np.linalg.norm(w)
            # target is relative to the ACTUAL hand orientation, so it cannot wind up
            self.R_tgt = rodrigues(0.5 * w) @ self.d.site_xmat[self.tcp].reshape(3, 3).copy()
            self.ik_step(p0)
            self.step()
        for _ in range(15):          # hold the final orientation briefly
            self.ik_step(p0)
            self.step()

    def move_to(self, p_tgt, speed=0.15, tol=0.002, timeout=12.0, track=None, level=False):
        """Move TCP linearly to p_tgt. `track` optionally adjusts the target each step."""
        p0 = self.tcp_pos()
        dist = np.linalg.norm(p_tgt - p0)
        n = max(1, int(dist / speed / self.dt))
        for i in range(1, n + 1):
            wp = p0 + (p_tgt - p0) * i / n
            if track is not None:
                wp = wp + track()
            if level:
                self.level_plug()
            self.ik_step(wp)
            self.step()
        t0 = self.d.time
        while np.linalg.norm(self.tcp_pos() - (p_tgt + (track() if track else 0))) > tol and self.d.time - t0 < timeout:
            if level:
                self.level_plug()
            self.ik_step(p_tgt + (track() if track else 0))
            self.step()

    def rotate_in_place(self, timeout=6.0, tol=0.03):
        p = self.tcp_pos(); t0 = self.d.time
        while self.d.time - t0 < timeout:
            R_cur = self.d.site_xmat[self.tcp].reshape(3, 3)
            if np.linalg.norm(rot_vec(R_cur, self.R_tgt)) < tol:
                break
            self.ik_step(p); self.step()

    def hold_gripper(self, value, seconds=0.7):
        self.grip = value
        self.settle(seconds)

    # --- task ---
    def run(self):
        log = lambda s: print(f"[{self.d.time:6.2f}s] {s}")
        plug = self.d.site_xpos[self.grip_site].copy()
        log(f"plug grip site at {plug.round(3)}; target port {self.port} at {self.port_pos().round(4)}")

        # align hand yaw with the plug axis (projected on the table plane)
        a = self.plug_axis(); yaw = np.arctan2(a[1], a[0])
        R_plug = rodrigues(np.array([0, 0, yaw])) @ R_DOWN

        self.grip = GRIP_OPEN
        # the palm is 0.2 m wide along the finger axis: rise and yaw well above the rack
        # before descending, otherwise it clips the socket funnels
        self.R_tgt = self.d.site_xmat[self.tcp].reshape(3, 3).copy()
        p = self.tcp_pos()
        # loose tolerance + short timeouts: the strict defaults (2 mm / 12 s, 6 s) made the arm sit still
        # for ~10 s here, a dead stretch ACT cannot learn to leave (the scene looks identical throughout)
        self.move_to(np.array([p[0], p[1], 0.72]), speed=0.3, tol=0.01, timeout=1.5)
        self.R_tgt = R_plug
        self.rotate_in_place(timeout=2.0)
        log(f"yawed above rack; hand R err {np.linalg.norm(rot_vec(self.d.site_xmat[self.tcp].reshape(3, 3), R_plug)):.3f}")
        self.move_to(np.array([plug[0], plug[1], 0.45]), speed=0.3)
        log("above plug")
        self.move_to(np.array([plug[0], plug[1], plug[2] + 0.008]), speed=0.08)
        log(f"descended, tcp {self.tcp_pos().round(3)}")
        self.hold_gripper(GRIP_CLOSED, 0.8)
        log("grasped")
        self.move_to(self.tcp_pos() + np.array([0, 0, 0.18]), speed=0.1)
        log(f"lifted; plug axis {self.plug_axis().round(3)}")

        # carry: compute where the TCP must be so the plug tip lands at the pre-insert point
        port = self.port_pos()
        pre_tip = port + np.array([-0.03, 0, 0])
        self.hold_gripper(GRIP_CLOSED, 0.2)
        self.align_plug()
        log(f"leveled; plug axis {self.plug_axis().round(3)}")
        # stage 10 cm in front of the panel at the port's y/z, so all lateral motion happens far
        # from the funnels; only then advance straight in along x
        stage_tip = port + np.array([-0.10, 0, 0])
        via = self.tcp_pos() + (stage_tip - self.tip_pos())
        self.move_to(via, speed=0.12, level=True)
        for _ in range(2):
            self.move_to(self.tcp_pos() + (stage_tip - self.tip_pos()), speed=0.05, tol=0.001, timeout=3.0, level=True)
        log(f"staged; tip err {np.round((self.tip_pos() - stage_tip) * 1000, 2)} mm")
        for _ in range(3):  # converge tip onto pre-insert point
            self.move_to(self.tcp_pos() + (pre_tip - self.tip_pos()), speed=0.05, tol=0.0005, timeout=3.0, level=True)
        err = self.tip_pos() - pre_tip
        log(f"pre-insert tip err {np.round(err * 1000, 2)} mm")

        # push in. Lateral (y,z) error is corrected every control step with an integral term;
        # the forward schedule stalls whenever the tip lags it (jam) and the commanded forward
        # offset is capped, so contact force stays bounded instead of winding up.
        depth_goal = bs.PANEL_FRONT_X - 0.002
        lat_int = np.zeros(2)
        sched = self.tip_pos()[0]
        for i in range(1, 900):
            e = port - self.tip_pos()
            lat_int = np.clip(lat_int + 0.04 * e[1:], -0.003, 0.003)
            tip_x = self.tip_pos()[0]
            if tip_x >= depth_goal or self.env.latched:
                break
            if sched - tip_x < 0.0015:
                sched = min(depth_goal, sched + 0.00015)
            self.level_plug(0.15)
            wp = self.tcp_pos()
            wp[0] += np.clip(sched - tip_x, -0.001, 0.0015)
            wp[1:] += e[1:] + lat_int
            self.ik_step(wp, gain=15.0)
            self.step()
            if self.debug and i in (60, 120, 200, 320):
                m_, d_ = self.m, self.d
                jid = [m_.joint(f"panda_joint{k}").id for k in range(1, 8)]
                q = d_.qpos[self.arm_qadr]
                R = d_.site_xmat[self.tcp].reshape(3, 3)
                print(f"  [state i={i}] tcp {self.tcp_pos().round(4)} tip {self.tip_pos().round(4)} sched {sched:.4f}")
                print(f"     q     {q.round(3)}")
                print(f"     q_des {self.q_des.round(3)}  (q_des-q) {(self.q_des - q).round(3)}")
                print(f"     lo    {m_.jnt_range[jid, 0].round(2)}  hi {m_.jnt_range[jid, 1].round(2)}")
                print(f"     rot err {np.round(rot_vec(R, self.R_tgt), 4)}  ctrl-tcp-x-cmd {np.clip(sched - self.tip_pos()[0], -0.001, 0.0015):.4f}")
            if self.debug and i % 20 == 0:
                fj = self.m.jnt_qposadr[self.m.joint("panda_finger_joint1").id]
                print(f"  push i={i:3d} tip-port mm {np.round((self.tip_pos() - port) * 1000, 1)} "
                      f"axis {self.plug_axis().round(3)} finger_q {self.d.qpos[fj]:.4f} lat_int mm {np.round(lat_int * 1000, 1)}")
        self.settle(0.5)
        e = self.tip_pos() - port
        log(f"pushed; tip rel. port (mm) {np.round(e * 1000, 2)}")

        self.hold_gripper(GRIP_OPEN, 0.8)
        self.move_to(self.tcp_pos() + np.array([-0.08, 0, 0]), speed=0.05, tol=0.005, timeout=3.0)
        self.settle(0.5)
        return self.success()

    def dump_contacts(self):
        m, d = self.m, self.d
        nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ("body:" + mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
        f = np.zeros(6)
        print("  --- contacts at stall (force N, excluding cable-cable) ---")
        for ci in range(d.ncon):
            c = d.contact[ci]
            a, b = nm(c.geom1), nm(c.geom2)
            if a.startswith("cable_g") and b.startswith("cable_g"):
                continue
            mujoco.mj_contactForce(m, d, ci, f)
            print(f"   {a:>22s} <-> {b:<22s} |Fn|={abs(f[0]):7.3f}  pos {np.round(c.pos, 4)}")
        print("  cable first/last seg pos:", d.xpos[m.body('cable0').id].round(3), d.xpos[self.plug_body].round(3))

    def success(self, verbose=True):
        e = self.env.tip_error()
        ok = bool(self.env.latched) and abs(e[1]) < 0.003 and abs(e[2]) < 0.003
        if verbose:
            depth = self.tip[0] if False else self.env.tip_pos()[0] - (bs.PANEL_FRONT_X - bs.SOCKET_DEPTH)
            print(f"RESULT: {'SUCCESS' if ok else 'FAIL'} | latched={self.env.latched} | tip-port error (mm) "
                  f"{np.round(e * 1000, 2)} | insertion depth {depth * 1000:.1f}/{bs.SOCKET_DEPTH * 1000:.0f} mm")
        return ok

    def save_video(self, path):
        import imageio
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=20, macro_block_size=1)
        print(f"saved {path} ({len(self.frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2, choices=range(len(bs.PORT_Y)))
    ap.add_argument("--video", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="randomize cable start pose")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = RackCableEnv(port=args.port, randomize=args.randomize, render=False)
    env.reset(seed=args.seed)
    demo = Demo(env, args.video)
    demo.debug = args.debug
    demo.trace = (21.0, 42.0) if args.debug else None
    ok = demo.run()
    if args.video:
        demo.save_video(args.video)
    sys.exit(0 if ok else 1)
