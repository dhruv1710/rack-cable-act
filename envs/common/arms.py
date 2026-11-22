"""Franka Panda (MuJoCo Menagerie) attach + handle + damped-least-squares IK."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PANDA_XML = Path(__file__).resolve().parents[2] / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
Q_HOME = np.array([0.0, -0.45, 0.0, -2.35, 0.0, 1.95, -0.785])  # hand pointing down, ~0.45 m ahead of the base
GRIP_FORCE_SCALE = 10.0  # Menagerie pinch is only ~2 N; scale gain+bias together
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
GRIP_SQUEEZE = 150.0  # ctrl magnitude below 0 when closed: ~23 N constant grasp force on top of the position spring


@dataclass
class ArmCfg:
    prefix: str = "panda_"
    pos: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    obs_name: str = "wrist"  # camera obs key for this arm


def attach_panda(spec: mujoco.MjSpec, cfg: ArmCfg):
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    c, s = np.cos(cfg.yaw / 2), np.sin(cfg.yaw / 2)
    frame = spec.worldbody.add_frame(pos=cfg.pos, quat=(c, 0, 0, s))
    spec.attach(panda, frame=frame, prefix=cfg.prefix)
    hand = spec.body(cfg.prefix + "hand")
    hand.add_site(name=cfg.prefix + "tcp", pos=(0, 0, 0.1034), size=(0.003,) * 3)
    hand.add_camera(name=cfg.prefix + "wrist", pos=(0.07, 0, 0.04), xyaxes=(1, 0, 0, 0, -1, 0), fovy=75)
    for side in (
        "left_finger",
        "right_finger",
    ):  # stiff pads with torsional friction (silicone-like), so pinched objects
        for g in spec.body(cfg.prefix + side).geoms:  # neither sink into thin walls nor spin freely in the grip
            if g.contype or g.conaffinity:
                g.solref, g.solimp = (0.004, 1.0), (0.99, 0.999, 0.0005, 0.5, 2)
                g.condim = 4
                g.friction = (max(g.friction[0], 1.0), 0.02, 0.001)
    act = spec.actuator(cfg.prefix + "actuator8")
    act.ctrlrange = (
        -GRIP_SQUEEZE,
        GRIP_OPEN,
    )  # negative ctrl = constant squeeze (a Franka hand holds a commanded force)
    act.gainprm[0] *= GRIP_FORCE_SCALE
    act.biasprm[1] *= GRIP_FORCE_SCALE
    act.biasprm[2] *= GRIP_FORCE_SCALE


class Arm:
    """Resolved ids for one attached Panda."""

    def __init__(self, m: mujoco.MjModel, cfg: ArmCfg):
        p = cfg.prefix
        self.cfg, self.prefix = cfg, p
        jn = [m.joint(f"{p}joint{i}") for i in range(1, 8)]
        self.jid = np.array([j.id for j in jn])
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.lo, self.hi = m.jnt_range[self.jid, 0].copy(), m.jnt_range[self.jid, 1].copy()
        self.act = np.array([m.actuator(f"{p}actuator{i}").id for i in range(1, 8)])
        self.grip_act = m.actuator(f"{p}actuator8").id
        self.finger_qadr = m.jnt_qposadr[m.joint(f"{p}finger_joint1").id]
        self.tcp = m.site(f"{p}tcp").id
        self.hand = m.body(f"{p}hand").id
        self.cam = f"{p}wrist"
        fb = {m.body(f"{p}left_finger").id, m.body(f"{p}right_finger").id}
        self.finger_bodies = fb
        self.finger_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in fb}
        self.hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in fb | {self.hand}}

    def state(self, d):
        return np.r_[d.qpos[self.qadr], np.clip(d.qpos[self.finger_qadr] / 0.04, 0, 1)].astype(np.float32)

    def tcp_pos(self, d):
        return d.site_xpos[self.tcp].copy()


def down_rot(yaw=0.0):
    """TCP rotation with the gripper pointing straight down and fingers opening along (cos yaw, sin yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])


def forward_rot(yaw=0.0, roll=0.0):
    """TCP pointing along the horizontal direction yaw (z_tcp = (cos, sin, 0)); fingers open along y' (roll=0)
    or vertically (roll=pi/2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    zt = np.array([c, s, 0.0])
    up = np.array([0, 0, 1.0])
    yt = np.cross(up, zt)  # horizontal, perpendicular
    xt = np.cross(yt, zt)  # points down-ish (= -up)
    R = np.c_[xt, yt, zt]
    cr, sr = np.cos(roll), np.sin(roll)
    return R @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])


def solve_ik(
    m, d, arm: Arm, target_pos, target_R=None, iters=200, tol=1e-4, damping=0.05, q_init=None, restarts=0, seed=0
):
    """Damped least-squares IK for the TCP, optionally with random restarts. Returns (q[7], converged)."""
    q, ok = _solve_ik(m, d, arm, target_pos, target_R, iters, tol, damping, q_init)
    rng = np.random.default_rng(seed)
    tries = 0
    while not ok and tries < restarts:
        seed_q = np.clip(Q_HOME + rng.normal(0, 0.6, 7), arm.lo + 0.05, arm.hi - 0.05)
        q, ok = _solve_ik(m, d, arm, target_pos, target_R, iters, tol, damping, seed_q)
        tries += 1
    return q, ok


def _solve_ik(m, d, arm, target_pos, target_R, iters, tol, damping, q_init):
    s = mujoco.MjData(m)
    s.qpos[:] = d.qpos
    if q_init is not None:
        s.qpos[arm.qadr] = q_init
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    ok = False
    for _ in range(iters):
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        pos = s.site_xpos[arm.tcp]
        R = s.site_xmat[arm.tcp].reshape(3, 3)
        ep = np.asarray(target_pos) - pos
        if target_R is not None:
            er = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
        else:
            er = np.zeros(3)
        if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 1e-3:
            ok = True
            break
        mujoco.mj_jacSite(m, s, jp, jr, arm.tcp)
        J = np.vstack([jp[:, arm.dadr], jr[:, arm.dadr]]) if target_R is not None else jp[:, arm.dadr]
        e = np.r_[ep, er] if target_R is not None else ep
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
        s.qpos[arm.qadr] = np.clip(s.qpos[arm.qadr] + np.clip(dq, -0.2, 0.2), arm.lo, arm.hi)
    return s.qpos[arm.qadr].copy(), ok


def grip_ctrl(g: float) -> float:
    """Map a normalised gripper command (0 = closed/squeezing, 1 = open) to the actuator control."""
    g = float(np.clip(g, 0.0, 1.0))
    return g * GRIP_OPEN - (1.0 - g) * GRIP_SQUEEZE
