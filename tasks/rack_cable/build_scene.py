"""Builds the rack + cable + Franka MuJoCo model.

Cable: chain of capsule segments joined by spring-damped ball joints (MuJoCo 4 has no
cable composite). Plug: grip block + tip on the last segment. Ports: 4 sockets on a panel.
"""
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
PANDA_XML = HERE.parents[1] / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

# geometry (metres)
PANEL_FRONT_X = 0.675
PANEL_Z = 0.55
PORT_Y = (-0.12, -0.04, 0.04, 0.12)
SOCKET_DEPTH = 0.025
FUNNEL_LEN, FUNNEL_ANGLE = 0.014, np.deg2rad(30)  # lead-in flare at each socket mouth
CLEARANCE = 0.0015  # per side between plug tip and socket hole (real RJ45 is ~0.3-0.5 mm)
WALL = 0.003
TIP_HALF = (0.020, 0.006, 0.005)  # plug tip half-extents; 40 mm long > socket depth + funnel so the wide body stays clear
HOLE_HALF_Y, HOLE_HALF_Z = TIP_HALF[1] + CLEARANCE, TIP_HALF[2] + CLEARANCE
BLOCK_HALF = (0.016, 0.009, 0.011)  # long body: fingers must stay behind the socket funnel when seated
GRASP_OFFSET = -0.006  # grasp the rear part of the body (relative to its centre)

GRIP_FORCE_SCALE = 10.0
N_SEG = 14
SEG_LEN = 0.015
CABLE_R = 0.004
CABLE_ROOT = (0.32, 0.0, 0.306)  # on the table (top at z=0.30)


def _quat(axis, ang):
    axis = np.asarray(axis, float)
    return np.concatenate([[np.cos(ang / 2)], axis * np.sin(ang / 2)])


def _add_funnel(rack, i, y):
    """Four angled plates flaring outward (toward the robot) from the socket mouth."""
    th, L, t = FUNNEL_ANGLE, FUNNEL_LEN, WALL
    xf = PANEL_FRONT_X - SOCKET_DEPTH            # socket mouth plane
    c, s_ = np.cos(th), np.sin(th)
    wy, wz = HOLE_HALF_Y + WALL + L * s_, HOLE_HALF_Z + WALL + L * s_
    # (name, inner-edge offset (y,z), slope dir in (x,y,z), outward normal, rot axis, rot angle, half-size)
    specs = [
        ("t", (0, HOLE_HALF_Z), (c, 0, -s_), (s_, 0, c), (0, 1, 0), th, (L / 2, wy, t / 2)),
        ("b", (0, -HOLE_HALF_Z), (c, 0, s_), (s_, 0, -c), (0, 1, 0), -th, (L / 2, wy, t / 2)),
        ("l", (-HOLE_HALF_Y, 0), (c, s_, 0), (s_, -c, 0), (0, 0, 1), th, (L / 2, t / 2, wz)),
        ("r", (HOLE_HALF_Y, 0), (c, -s_, 0), (s_, c, 0), (0, 0, 1), -th, (L / 2, t / 2, wz)),
    ]
    for k, (oy, oz), d, n, ax, ang, size in specs:
        inner = np.array([xf, y + oy, PANEL_Z + oz])
        mid = inner - np.array([c, 0, 0]) * 0  # placeholder to keep shapes explicit
        # outer edge is L along -d from the mouth edge (flare goes toward -x)
        mid = inner - np.array(d) * (L / 2)
        mid = mid + np.array(n) * (t / 2)
        rack.add_geom(name=f"port{i}_f{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=mid, size=size,
                      quat=_quat(ax, ang), material="socket", contype=1, conaffinity=3)


def _add_sockets(spec):
    rack = spec.body("rack")
    for i, y in enumerate(PORT_Y):
        cx = PANEL_FRONT_X - SOCKET_DEPTH / 2
        oz = HOLE_HALF_Z + WALL
        oy = HOLE_HALF_Y + WALL
        walls = {
            "l": ((cx, y - HOLE_HALF_Y - WALL / 2, PANEL_Z), (SOCKET_DEPTH / 2, WALL / 2, oz)),
            "r": ((cx, y + HOLE_HALF_Y + WALL / 2, PANEL_Z), (SOCKET_DEPTH / 2, WALL / 2, oz)),
            "t": ((cx, y, PANEL_Z + HOLE_HALF_Z + WALL / 2), (SOCKET_DEPTH / 2, oy, WALL / 2)),
            "b": ((cx, y, PANEL_Z - HOLE_HALF_Z - WALL / 2), (SOCKET_DEPTH / 2, oy, WALL / 2)),
        }
        for k, (pos, size) in walls.items():
            rack.add_geom(name=f"port{i}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                          material="socket", contype=1, conaffinity=3)
        _add_funnel(rack, i, y)
        rack.add_site(name=f"port{i}", pos=(PANEL_FRONT_X, y, PANEL_Z), size=(0.002,) * 3)


def _add_cable(spec):
    parent = spec.worldbody.add_body(name="cable0", pos=CABLE_ROOT)
    parent.add_freejoint(name="cable_free")
    body = parent
    for i in range(N_SEG):
        if i > 0:
            body = body.add_body(name=f"cable{i}", pos=(SEG_LEN, 0, 0))
            body.add_joint(name=f"cable_j{i}", type=mujoco.mjtJoint.mjJNT_BALL,
                           stiffness=0.012, damping=0.001, range=(0, 1.2), limited=True,
                           pos=(0, 0, 0))
        body.add_geom(name=f"cable_g{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=(0, 0, 0, SEG_LEN, 0, 0), size=(CABLE_R, 0, 0),
                      rgba=(0.08, 0.08, 0.08, 1), density=1200, friction=(0.3, 0.005, 1e-4),
                      contype=2, conaffinity=0)
    # plug on last segment: grip block then tip, both along +x
    bx = SEG_LEN + BLOCK_HALF[0]
    body.add_geom(name="plug_block", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(bx, 0, 0),
                  size=BLOCK_HALF, material="plug", density=1500, friction=(1.5, 0.02, 0.002), condim=6,
                  contype=2, conaffinity=1)
    tx = SEG_LEN + 2 * BLOCK_HALF[0] + TIP_HALF[0]
    body.add_geom(name="plug_tip", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(tx, 0, 0),
                  size=TIP_HALF, material="plug", density=1500, friction=(0.5, 0.005, 1e-4),
                  contype=2, conaffinity=1)
    body.add_site(name="plug_tip_site", pos=(SEG_LEN + 2 * BLOCK_HALF[0] + 2 * TIP_HALF[0], 0, 0),
                  size=(0.002,) * 3)
    body.add_site(name="plug_grip_site", pos=(bx + GRASP_OFFSET, 0, 0), size=(0.002,) * 3)


def _add_cameras(spec):
    # static target the fixed cameras look at (middle of table -> panel)
    spec.worldbody.add_body(name="look", pos=(0.5, 0.0, 0.45))
    tb = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    spec.worldbody.add_camera(name="front", pos=(0.45, -1.3, 1.0), mode=tb, targetbody="look", fovy=60)
    spec.worldbody.add_camera(name="closeup", pos=(0.2, -0.75, 0.85), mode=tb, targetbody="look", fovy=50)


def _add_hand_sites(spec):
    hand = spec.body("panda_hand")
    hand.add_site(name="tcp", pos=(0, 0, 0.1034), size=(0.003,) * 3)
    # wrist camera: looks along the hand +z (down at home), offset to sit beside the fingers
    hand.add_camera(name="wrist", pos=(0.07, 0, 0.04), xyaxes=(1, 0, 0, 0, -1, 0), fovy=75)


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(HERE / "scene.xml"))
    _add_sockets(spec)
    _add_cable(spec)
    _add_cameras(spec)

    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    # Menagerie panda has its own floor-less model; attach at world origin
    frame = spec.worldbody.add_frame(pos=(0, 0, 0))
    spec.attach(panda, frame=frame, prefix="panda_")
    _add_hand_sites(spec)
    # port latch: weld between the plug body and the world, OFF until a controller seats the plug
    eq = spec.add_equality(name="latch", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1=f"cable{N_SEG - 1}", name2="world")
    eq.active = False
    eq.solref = (0.01, 1.0)
    # Menagerie's gripper pinches with only ~2 N; scale gain+bias together (keeps the open
    # position) so a plug dragged by a cable doesn't rotate in the fingers.
    act = spec.actuator("panda_actuator8")
    act.gainprm[0] *= GRIP_FORCE_SCALE
    act.biasprm[1] *= GRIP_FORCE_SCALE
    act.biasprm[2] *= GRIP_FORCE_SCALE
    return spec


def build_model():
    spec = build_spec()
    return spec.compile(), spec


if __name__ == "__main__":
    m, _ = build_model()
    print("ok: nbody", m.nbody, "njnt", m.njnt, "nu", m.nu, "ncam", m.ncam)
