"""Pouring beads: pour the beads from a cup into a bowl, then put the cup back down.

A paper-cup-sized vessel (inner 64 mm, 85 mm tall) holds 100 glass beads (7 mm). A flared bowl stands
~25 cm away. The cup is built from 16 thin wall panels so beads really spill over the rim when tipped.
Success: >=85 % of the beads are in the bowl, <=8 % on the table, and the cup is back upright on the
table, released.
"""

import mujoco
import numpy as np

from ..common.arms import down_rot
from ..common.base_env import TaskEnv
from ..common.geom import hollow_cylinder

CUP_C, CUP_R, CUP_H = np.array([0.36, -0.20]), 0.032, 0.085
BOWL_C, BOWL_R, BOWL_H, BOWL_FLARE = (
    np.array([0.47, 0.02]),
    0.050,
    0.050,
    0.40,
)  # a forward ~2 rad tip is only reachable for TCP x <= 0.48 m
BEAD_R, N_BEADS = 0.0035, 100
BASE_T = 0.004
CUP_WALL = 0.005  # ceramic/plastic cup wall (outer diameter 74 mm < 80 mm gripper opening)
STIFF, STIFF_IMP = (0.004, 1.0), (0.99, 0.999, 0.0005, 0.5, 2)  # thin walls must not let finger pads sink through
COLORS = ["red", "blue", "yellow", "green", "orange", "white", "purple", "hot_pink"]


def hex_points(radius, spacing):
    pts = []
    for iy in range(-12, 13):
        for ix in range(-12, 13):
            x = ix * spacing + (spacing / 2 if iy % 2 else 0)
            y = iy * spacing * 0.866
            if np.hypot(x, y) < radius:
                pts.append((x, y))
    return pts


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class PouringBeads(TaskEnv):
    name = "pouring_beads"
    max_seconds = 90.0
    success_hold = 0.6
    settle_seconds = 1.4
    overview = ((0.48, -0.58, 0.45), (0.48, 0.0, 0.04))

    def build(self, spec):
        w = spec.worldbody
        cup = w.add_body(name="cup", pos=(*CUP_C, 0.0005))
        cup.add_freejoint(name="cup_free")
        hollow_cylinder(
            cup,
            "cup",
            (0, 0, 0),
            CUP_R,
            CUP_H,
            wall=CUP_WALL,
            n=16,
            material="white",
            base_thick=BASE_T,
            density=250,
            condim=4,
            friction=(0.3, 0.02, 0.001),
            solref=STIFF,
            solimp=STIFF_IMP,
        )  # low wall friction so beads slide out; the pads supply the grip (contact friction = max of the two)
        for k in range(3):  # printed stripe (visual)
            cup.add_geom(
                name=f"cup_stripe{k}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=(0, 0, 0.02 + 0.02 * k),
                size=(CUP_R + CUP_WALL + 0.0003, 0.002, 0),
                material="red",
                contype=0,
                conaffinity=0,
            )
        bowl = w.add_body(name="bowl", pos=(*BOWL_C, 0.0))
        hollow_cylinder(
            bowl,
            "bowl",
            (0, 0, 0),
            BOWL_R,
            BOWL_H,
            wall=0.004,
            n=20,
            material="crate_blue",
            flare=BOWL_FLARE,
            base_thick=BASE_T,
            friction=(0.6, 0.01, 0.001),
            solref=STIFF,
            solimp=STIFF_IMP,
        )
        self.bead_names = []
        for i in range(N_BEADS):
            b = w.add_body(name=f"bead{i}", pos=(0.3, 0.3, 0.3))
            b.add_freejoint(name=f"bead{i}_free")
            b.add_geom(
                name=f"bead{i}_g",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(BEAD_R, 0, 0),
                material=COLORS[i % len(COLORS)],
                density=2500,
                friction=(0.35, 0.002, 0.0001),
                solref=(0.004, 1.0),
            )
            self.bead_names.append(f"bead{i}")

    def bind(self):
        m = self.m
        self.cup = m.body("cup").id
        self.cup_q = m.jnt_qposadr[m.joint("cup_free").id]
        self.cup_geoms = self.body_geoms("cup")
        self.bead_bid = np.array([m.body(n).id for n in self.bead_names])
        self.bead_q = [m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.bead_names]

    def randomize_state(self, rng):
        d = self.d
        jc = rng.uniform(-0.02, 0.02, 2) if self.randomize else np.zeros(2)
        jb = rng.uniform(-0.015, 0.015, 2) if self.randomize else np.zeros(2)
        self.cup_c = CUP_C + jc
        self.bowl_c = BOWL_C + jb
        self.m.body_pos[self.m.body("bowl").id] = (*self.bowl_c, 0.0)
        a = self.cup_q
        d.qpos[a : a + 7] = [*self.cup_c, 0.0005, 1, 0, 0, 0]
        # beads: hex-packed layers inside the cup
        sp = 2 * BEAD_R * 1.04
        pts = hex_points(CUP_R - BEAD_R - 0.002, sp)
        self._place_beads(pts, self.cup_c, sp, rng)

    def _place_beads(self, pts, center, sp, rng, z0=None):
        d = self.d
        z0 = BASE_T + BEAD_R + 0.0008 if z0 is None else z0
        n = layer = 0
        while n < N_BEADS:
            for x, y in pts:
                if n >= N_BEADS:
                    break
                z = z0 + layer * sp * 0.82
                jit = rng.uniform(-0.0004, 0.0004, 2)
                d.qpos[self.bead_q[n] : self.bead_q[n] + 7] = [
                    center[0] + x + jit[0],
                    center[1] + y + jit[1],
                    z,
                    1,
                    0,
                    0,
                    0,
                ]
                n += 1
            layer += 1

    def beads_in_cup(self):
        """Beads currently inside the cup volume (checked in the cup's own frame, so valid while tipped)."""
        d = self.d
        R = d.xmat[self.cup].reshape(3, 3)
        loc = (d.xpos[self.bead_bid] - d.xpos[self.cup]) @ R  # beads in cup frame
        rad = np.hypot(loc[:, 0], loc[:, 1])
        return int(((rad < CUP_R + 0.002) & (loc[:, 2] > 0.0) & (loc[:, 2] < CUP_H + 0.01)).sum())

    def _counts(self):
        p = self.d.xpos[self.bead_bid]
        db = np.hypot(p[:, 0] - self.bowl_c[0], p[:, 1] - self.bowl_c[1])
        in_bowl = (
            (db < BOWL_R + BOWL_H * np.tan(BOWL_FLARE) + 0.003)
            & (p[:, 2] > BASE_T + BEAD_R - 0.0015)
            & (p[:, 2] < BOWL_H + 0.02)
        )
        on_table = (p[:, 2] < BEAD_R + 0.0015) & ~in_bowl
        return int(in_bowl.sum()), int(on_table.sum())

    def evaluate(self):
        n_in, n_spill = self._counts()
        d = self.d
        cz = d.xmat[self.cup].reshape(3, 3)[2, 2]
        cup_down = cz > 0.95 and d.xpos[self.cup][2] < 0.01
        held = any(self.contacts_between(self.cup_geoms, a.finger_geoms) for a in self.arms)
        success = n_in >= 0.85 * N_BEADS and n_spill <= 0.08 * N_BEADS and cup_down and not held
        prog = 0.8 * n_in / N_BEADS - 0.4 * n_spill / N_BEADS + 0.2 * (cup_down and n_in > 0.5 * N_BEADS)
        return {
            "success": bool(success),
            "progress": float(np.clip(prog, 0, 1)),
            "beads_in_bowl": n_in,
            "beads_spilled": n_spill,
            "beads_in_cup": self.beads_in_cup(),
        }

    def oracle_solve(self):
        sp = 2 * BEAD_R * 1.05
        pts = hex_points(0.040, sp)
        self._place_beads(pts, self.bowl_c, sp, np.random.default_rng(0), z0=BASE_T + BEAD_R + 0.001)
        mujoco.mj_forward(self.m, self.d)

    def reach_targets(self):
        R = down_rot(0.0)  # fingers close along y; tip the cup about y to pour toward +x
        c = CUP_C
        pour = rot_y(1.9) @ R
        return [
            ("grasp cup", 0, np.array([*c, 0.05]), R),
            ("lift cup", 0, np.array([*c, 0.22]), R),
            ("pour over bowl", 0, np.array([BOWL_C[0] - 0.03, BOWL_C[1], 0.15]), pour),
        ]

    @property
    def instruction(self):
        return "pour the beads from the cup into the bowl, then put the cup back on the table"
