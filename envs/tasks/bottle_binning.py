"""Bottle binning: clear a conveyor of PET bottles into a crate.

Four 500 ml-style bottles (clear/blue, cola-brown, juice-orange, green) lie or stand on a static
conveyor belt (rubber belt between rollers and side rails). A ventilated plastic crate stands beside it.
Bottles are built from a cylindrical body, a sloped shoulder, a neck ring and a screw cap, with a
realistic empty-bottle mass (~28 g) and a bit of liquid mass in the bottom.
Success: all bottles rest inside the crate (below the rim) and none is on the belt.
"""

import mujoco
import numpy as np

from ..common.arms import down_rot
from ..common.base_env import TaskEnv
from ..common.geom import CYL, box, cyl, qmul, qx, qz

BELT_C, BELT_HALF = np.array([0.50, -0.20]), np.array([0.16, 0.11])
CRATE_C, CRATE_IN = np.array([0.50, 0.25]), np.array([0.15, 0.10])  # centre, inner half-extents (x, y)
CRATE_H = 0.13
N_BOTTLES = 4
R_BODY, H_BODY = 0.0315, 0.125
BOTTLE_STYLES = [("clear_blue", "blue"), ("glass", "blue"), ("amber", "red"), ("green", "orange")]
BOTTLE_LEN = H_BODY + 0.045 + 0.025


def _bottle(w, name, liquid, cap):
    b = w.add_body(name=name, pos=(0.3, 0.0, 0.3))
    b.add_freejoint(name=f"{name}_free")
    kw = {"friction": (0.6, 0.01, 0.001)}
    b.add_geom(
        name=f"{name}_body",
        type=CYL,
        pos=(0, 0, H_BODY / 2),
        size=(R_BODY, H_BODY / 2, 0),
        material=liquid,
        density=70,
        **kw,
    )
    b.add_geom(
        name=f"{name}_liquid",
        type=CYL,
        pos=(0, 0, 0.02),
        size=(R_BODY - 0.001, 0.02, 0),
        material=liquid,
        density=1000,
        contype=0,
        conaffinity=0,
    )
    b.add_geom(
        name=f"{name}_shoulder",
        type=CYL,
        pos=(0, 0, H_BODY + 0.012),
        size=(0.0255, 0.012, 0),
        material=liquid,
        density=70,
        **kw,
    )
    b.add_geom(
        name=f"{name}_shoulder2",
        type=CYL,
        pos=(0, 0, H_BODY + 0.030),
        size=(0.0165, 0.006, 0),
        material=liquid,
        density=70,
        **kw,
    )
    b.add_geom(
        name=f"{name}_neck",
        type=CYL,
        pos=(0, 0, H_BODY + 0.045 + 0.005),
        size=(0.0125, 0.01, 0),
        material=liquid,
        density=70,
        **kw,
    )
    b.add_geom(
        name=f"{name}_ring",
        type=CYL,
        pos=(0, 0, H_BODY + 0.04),
        size=(0.0155, 0.0015, 0),
        material=liquid,
        density=70,
        **kw,
    )
    b.add_geom(
        name=f"{name}_cap",
        type=CYL,
        pos=(0, 0, H_BODY + 0.045 + 0.0165),
        size=(0.0150, 0.0105, 0),
        material=cap,
        density=300,
        **kw,
    )
    b.add_geom(
        name=f"{name}_label",
        type=CYL,
        pos=(0, 0, 0.065),
        size=(R_BODY + 0.0004, 0.018, 0),
        material="white",
        contype=0,
        conaffinity=0,
    )
    return b


class BottleBinning(TaskEnv):
    name = "bottle_binning"
    max_seconds = 120.0
    success_hold = 0.5
    overview = ((0.50, -0.65, 0.60), (0.50, 0.03, 0.04))

    def build(self, spec):
        w = spec.worldbody
        bc = BELT_C
        # conveyor: belt, rails, rollers
        box(w, "belt", (*bc, 0.012), (BELT_HALF[0], BELT_HALF[1], 0.012), "rubber", friction=(0.9, 0.01, 0.001))
        box(
            w,
            "belt_frame",
            (*bc, 0.004),
            (BELT_HALF[0] + 0.01, BELT_HALF[1] + 0.012, 0.004),
            "dark_steel",
            contype=0,
            conaffinity=0,
        )
        for s in (-1, 1):
            box(
                w,
                f"belt_rail{s}",
                (bc[0], bc[1] + s * (BELT_HALF[1] + 0.007), 0.032),
                (BELT_HALF[0], 0.0035, 0.02),
                "alu",
            )
            cyl(
                w,
                f"belt_roller{s}",
                (bc[0] + s * BELT_HALF[0], bc[1], 0.012),
                0.012,
                BELT_HALF[1],
                "steel",
                quat=qx(np.pi / 2),
                contype=0,
                conaffinity=0,
            )
        for k in range(6):  # belt cleats (visual)
            box(
                w,
                f"cleat{k}",
                (bc[0] - 0.125 + 0.05 * k, bc[1], 0.0245),
                (0.002, BELT_HALF[1] - 0.003, 0.0005),
                "black",
                visual=True,
            )
        # crate: base slats + four walls with vent slots
        cc = CRATE_C
        hx, hy = CRATE_IN
        box(w, "crate_base", (*cc, 0.006), (hx + 0.006, hy + 0.006, 0.006), "crate_blue", friction=(0.6, 0.01, 0.001))
        for s in (-1, 1):
            box(
                w,
                f"crate_wx{s}",
                (cc[0] + s * (hx + 0.003), cc[1], CRATE_H / 2),
                (0.003, hy + 0.006, CRATE_H / 2),
                "crate_blue",
            )
            box(
                w,
                f"crate_wy{s}",
                (cc[0], cc[1] + s * (hy + 0.003), CRATE_H / 2),
                (hx + 0.006, 0.003, CRATE_H / 2),
                "crate_blue",
            )
            for k in range(5):
                box(
                    w,
                    f"crate_vent{s}_{k}",
                    (cc[0] - 0.10 + 0.05 * k, cc[1] + s * (hy + 0.0062), 0.07),
                    (0.016, 0.0003, 0.03),
                    "black",
                    visual=True,
                )
            box(
                w,
                f"crate_hand{s}",
                (cc[0] + s * (hx + 0.0062), cc[1], 0.10),
                (0.0003, 0.04, 0.012),
                "black",
                visual=True,
            )
        self.bottle_names = [f"bottle{i}" for i in range(N_BOTTLES)]
        for i, n in enumerate(self.bottle_names):
            liquid, cap = BOTTLE_STYLES[i % len(BOTTLE_STYLES)]
            _bottle(w, n, liquid, cap)

    def bind(self):
        m = self.m
        self.bid = [m.body(n).id for n in self.bottle_names]
        self.bq = [m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.bottle_names]

    def randomize_state(self, rng):
        """Two lanes along the belt. A lane holds either one lying bottle (aligned with the belt, +-0.2 rad) plus one
        standing bottle, or two standing bottles, so nothing overlaps or leans on a neighbour at reset."""
        d = self.d
        lane_y = [BELT_C[1] - 0.05, BELT_C[1] + 0.05]
        spots = []  # (x, y, kind)
        for li, y in enumerate(lane_y):
            lying = (rng.random() < 0.5) if self.randomize else (li == 0)
            side = (1 if rng.random() < 0.5 else -1) if self.randomize else 1
            if lying:
                spots.append((BELT_C[0] - side * 0.055, y, "lying"))
                spots.append((BELT_C[0] + side * 0.10, y, "standing"))
            else:
                spots.append((BELT_C[0] - 0.08, y, "standing"))
                spots.append((BELT_C[0] + 0.08, y, "standing"))
        order = rng.permutation(len(spots)) if self.randomize else np.arange(len(spots))
        for a, k in zip(self.bq, order):
            x, y, kind = spots[k]
            j = rng.uniform(-0.006, 0.006, 2) if self.randomize else np.zeros(2)
            if kind == "standing":
                yaw = rng.uniform(-np.pi, np.pi) if self.randomize else 0.0
                d.qpos[a : a + 7] = [x + j[0], y + j[1], 0.024 + 0.002, *qz(yaw)]
            else:  # lying, axis along the belt (x)
                yaw = rng.uniform(-0.05, 0.05) if self.randomize else 0.0
                q = qmul(qz(yaw + np.pi / 2), qx(np.pi / 2))
                # the body origin is at the base end and the axis points along +x: centre the bottle on the spot
                d.qpos[a : a + 7] = [x - BOTTLE_LEN / 2 + j[0], y + j[1], 0.024 + R_BODY + 0.002, *q]

    def _in_crate(self, p):
        return abs(p[0] - CRATE_C[0]) < CRATE_IN[0] and abs(p[1] - CRATE_C[1]) < CRATE_IN[1] and p[2] < CRATE_H

    def evaluate(self):
        d = self.d
        n_in = sum(self._in_crate(d.xpos[b]) for b in self.bid)
        # a bottle counts only when it lies fully inside: centre inside and not rim-balanced (z below rim)
        return {"success": n_in == N_BOTTLES, "progress": n_in / N_BOTTLES, "bottles_in_crate": n_in}

    def oracle_solve(self):
        d = self.d
        for k, a in enumerate(self.bq):
            d.qpos[a : a + 7] = [
                CRATE_C[0] - 0.10 + 0.07 * (k % 4),
                CRATE_C[1] + (0.04 if k % 2 else -0.04),
                0.03,
                *qx(np.pi / 2),
            ]
        mujoco.mj_forward(self.m, d)

    def reach_targets(self):
        return [
            ("belt bottle", 0, np.array([*BELT_C, 0.085]), down_rot(0.0)),
            ("belt bottle far", 0, np.array([BELT_C[0] + 0.11, BELT_C[1] + 0.045, 0.085]), down_rot(0.0)),
            ("above crate", 0, np.array([*CRATE_C, 0.22]), down_rot(0.0)),
        ]

    @property
    def instruction(self):
        return "put all the bottles from the conveyor into the crate"
