"""Parts sorting: sort loose hardware into the labelled compartments of a parts organiser.

Nine parts (3 hex bolts, 3 hex nuts, 3 standoffs) are scattered in a loose-parts zone. A 3-compartment
organiser has a sample part of each type fixed to the back wall of its compartment as the label; the
type->compartment mapping is randomised every episode, so the policy must read the labels.
Success: every part rests inside the compartment whose label matches its type.
"""

import mujoco
import numpy as np

from ..common.arms import down_rot
from ..common.base_env import TaskEnv
from ..common.geom import CYL, add_hex_mesh, box, qmul, qx, qz

TYPES = ("bolt", "nut", "standoff")
ORG_C = np.array([0.62, -0.17])  # organiser centre (x, y)
ORG_HX, COMP_W, WALL_H = 0.065, 0.105, 0.032
ZONE_C, ZONE_HALF = np.array([0.44, 0.22]), np.array([0.09, 0.10])
PER_TYPE = 3
MAT = {"bolt": "dark_steel", "nut": "brass", "standoff": "blue"}


class PartsSorting(TaskEnv):
    name = "parts_sorting"
    max_seconds = 180.0
    success_hold = 0.5
    overview = ((0.50, -0.55, 0.55), (0.52, 0.03, 0.0))

    def build(self, spec):
        w = spec.worldbody
        add_hex_mesh(spec, "hex_head", 0.017, 0.007)  # M10 bolt head, 17 mm A/F
        add_hex_mesh(spec, "hex_nut", 0.017, 0.008)  # M10 nut
        # organiser: floor, outer walls, two dividers, labels
        org = w.add_body(name="organiser", pos=(ORG_C[0], ORG_C[1], 0.0))
        box(org, "org_floor", (0, 0, 0.002), (ORG_HX, 1.5 * COMP_W, 0.002), "plastic_grey", friction=(0.6, 0.01, 0.001))
        for s in (-1, 1):
            box(
                org,
                f"org_side{s}",
                (0, s * (1.5 * COMP_W + 0.0015), WALL_H / 2),
                (ORG_HX, 0.0015, WALL_H / 2),
                "plastic_grey",
            )
        box(
            org,
            "org_back",
            (ORG_HX + 0.0015, 0, WALL_H / 2),
            (0.0015, 1.5 * COMP_W + 0.003, WALL_H / 2),
            "plastic_grey",
        )
        box(
            org,
            "org_front",
            (-ORG_HX - 0.0015, 0, WALL_H / 2),
            (0.0015, 1.5 * COMP_W + 0.003, WALL_H / 2),
            "plastic_grey",
        )
        for k in (-1, 1):
            box(org, f"org_div{k}", (0, k * COMP_W / 2, WALL_H / 2), (ORG_HX, 0.0015, WALL_H / 2), "plastic_grey")
        # label samples: one visual body per (compartment, type); only the matching one is shown each episode
        self.label_bodies = {}
        for c in range(3):
            for t in TYPES:
                lb = w.add_body(name=f"label{c}_{t}", pos=(0, 0, -3.0))
                self._add_part_geoms(spec, lb, t, visual=True, suffix=f"lab{c}_{t}")
                self.label_bodies[(c, t)] = lb.name
        # parts
        self.part_names, self.part_type = [], {}
        for t in TYPES:
            for k in range(PER_TYPE):
                n = f"{t}{k}"
                b = w.add_body(name=n, pos=(0.3, 0.0, 0.1))
                b.add_freejoint(name=f"{n}_free")
                self._add_part_geoms(spec, b, t, visual=False, suffix=n)
                self.part_names.append(n)
                self.part_type[n] = t

    @staticmethod
    def _add_part_geoms(spec, b, t, visual, suffix):
        kw = {"contype": 0, "conaffinity": 0} if visual else {"friction": (0.9, 0.01, 0.001)}
        if t == "bolt":  # axis along body z: shank below the head
            b.add_geom(
                name=f"{suffix}_shank",
                type=CYL,
                pos=(0, 0, -0.0255),
                size=(0.005, 0.0225, 0),
                material="dark_steel",
                density=7800,
                **kw,
            )
            b.add_geom(
                name=f"{suffix}_head",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="hex_head",
                pos=(0, 0, 0.0035),
                material="dark_steel",
                density=7800,
                **kw,
            )
        elif t == "nut":
            b.add_geom(
                name=f"{suffix}_nut",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="hex_nut",
                material="brass",
                density=6000,
                **kw,
            )
        else:  # standoff: 12 mm round, 25 mm long
            b.add_geom(name=f"{suffix}_so", type=CYL, size=(0.006, 0.0125, 0), material="blue", density=2700, **kw)

    def bind(self):
        m = self.m
        self.part_bid = {n: m.body(n).id for n in self.part_names}
        self.part_q = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.part_names}
        self.label_bid = {k: m.body(n).id for k, n in self.label_bodies.items()}
        self.mapping = list(TYPES)  # compartment c (0 = robot's left, +y) holds type mapping[c]

    def comp_center(self, c):
        return np.array([ORG_C[0], ORG_C[1] + (1 - c) * COMP_W])  # c=0 -> +y (left)

    def randomize_state(self, rng):
        m, d = self.m, self.d
        self.mapping = list(rng.permutation(TYPES)) if self.randomize else list(TYPES)
        # labels: a sample part of the right type stands on the compartment's back wall
        for (c, t), bid in self.label_bid.items():
            if self.mapping[c] == t:
                cc = self.comp_center(c)
                m.body_pos[bid] = (ORG_C[0] + ORG_HX + 0.0035, cc[1], WALL_H + 0.012)
                m.body_quat[bid] = qx(np.pi / 2)
            else:
                m.body_pos[bid] = (0, 0, -3.0)
        # parts scattered without overlap
        placed = []
        order = list(self.part_names)
        rng.shuffle(order)
        for n in order:
            for _ in range(200):
                p = ZONE_C + rng.uniform(-1, 1, 2) * ZONE_HALF
                if all(np.linalg.norm(p - q) > 0.045 for q in placed):
                    break
            placed.append(p)
            t = self.part_type[n]
            yaw = rng.uniform(-np.pi, np.pi)
            a = self.part_q[n]
            if t == "bolt":  # lying on a hex flat, shank horizontal
                q, z = qmul(qz(yaw), qx(np.pi / 2)), 0.0095
            elif t == "nut":
                q, z = qz(yaw), 0.0045
            elif rng.random() < 0.6:  # standoff lying
                q, z = qmul(qz(yaw), qx(np.pi / 2)), 0.0065
            else:  # standoff standing
                q, z = qz(yaw), 0.0135
            d.qpos[a : a + 7] = [p[0], p[1], z + 0.001, *q]

    def evaluate(self):
        d = self.d
        ok = 0
        for n in self.part_names:
            p = d.xpos[self.part_bid[n]]
            c = int(np.clip(1 - np.floor((p[1] - ORG_C[1]) / COMP_W + 0.5), -1, 3))
            inside = abs(p[0] - ORG_C[0]) < ORG_HX and abs(p[1] - ORG_C[1]) < 1.5 * COMP_W and p[2] < WALL_H
            if inside and 0 <= c <= 2 and self.mapping[c] == self.part_type[n]:
                ok += 1
        return {"success": ok == len(self.part_names), "progress": ok / len(self.part_names), "sorted": ok}

    def oracle_solve(self):
        d = self.d
        cnt = dict.fromkeys(range(3), 0)
        for n in self.part_names:
            c = self.mapping.index(self.part_type[n])
            cc = self.comp_center(c)
            k = cnt[c]
            cnt[c] += 1
            a = self.part_q[n]
            d.qpos[a : a + 7] = [cc[0] - 0.035 + 0.035 * k, cc[1], 0.02, 1, 0, 0, 0]
        mujoco.mj_forward(self.m, d)

    def reach_targets(self):
        return [
            ("loose-part zone", 0, np.array([*ZONE_C, 0.03]), down_rot(0.0)),
            ("compartment 0", 0, np.array([*self.comp_center(0), 0.07]), down_rot(0.0)),
            ("compartment 2", 0, np.array([*self.comp_center(2), 0.07]), down_rot(0.0)),
        ]

    @property
    def instruction(self):
        return "sort the parts into the compartments by type, matching the sample part on each compartment"
