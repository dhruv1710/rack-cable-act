"""Place an object relative to another object ("put the can to the left of the mug").

Six household objects stand on the bench: mug (with handle), bowl, cereal box, soda can, apple, toy block.
Each episode samples a (target, reference, relation) triple; relations are given from the *robot's* point of
view: left (+y), right (-y), in front of (toward the robot, -x), behind (+x), on top of.
Success: the target rests (not held, not moving) in the commanded relation to the reference: for planar
relations, offset along the relation axis is between 0.7x and ~1x+9 cm of the summed radii with small lateral
error; for "on top of", it sits on the reference's flat top.
"""

import mujoco
import numpy as np

from ..common.arms import down_rot
from ..common.base_env import TaskEnv
from ..common.geom import CAP, CYL, SPH, box, hollow_cylinder, qz

# name: (radius_for_relations, full_height, flat_top, graspable stand height of body origin above table)
CATALOG = {
    "mug": (0.037, 0.095, True),
    "bowl": (0.062, 0.052, False),
    "cereal box": (0.040, 0.120, True),
    "soda can": (0.033, 0.122, True),
    "apple": (0.036, 0.072, False),
    "toy block": (0.027, 0.054, True),
}
REL = {
    "to the left of": (0.0, 1.0),
    "to the right of": (0.0, -1.0),
    "in front of": (-1.0, 0.0),
    "behind": (1.0, 0.0),
    "on top of": None,
}
ZONE_X, ZONE_Y = (0.31, 0.62), (-0.27, 0.27)


def _build_object(w, name):
    n = name.replace(" ", "_")
    r, h, _ = CATALOG[name]
    b = w.add_body(name=n, pos=(0.5, 0.0, h / 2 + 0.002))
    b.add_freejoint(name=f"{n}_free")
    kw = {"friction": (0.8, 0.01, 0.001)}
    if name == "mug":
        b.add_geom(name=f"{n}_body", type=CYL, size=(r, h / 2, 0), material="blue", density=450, **kw)
        b.add_geom(
            name=f"{n}_coffee",
            type=CYL,
            pos=(0, 0, h / 2 - 0.006),
            size=(r - 0.004, 0.0006, 0),
            material="wood",
            contype=0,
            conaffinity=0,
        )
        b.add_geom(
            name=f"{n}_h1",
            type=CAP,
            fromto=(r + 0.011, 0, 0.026, r + 0.011, 0, -0.026),
            size=(0.0055, 0, 0),
            material="blue",
            density=450,
            **kw,
        )
        b.add_geom(
            name=f"{n}_h2",
            type=CAP,
            fromto=(r - 0.002, 0, 0.030, r + 0.011, 0, 0.026),
            size=(0.0055, 0, 0),
            material="blue",
            density=450,
            **kw,
        )
        b.add_geom(
            name=f"{n}_h3",
            type=CAP,
            fromto=(r - 0.002, 0, -0.030, r + 0.011, 0, -0.026),
            size=(0.0055, 0, 0),
            material="blue",
            density=450,
            **kw,
        )
    elif name == "bowl":
        hollow_cylinder(
            b,
            n,
            (0, 0, -h / 2),
            0.045,
            h - 0.004,
            wall=0.004,
            n=14,
            material="ceramic",
            flare=0.45,
            base_thick=0.004,
            density=700,
            **kw,
        )
    elif name == "cereal box":
        b.add_geom(
            name=f"{n}_body",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.040, 0.018, h / 2),
            material="orange",
            density=140,
            **kw,
        )
        b.add_geom(
            name=f"{n}_art",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(0, -0.0182, 0.0),
            size=(0.032, 0.0004, 0.045),
            material="yellow",
            contype=0,
            conaffinity=0,
        )
    elif name == "soda can":
        b.add_geom(name=f"{n}_body", type=CYL, size=(r, h / 2 - 0.004, 0), material="red", density=600, **kw)
        b.add_geom(
            name=f"{n}_top",
            type=CYL,
            pos=(0, 0, h / 2 - 0.002),
            size=(r - 0.004, 0.002, 0),
            material="alu",
            density=600,
            **kw,
        )
        b.add_geom(name=f"{n}_band", type=CYL, size=(r + 0.0003, 0.012, 0), material="white", contype=0, conaffinity=0)
    elif name == "apple":
        b.add_geom(
            name=f"{n}_body",
            type=SPH,
            size=(h / 2, 0, 0),
            material="green",
            density=800,
            condim=6,
            friction=(0.8, 0.01, 0.006),
        )
        b.add_geom(
            name=f"{n}_stem",
            type=CAP,
            fromto=(0, 0, h / 2 - 0.002, 0.003, 0, h / 2 + 0.012),
            size=(0.0015, 0, 0),
            material="wood",
            contype=0,
            conaffinity=0,
        )
    else:
        b.add_geom(
            name=f"{n}_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=(r, r, h / 2), material="purple", density=500, **kw
        )
        for k, s in enumerate((-1, 1)):
            b.add_geom(
                name=f"{n}_dot{k}",
                type=CYL,
                pos=(0, s * 0.009, h / 2 + 0.0003),
                size=(0.006, 0.0004, 0),
                material="white",
                contype=0,
                conaffinity=0,
            )
    return b


class RelativePlacement(TaskEnv):
    name = "relative_placement"
    max_seconds = 60.0
    success_hold = 0.5
    overview = ((0.50, -0.62, 0.55), (0.48, 0.0, 0.03))

    def build(self, spec):
        w = spec.worldbody
        box(w, "mat", (0.46, 0.0, 0.0005), (0.20, 0.32, 0.0005), "mat", contype=0, conaffinity=0)
        for name in CATALOG:
            _build_object(w, name)

    def bind(self):
        m = self.m
        self.names = list(CATALOG)
        self.bid = {n: m.body(n.replace(" ", "_")).id for n in self.names}
        self.qadr = {n: m.jnt_qposadr[m.joint(n.replace(" ", "_") + "_free").id] for n in self.names}
        self.dof = {n: m.jnt_dofadr[m.joint(n.replace(" ", "_") + "_free").id] for n in self.names}
        self.geoms = {n: self.body_geoms(n.replace(" ", "_")) for n in self.names}
        self.target, self.ref, self.rel = "soda can", "mug", "to the left of"

    # ---- sampling
    def _relation_holds(self, pos_t, pos_r, tgt, ref, rel):
        rt, ht, _ = CATALOG[tgt]
        rr, hr, _flat = CATALOG[ref]
        d = pos_t - pos_r
        if rel == "on top of":
            top = pos_r[2] + hr / 2
            return np.hypot(d[0], d[1]) < 0.02 + 0.25 * rr and top + ht / 2 - 0.012 < pos_t[2] < top + ht / 2 + 0.025
        ax = np.array(REL[rel])
        D0 = rt + rr
        along = d[0] * ax[0] + d[1] * ax[1]
        lat = abs(-d[0] * ax[1] + d[1] * ax[0])
        low = pos_t[2] < ht / 2 + 0.02
        return 0.7 * D0 < along < D0 + 0.09 and lat < 0.5 * D0 + 0.02 and low

    def randomize_state(self, rng):
        d = self.d
        names = self.names
        for _ in range(500):
            pts = [np.array([rng.uniform(*ZONE_X), rng.uniform(*ZONE_Y)]) for _ in names]
            if all(np.linalg.norm(pts[i] - pts[j]) > 0.125 for i in range(len(pts)) for j in range(i)):
                break
        if not self.randomize:
            pts = [np.array([0.40 + 0.11 * (i % 3), -0.18 + 0.18 * (i // 3) * 2 - 0.0 + 0.0]) for i in range(6)]
            pts = [np.array([0.36 + 0.13 * (i % 3), -0.15 + 0.3 * (i // 3)]) for i in range(6)]
        rel = list(REL)[int(rng.integers(0, 5))] if self.randomize else "to the left of"
        flat_names = [n for n in names if CATALOG[n][2]]
        if rel == "on top of":
            # only physically stable stacks: flat-topped reference at least as wide as the target
            pairs = [
                (t, r)
                for t in flat_names
                for r in flat_names
                if t != r and t != "cereal box" and CATALOG[t][0] <= CATALOG[r][0] + 0.001
            ]
            tgt, ref = pairs[int(rng.integers(0, len(pairs)))] if self.randomize else ("toy block", "cereal box")
        else:
            ref = names[int(rng.integers(0, len(names)))] if self.randomize else "mug"
            cands = [n for n in names if n != ref]
            tgt = cands[int(rng.integers(0, len(cands)))] if self.randomize else cands[0]
        self.target, self.ref, self.rel = tgt, ref, rel
        for n, p in zip(names, pts):
            q = qz(rng.uniform(-np.pi, np.pi)) if self.randomize else np.array([1.0, 0, 0, 0])
            a = self.qadr[n]
            d.qpos[a : a + 7] = [p[0], p[1], CATALOG[n][1] / 2 + 0.003, *q]

    def _held(self, n):
        return any(self.contacts_between(self.geoms[n], a.finger_geoms) for a in self.arms)

    def evaluate(self):
        d = self.d
        pt, pr = d.xpos[self.bid[self.target]], d.xpos[self.bid[self.ref]]
        ok = self._relation_holds(pt, pr, self.target, self.ref, self.rel)
        v = np.linalg.norm(d.qvel[self.dof[self.target] : self.dof[self.target] + 6])
        success = ok and v < 0.05 and not self._held(self.target)
        dist = np.linalg.norm(pt[:2] - pr[:2])
        prog = 1.0 if ok else float(np.clip(1 - (dist - 0.08) / 0.4, 0, 0.9))
        return {
            "success": bool(success),
            "progress": prog,
            "target": self.target,
            "reference": self.ref,
            "relation": self.rel,
        }

    def oracle_solve(self):
        d = self.d
        rt, ht, _ = CATALOG[self.target]
        rr, hr, _ = CATALOG[self.ref]
        pr = d.xpos[self.bid[self.ref]].copy()
        if self.rel == "on top of":
            p = np.array([pr[0], pr[1], pr[2] + hr / 2 + ht / 2 + 0.002])
        else:
            ax = np.array(REL[self.rel])
            p = np.array([pr[0] + ax[0] * (rt + rr + 0.02), pr[1] + ax[1] * (rt + rr + 0.02), ht / 2 + 0.003])
        a = self.qadr[self.target]
        d.qpos[a : a + 7] = [*p, 1, 0, 0, 0]
        d.qvel[self.dof[self.target] : self.dof[self.target] + 6] = 0
        mujoco.mj_forward(self.m, d)

    def reach_targets(self):
        return [
            ("near-left object", 0, np.array([0.34, 0.25, 0.07]), down_rot(0.0)),
            ("far-right object", 0, np.array([0.60, -0.25, 0.07]), down_rot(0.0)),
            ("stack height", 0, np.array([0.45, 0.0, 0.22]), down_rot(0.0)),
        ]

    @property
    def instruction(self):
        return f"place the {self.target} {self.rel} the {self.ref}" + (
            " (left and right are from the robot's point of view)" if self.rel != "on top of" else ""
        )
