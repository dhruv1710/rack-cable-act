"""Fuse-holder handling: replace the blown 10x38 mm fuse in a DIN-rail fuse holder.

A four-pole row of hinged-lid fuse holders sits on a DIN rail on a vertical mounting plate. One holder
(target) contains a blown fuse; the others hold good ones. Fresh fuses stand in a foam caddy.
Procedure: flip the target holder's lid open, pull the blown fuse (spring-clip retention ~6 N), put it
in the scrap bin, seat a fresh fuse in the cradle, and push the lid shut.
Success: a fresh fuse is latched in the target holder, its lid is closed, the blown fuse is out of every
holder, and no other holder was disturbed.
"""

import mujoco
import numpy as np

from ..common import latch
from ..common.arms import down_rot
from ..common.base_env import TaskEnv
from ..common.geom import CYL, box, cyl, hollow_cylinder

PLATE_X, X_F, X_B = 0.700, 0.6465, 0.6925  # plate front face, holder front, holder back
RAIL_Z, ZB, ZT = 0.17, 0.13, 0.21  # rail centre, holder bottom, holder top
PITCH = 0.034
YS = np.array([0.051, 0.017, -0.017, -0.051])  # holder 1 is on the robot's left (+y)
FUSE_X = 0.664
CRADLE_TOP = ZB + 0.006
SEAT_Z = CRADLE_TOP + 0.019  # fuse centre height when seated
FUSE_HALF = 0.019
RELEASE_N = 6.0
N_NEW = 2


def _fuse(w, name, blown):
    b = w.add_body(name=name, pos=(0.3, 0.0, 0.3))
    b.add_freejoint(name=f"{name}_free")
    mat = {"density": 2700, "friction": (1.0, 0.005, 1e-4)}
    b.add_geom(
        name=f"{name}_glass", type=CYL, size=(0.0049, 0.011, 0), material="glass" if not blown else "charcoal", **mat
    )
    for s in (-1, 1):
        b.add_geom(
            name=f"{name}_cap{s}", type=CYL, pos=(0, 0, s * 0.015), size=(0.0050, 0.004, 0), material="alu", **mat
        )
    # filament (visual): intact wire vs. gap with a dark smudge
    if blown:
        for s in (-1, 1):
            box(b, f"{name}_wire{s}", (0, 0, s * 0.0065), (0.0003, 0.0003, 0.0045), "dark_steel", visual=True)
        b.add_geom(
            name=f"{name}_smudge", type=CYL, size=(0.0051, 0.0035, 0), material="black", contype=0, conaffinity=0
        )
    else:
        box(b, f"{name}_wire", (0, 0, 0), (0.0003, 0.0003, 0.011), "copper", visual=True)
    return b


class FuseHolder(TaskEnv):
    name = "fuse_holder"
    max_seconds = 120.0
    success_hold = 0.5
    overview = ((0.30, -0.42, 0.34), (0.66, 0.0, 0.14))

    def build(self, spec):
        w = spec.worldbody
        box(w, "plate", (PLATE_X + 0.006, 0, 0.20), (0.006, 0.30, 0.20), "alu")
        box(w, "plate_stiffener", (PLATE_X + 0.013, 0, 0.10), (0.007, 0.30, 0.012), "steel", visual=True)
        box(w, "din_rail", (PLATE_X - 0.0035, 0, RAIL_Z), (0.0035, 0.28, 0.0175), "steel")
        box(w, "din_rail_slot", (PLATE_X - 0.0072, 0, RAIL_Z), (0.0005, 0.28, 0.0125), "black", visual=True)
        for i, yc in enumerate(YS):
            hw = 0.0085
            box(w, f"h{i}_back", (X_B - 0.002, yc, 0.17), (0.002, hw, 0.04), "plastic_grey")
            box(w, f"h{i}_floor", (0.5 * (X_F + X_B), yc, ZB + 0.003), ((X_B - X_F) / 2, hw, 0.003), "plastic_grey")
            for s in (-1, 1):
                box(
                    w,
                    f"h{i}_side{s}",
                    (0.5 * (X_F + X_B) + 0.004, yc + s * 0.0073, ZB + 0.006 + 0.007),
                    ((X_B - X_F) / 2 - 0.004, 0.00075, 0.007),
                    "plastic_grey",
                )
            box(
                w,
                f"h{i}_terminal",
                (X_B - 0.0135, yc, ZB + 0.03 + 0.012),
                (0.0095, hw - 0.001, 0.012),
                "dark_steel",
                visual=True,
            )
            for k in range(i + 1):  # pole number pips
                box(
                    w,
                    f"h{i}_pip{k}",
                    (X_B - 0.004, yc - 0.005 + k * 0.0033, ZT + 0.0005),
                    (0.0012, 0.0012, 0.0005),
                    "white",
                    visual=True,
                )
            lid = w.add_body(name=f"lid{i}", pos=(X_F - 0.002, yc, ZB))
            lid.add_joint(
                name=f"lid{i}_hinge",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=(0, -1, 0),
                range=(0, 1.75),
                damping=0.002,
                frictionloss=0.004,
            )
            box(lid, f"lid{i}_plate", (0, 0, 0.04), (0.002, 0.0078, 0.04), "orange", density=700)
            box(lid, f"lid{i}_tab", (-0.0085, 0, 0.0775), (0.0085, 0.007, 0.0015), "orange", density=700)
            box(lid, f"lid{i}_ridge", (-0.0155, 0, 0.0795), (0.0005, 0.007, 0.0035), "orange", density=700)
            cyl(
                lid,
                f"lid{i}_barrel",
                (0, 0, 0),
                0.0025,
                0.0078,
                "dark_steel",
                quat=np.array([0.7071, 0.7071, 0, 0]),
                visual=True,
            )
        # fuses: 0 = blown, 1..3 = good (seated), then N_NEW fresh in the caddy
        self.fuse_names = ["fuse_blown"] + [f"fuse_good{k}" for k in range(3)] + [f"fuse_new{k}" for k in range(N_NEW)]
        for n in self.fuse_names:
            _fuse(w, n, blown=(n == "fuse_blown"))
            latch.add_weld(spec, f"{n}_latch", n)
        # caddy with collars
        cad = w.add_body(name="caddy", pos=(0.50, -0.22, 0.0))
        box(cad, "caddy_foam", (0, 0, 0.0125), (0.06, 0.04, 0.0125), "foam")
        self.caddy_xy = [(0.50, -0.245 + 0.05 * k) for k in range(N_NEW)]
        for k, (cx, cy) in enumerate(self.caddy_xy):
            hollow_cylinder(
                cad,
                f"caddy_col{k}",
                (cx - 0.50, cy + 0.22, 0.025),
                0.0058,
                0.012,
                wall=0.003,
                n=8,
                material="foam",
                base=False,
            )
        # scrap bin
        bin_ = w.add_body(name="scrap_bin", pos=(0.50, 0.22, 0.0))
        box(bin_, "bin_base", (0, 0, 0.002), (0.07, 0.05, 0.002), "red")
        for s, (px, py, hx, hy) in enumerate(
            [(0.07, 0, 0.002, 0.05), (-0.07, 0, 0.002, 0.05), (0, 0.05, 0.07, 0.002), (0, -0.05, 0.07, 0.002)]
        ):
            box(bin_, f"bin_wall{s}", (px, py, 0.02), (hx, hy, 0.02), "red")

    # ---------------------------------------------------------------- bind
    def bind(self):
        m = self.m
        self.fuse_bid = {n: m.body(n).id for n in self.fuse_names}
        self.fuse_eq = {n: m.equality(f"{n}_latch").id for n in self.fuse_names}
        self.fuse_qadr = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.fuse_names}
        self.fuse_dof = {n: m.jnt_dofadr[m.joint(f"{n}_free").id] for n in self.fuse_names}
        self.latched_at = dict.fromkeys(self.fuse_names, -1.0)
        self.lid_q = [m.jnt_qposadr[m.joint(f"lid{i}_hinge").id] for i in range(4)]
        self.seat = np.array([[FUSE_X, y, SEAT_Z] for y in YS])
        self.armed = dict.fromkeys(self.fuse_names, True)
        self.target = 0
        self.holder_of = {}  # fuse name -> holder idx it is latched in (or None)
        self.new_in_target = None

    # ---------------------------------------------------------------- reset
    def randomize_state(self, rng):
        d = self.d
        self.target = int(rng.integers(0, 4)) if self.randomize else 1
        goods = iter(["fuse_good0", "fuse_good1", "fuse_good2"])
        self.holder_fuse = {}
        for i in range(4):
            n = "fuse_blown" if i == self.target else next(goods)
            self.holder_fuse[i] = n
            a = self.fuse_qadr[n]
            d.qpos[a : a + 7] = [*self.seat[i], 1, 0, 0, 0]
        order = rng.permutation(N_NEW) if self.randomize else np.arange(N_NEW)
        for k in range(N_NEW):
            a = self.fuse_qadr[f"fuse_new{k}"]
            cx, cy = self.caddy_xy[order[k]]
            d.qpos[a : a + 7] = [cx, cy, 0.025 + 0.019 + 0.001, 1, 0, 0, 0]
        self.armed = dict.fromkeys(self.fuse_names, False)
        self.holder_of = dict.fromkeys(self.fuse_names)
        for n in self.fuse_names:
            d.eq_active[self.fuse_eq[n]] = 0

    def after_settle(self):
        for i, n in self.holder_fuse.items():  # factory-installed fuses are clipped in
            latch.engage(self.m, self.d, self.fuse_eq[n], self.fuse_bid[n])
            self.holder_of[n] = i
            self.latched_at[n] = self.d.time

    # ---------------------------------------------------------------- latch logic
    def on_substep(self):
        d, m = self.d, self.m
        for n in self.fuse_names:
            eq, b = self.fuse_eq[n], self.fuse_bid[n]
            if d.eq_active[eq]:
                if d.time - self.latched_at[n] > 0.1 and latch.holding_force(m, d, eq) > RELEASE_N:
                    latch.release(d, eq)
                    self.holder_of[n] = None
                    self.armed[n] = False
                continue
            p = d.xpos[b]
            ds = np.linalg.norm(self.seat - p, axis=1)
            j = int(ds.argmin())
            if not self.armed[n]:
                if ds[j] > 0.012:
                    self.armed[n] = True
                continue
            axis_z = d.xmat[b].reshape(3, 3)[2, 2]
            occupied = any(h == j for h in self.holder_of.values())
            slow = np.linalg.norm(d.qvel[self.fuse_dof[n] : self.fuse_dof[n] + 3]) < 0.25
            if (
                slow
                and not occupied
                and abs(p[0] - FUSE_X) < 0.003
                and abs(p[1] - YS[j]) < 0.003
                and abs(p[2] - SEAT_Z) < 0.003
                and abs(axis_z) > 0.985
            ):
                latch.engage(m, d, eq, b)
                self.holder_of[n] = j
                self.latched_at[n] = d.time

    # ---------------------------------------------------------------- evaluation
    def evaluate(self):
        tgt = self.target
        new_in = [n for n in self.fuse_names if n.startswith("fuse_new") and self.holder_of.get(n) == tgt]
        lid = float(self.d.qpos[self.lid_q[tgt]])
        blown_out = self.holder_of.get("fuse_blown") is None
        others_ok = all(self.holder_of.get(self.holder_fuse[i]) == i for i in range(4) if i != tgt)
        success = bool(new_in) and lid < 0.08 and blown_out and others_ok
        prog = (
            0.25 * (lid > 0.9 or bool(new_in) or blown_out)
            + 0.25 * blown_out
            + 0.3 * bool(new_in)
            + 0.2 * (bool(new_in) and lid < 0.08)
        )
        return {
            "success": bool(success),
            "progress": float(min(prog, 1.0)),
            "lid_angle": lid,
            "blown_out": blown_out,
            "new_seated": bool(new_in),
        }

    def oracle_solve(self):
        d, m = self.d, self.m
        tgt = self.target
        n_old = "fuse_blown"
        latch.release(d, self.fuse_eq[n_old])
        self.holder_of[n_old] = None
        a = self.fuse_qadr[n_old]
        d.qpos[a : a + 7] = [0.50, 0.22, 0.05, 1, 0, 0, 0]
        d.qvel[m.jnt_dofadr[m.joint(f"{n_old}_free").id] :][:6] = 0
        a = self.fuse_qadr["fuse_new0"]
        d.qpos[a : a + 7] = [*self.seat[tgt], 1, 0, 0, 0]
        self.armed["fuse_new0"] = True
        d.qpos[self.lid_q[tgt]] = 0.0
        mujoco.mj_forward(m, d)

    def reach_targets(self):
        out = []
        for i in range(4):
            out.append((f"fuse in holder {i + 1}", 0, np.array([FUSE_X, YS[i], SEAT_Z + 0.012]), down_rot(0.0)))
        for k, (cx, cy) in enumerate(self.caddy_xy):
            out.append((f"new fuse {k}", 0, np.array([cx, cy, 0.05]), down_rot(0.0)))
        return out

    @property
    def instruction(self):
        return f"replace the blown fuse in holder {self.target + 1} with a new fuse"
