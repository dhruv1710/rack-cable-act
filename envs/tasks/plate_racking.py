"""Plate racking: stand a 200 mm melamine dinner plate upright in the requested slot of a wire dish rack.

The plate rests on four posts (so a gripper can get under the rim). The rack is a tray with five colour-coded
slots; each slot is a pair of tine rows (front/back) whose gap is ~14 mm for a 9 mm plate. The plate must be
picked flat, rotated to vertical (wrist roll) and lowered into the target slot.
Success: plate standing in the target slot (normal horizontal within ~20 deg, centred in the slot gap and along
the tines), supported by the rack base, at rest, and not held.
"""

import mujoco
import numpy as np

from ..common.arms import forward_rot
from ..common.base_env import TaskEnv
from ..common.geom import CYL, box, cyl, qx, qz

PLATE_R, PLATE_T = 0.100, 0.009
SRC_C = np.array([0.52, -0.20])
POST_H = 0.16  # plate stand height: a horizontal Panda hand needs >=12 cm of palm clearance
RACK_C = np.array([0.58, 0.14])
SLOT_PITCH, N_SLOTS = 0.042, 5
TINE_R, TINE_H, TINE_GAP = 0.0028, 0.085, 0.0095  # tine radius, height, half-gap between the pair
TINE_X = 0.068
BASE_T = 0.010
SLOT_COLORS = ["red", "orange", "yellow", "green", "blue"]


class PlateRacking(TaskEnv):
    name = "plate_racking"
    max_seconds = 120.0
    success_hold = 0.5
    overview = ((0.45, -0.60, 0.50), (0.50, 0.0, 0.06))

    def slot_y(self, k):
        return RACK_C[1] + (k - (N_SLOTS - 1) / 2) * SLOT_PITCH

    def build(self, spec):
        w = spec.worldbody
        # source stand: 4 posts under the plate
        for i, (dx, dy) in enumerate([(0.05, 0.05), (0.05, -0.05), (-0.01, 0.07), (-0.01, -0.07)]):
            cyl(
                w,
                f"post{i}",
                (SRC_C[0] + dx, SRC_C[1] + dy, POST_H / 2),
                0.006,
                POST_H / 2,
                "dark_steel",
                friction=(0.8, 0.01, 0.001),
            )
        box(
            w, "src_top", (SRC_C[0] + 0.02, SRC_C[1], POST_H - 0.003), (0.085, 0.085, 0.0015), "dark_steel", visual=True
        )
        box(w, "src_base", (*SRC_C, 0.002), (0.10, 0.10, 0.002), "dark_steel", contype=0, conaffinity=0)
        p = w.add_body(name="plate", pos=(*SRC_C, POST_H + PLATE_T / 2 + 0.002))
        p.add_freejoint(name="plate_free")
        p.add_geom(
            name="plate_disc",
            type=CYL,
            size=(PLATE_R, PLATE_T / 2, 0),
            material="ceramic",
            density=750,
            condim=4,
            friction=(0.7, 0.015, 0.001),
        )  # melamine plate ~0.21 kg (a 0.45 kg porcelain one is too heavy to hold by a 9 mm rim)
        p.add_geom(
            name="plate_well",
            type=CYL,
            pos=(0, 0, PLATE_T / 2 + 0.0004),
            size=(PLATE_R - 0.022, 0.0004, 0),
            material="white",
            contype=0,
            conaffinity=0,
        )
        p.add_geom(
            name="plate_ring",
            type=CYL,
            pos=(0, 0, PLATE_T / 2 + 0.0006),
            size=(PLATE_R - 0.012, 0.0004, 0),
            material="blue",
            contype=0,
            conaffinity=0,
        )
        # rack: wire tray + tine pairs
        rk = w.add_body(name="rack", pos=(RACK_C[0], RACK_C[1], 0.0))
        half_y = N_SLOTS * SLOT_PITCH / 2 + 0.012
        box(rk, "rack_base", (0, 0, BASE_T / 2), (0.085, half_y, BASE_T / 2), "alu", friction=(0.7, 0.01, 0.001))
        for s in (-1, 1):
            box(
                rk,
                f"rack_rail{s}",
                (0, s * (half_y - 0.002), BASE_T + 0.005),
                (0.085, 0.002, 0.005),
                "alu",
                visual=True,
            )
        for k in range(N_SLOTS):
            y = (k - (N_SLOTS - 1) / 2) * SLOT_PITCH
            box(rk, f"slot_mark{k}", (0.0, y, BASE_T + 0.0005), (0.075, 0.004, 0.0005), SLOT_COLORS[k], visual=True)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    cyl(
                        rk,
                        f"tine{k}_{sx}_{sy}",
                        (sx * TINE_X, y + sy * TINE_GAP, BASE_T + TINE_H / 2),
                        TINE_R,
                        TINE_H / 2,
                        "steel",
                        friction=(0.4, 0.01, 0.001),
                    )
                    rk.add_geom(
                        name=f"tine{k}_{sx}_{sy}_tip",
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        pos=(sx * TINE_X, y + sy * TINE_GAP, BASE_T + TINE_H),
                        size=(TINE_R * 1.6, 0, 0),
                        material="steel",
                        friction=(0.4, 0.01, 0.001),
                    )
        self.slot_names = SLOT_COLORS

    def bind(self):
        m = self.m
        self.plate = m.body("plate").id
        self.plate_q = m.jnt_qposadr[m.joint("plate_free").id]
        self.plate_dof = m.jnt_dofadr[m.joint("plate_free").id]
        self.plate_geoms = self.body_geoms("plate")
        self.target = 2

    def randomize_state(self, rng):
        self.target = int(rng.integers(0, N_SLOTS)) if self.randomize else 2
        j = rng.uniform(-0.015, 0.015, 2) if self.randomize else np.zeros(2)
        yaw = rng.uniform(-0.3, 0.3) if self.randomize else 0.0
        a = self.plate_q
        self.d.qpos[a : a + 7] = [*(SRC_C + j), POST_H + PLATE_T / 2 + 0.002, *qz(yaw)]

    def _held(self):
        return any(self.contacts_between(self.plate_geoms, a.finger_geoms) for a in self.arms)

    def evaluate(self):
        d = self.d
        p = d.xpos[self.plate]
        nz = d.xmat[self.plate].reshape(3, 3)[:, 2]  # plate normal
        tilt = float(np.arccos(np.clip(abs(nz[1]), -1, 1)))  # angle between normal and the y axis (slot normal)
        sy = self.slot_y(self.target)
        in_slot = abs(p[1] - sy) < 0.010 and abs(p[0] - RACK_C[0]) < 0.03
        low = BASE_T + PLATE_R - 0.012 < p[2] < BASE_T + PLATE_R + 0.03
        speed = float(np.linalg.norm(d.qvel[self.plate_dof : self.plate_dof + 6]))
        upright = tilt < 0.35
        success = in_slot and low and upright and speed < 0.05 and not self._held()
        near = np.clip(1 - np.linalg.norm(p[:2] - np.array([RACK_C[0], sy])) / 0.4, 0, 1)
        prog = 0.2 * near + 0.3 * upright * (p[2] > 0.06) + 0.5 * bool(in_slot and low and upright)
        return {
            "success": bool(success),
            "progress": float(min(prog, 1.0)),
            "tilt_deg": float(np.degrees(tilt)),
            "target_slot": self.slot_names[self.target],
        }

    def oracle_solve(self):
        sy = self.slot_y(self.target)
        a = self.plate_q
        # plate normal along y: rotate cylinder axis (z) to y
        self.d.qpos[a : a + 7] = [RACK_C[0], sy, BASE_T + PLATE_R + 0.0005, *qx(np.pi / 2)]
        self.d.qvel[self.plate_dof : self.plate_dof + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def reach_targets(self):
        # hand horizontal; fingers close vertically on the flat plate's near rim, then along y on the standing plate
        rim = np.array([SRC_C[0] - PLATE_R + 0.012, SRC_C[1], POST_H + PLATE_T / 2 + 0.002])
        hold = lambda k: np.array([RACK_C[0] - PLATE_R + 0.012, self.slot_y(k), BASE_T + PLATE_R + 0.05])
        return [
            ("grip flat plate rim", 0, rim, forward_rot(0.0, np.pi / 2)),
            ("hold plate over slot 0", 0, hold(0), forward_rot(0.0, 0.0)),
            ("hold plate over slot 4", 0, hold(4), forward_rot(0.0, 0.0)),
        ]

    @property
    def instruction(self):
        return f"stand the plate upright in the {self.slot_names[self.target]} slot of the rack"
