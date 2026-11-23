"""Cable routing: thread a patch cable through a row of cable-management clips on a tray.

One end of a Cat6-style cable is anchored at the tray; the free end has a moulded RJ45-style plug.
The cable starts lying beside the clip row. The robot must lift it, lay it into every U-channel clip
(open-top, flared lead-in) and leave the plug beyond the last clip.
Success: every clip holds the cable (cable passes through it, entering and leaving on opposite sides) and
the plug is past the last clip, with the cable at rest.
"""

import mujoco
import numpy as np

from ..common.base_env import TaskEnv
from ..common.geom import CAP, box, qx

N_SEG, SEG_LEN, CABLE_R = 52, 0.0125, 0.0030  # 0.65 m of cable plus the plug
BOARD_Z = 0.012  # tray surface height
CLIP_X = (0.44, 0.56, 0.68)
CLIP_L, CLIP_W, CLIP_H = 0.016, 0.0125, 0.020  # length (along cable), clear width, wall height
ANCHOR = (0.31, 0.08)


class CableRouting(TaskEnv):
    name = "cable_routing"
    max_seconds = 90.0
    success_hold = 0.5
    overview = ((0.52, -0.50, 0.50), (0.55, 0.0, 0.01))

    def build(self, spec):
        w = spec.worldbody
        tray = w.add_body(name="tray", pos=(0.55, 0.0, 0.0))
        box(tray, "tray_plate", (0, 0, BOARD_Z / 2), (0.27, 0.24, BOARD_Z / 2), "steel", friction=(0.6, 0.01, 0.001))
        box(tray, "tray_lip_back", (0.27 - 0.003, 0, BOARD_Z + 0.008), (0.003, 0.24, 0.008), "steel")
        for i, x in enumerate(np.linspace(-0.22, 0.22, 5)):  # perforation slots (visual)
            for s in (-1, 1):
                box(
                    tray, f"slot{i}_{s}", (x, s * 0.215, BOARD_Z + 0.0005), (0.012, 0.004, 0.0005), "black", visual=True
                )
        self.clip_names = []
        for i, cx in enumerate(CLIP_X):
            c = w.add_body(name=f"clip{i}", pos=(cx, 0.0, BOARD_Z))  # y set in randomize_state
            hw = CLIP_W / 2
            box(c, f"clip{i}_base", (0, 0, 0.0015), (CLIP_L / 2 + 0.004, hw + 0.006, 0.0015), "plastic_grey")
            for s in (-1, 1):
                box(
                    c,
                    f"clip{i}_wall{s}",
                    (0, s * (hw + 0.002), 0.0015 + CLIP_H / 2),
                    (CLIP_L / 2, 0.002, CLIP_H / 2),
                    "plastic_grey",
                )
                # flared lead-in at the top of each wall
                box(
                    c,
                    f"clip{i}_lip{s}",
                    (0, s * (hw + 0.0045), 0.0015 + CLIP_H - 0.001),
                    (CLIP_L / 2, 0.0035, 0.0012),
                    "plastic_grey",
                    quat=qx(s * 0.55),
                )
            self.clip_names.append(c.name)
        # cable: anchored root (static body) -> ball-joint chain -> plug
        root = w.add_body(name="cable0", pos=(ANCHOR[0], ANCHOR[1], BOARD_Z + CABLE_R + 0.0005))
        box(root, "anchor_boot", (-0.012, 0, 0), (0.014, 0.007, 0.007), "black", contype=0, conaffinity=0)
        body = root
        for i in range(N_SEG):
            if i > 0:
                body = body.add_body(name=f"cable{i}", pos=(SEG_LEN, 0, 0))
                body.add_joint(
                    name=f"cable_j{i}", type=mujoco.mjtJoint.mjJNT_BALL, stiffness=0.006, damping=0.0008, range=(0, 1.3)
                )
            body.add_geom(
                name=f"cable_g{i}",
                type=CAP,
                fromto=(0, 0, 0, SEG_LEN, 0, 0),
                size=(CABLE_R, 0, 0),
                material="plastic_grey",
                density=1100,
                friction=(0.7, 0.005, 1e-4),
                rgba=(0.12, 0.30, 0.65, 1),
            )
        # RJ45-style plug: strain-relief boot + clear body
        box(
            body,
            "plug_boot",
            (SEG_LEN + 0.010, 0, 0),
            (0.012, 0.0085, 0.0085),
            "blue",
            density=1300,
            friction=(1.2, 0.01, 0.001),
            condim=4,
        )
        box(
            body,
            "plug_body",
            (SEG_LEN + 0.033, 0, 0),
            (0.0115, 0.0068, 0.0058),
            "clear_blue",
            density=1300,
            friction=(1.2, 0.01, 0.001),
            condim=4,
        )
        body.add_site(name="plug_site", pos=(SEG_LEN + 0.03, 0, 0), size=(0.003,) * 3)
        self._last = body.name

    def bind(self):
        m = self.m
        self.clip_bid = [m.body(n).id for n in self.clip_names]
        self.seg_bid = [m.body(f"cable{i}").id for i in range(N_SEG)]
        self.plug_site = m.site("plug_site").id
        self.root_bid = m.body("cable0").id
        self.clip_y = np.zeros(3)
        self._root_pos0 = m.body_pos[self.root_bid].copy()
        self._clip_pos0 = [m.body_pos[b].copy() for b in self.clip_bid]
        # reference joint addresses so we can bend the cable at reset
        self.jq = [m.jnt_qposadr[m.joint(f"cable_j{i}").id] for i in range(1, N_SEG)]

    def reset_dynamic_model(self):
        self.m.body_pos[self.root_bid] = self._root_pos0
        for b, p in zip(self.clip_bid, self._clip_pos0):
            self.m.body_pos[b] = p

    def randomize_state(self, rng):
        m, d = self.m, self.d
        # Clips are staggered a little sideways. The cable must weave through them, so the path
        # anchor -> clip 0 -> 1 -> 2 -> plug has to be shorter than the cable. With +-7 cm stagger it was about 0.62 m
        # against a 0.54 m cable, which is infeasible.
        ys = (
            np.array([-0.02, 0.02, -0.01])
            if not self.randomize
            else np.array([-0.02, 0.02, -0.01]) + rng.uniform(-0.01, 0.01, 3)
        )
        self.clip_y = ys
        for b, p, y in zip(self.clip_bid, self._clip_pos0, ys):
            m.body_pos[b] = (p[0], y, p[2])
        ay = ANCHOR[1] + (rng.uniform(-0.02, 0.02) if self.randomize else 0.0)
        m.body_pos[self.root_bid] = (ANCHOR[0], ay, self._root_pos0[2])
        # initial cable layout: gentle sideways S-curve that stays off the clip row
        bend = rng.uniform(-0.05, 0.05) if self.randomize else 0.0
        for k, q in enumerate(self.jq):
            ang = bend * np.sin(k / 6.0)
            d.qpos[q : q + 4] = [np.cos(ang / 2), 0, 0, np.sin(ang / 2)]

    # ---- evaluation
    def _clip_state(self):
        d = self.d
        pts = np.array([d.xpos[b] for b in self.seg_bid])
        pts = np.vstack([pts, d.site_xpos[self.plug_site]])
        holding = []
        for cb in self.clip_bid:
            c = self.m.body_pos[cb]
            rel = pts - np.array([c[0], c[1], c[2]])
            u, v, z = rel[:, 0], rel[:, 1], rel[:, 2]
            inside = (np.abs(u) < CLIP_L / 2) & (np.abs(v) < CLIP_W / 2 - 0.0015) & (z > 0.0) & (z < 0.0015 + CLIP_H)
            holding.append(bool(inside.any() and (u < -CLIP_L / 2).any() and (u > CLIP_L / 2).any()))
        return holding, pts

    def evaluate(self):
        holding, pts = self._clip_state()
        plug = pts[-1]
        beyond = plug[0] > CLIP_X[-1] + 0.025
        success = all(holding) and beyond
        prog = (sum(holding) + float(beyond)) / (len(holding) + 1)
        return {"success": bool(success), "progress": prog, "clips_holding": holding}

    def oracle_solve(self):
        """Lay the cable straight along the clip row, through every clip."""
        m, d = self.m, self.d
        m.body_pos[self.root_bid] = (ANCHOR[0], 0.0, self._root_pos0[2])
        for b, p in zip(self.clip_bid, self._clip_pos0):
            m.body_pos[b] = (p[0], 0.0, p[2])
        for q in self.jq:
            d.qpos[q : q + 4] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)

    @property
    def instruction(self):
        return "route the cable through all three clips on the tray"
