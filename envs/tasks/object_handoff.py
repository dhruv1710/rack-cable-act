"""Object handoff between two grippers (bimanual): left arm picks a bar, hands it to the right arm,
which places it on the green target pad.

Two Franka Pandas face each other across a 1.45 m bench (bases at x=0 and x=1.0). The object is a
40x40x150 mm anodised bar standing near the left arm. A genuine handoff is required: the receiving
gripper must hold the bar while the giving gripper still does (overlap), then the giver lets go.
Success: handoff completed, bar resting on the target pad, neither gripper touching it.
"""

import mujoco
import numpy as np

from ..common.arms import ArmCfg, down_rot
from ..common.base_env import TaskEnv
from ..common.geom import box, qz

BAR_HALF = (0.02, 0.02, 0.075)
PAD_C, PAD_HALF = np.array([0.72, 0.24]), 0.055
START_C = np.array([0.30, -0.22])


class ObjectHandoff(TaskEnv):
    name = "object_handoff"
    arm_cfgs = (
        ArmCfg("left_", (0.0, 0.0, 0.0), 0.0, "wrist_left"),
        ArmCfg("right_", (1.0, 0.0, 0.0), np.pi, "wrist_right"),
    )
    world_kw = {"table_x": (-0.35, 1.35)}
    max_seconds = 90.0
    success_hold = 0.5
    overview = ((0.50, -1.15, 0.80), (0.50, 0.0, 0.10))

    def build(self, spec):
        w = spec.worldbody
        box(w, "pad", (*PAD_C, 0.0015), (PAD_HALF, PAD_HALF, 0.0015), "green", friction=(0.8, 0.01, 0.001))
        for s in (-1, 1):
            box(
                w,
                f"pad_edge_x{s}",
                (PAD_C[0] + s * PAD_HALF, PAD_C[1], 0.0016),
                (0.002, PAD_HALF, 0.0006),
                "white",
                visual=True,
            )
            box(
                w,
                f"pad_edge_y{s}",
                (PAD_C[0], PAD_C[1] + s * PAD_HALF, 0.0016),
                (PAD_HALF, 0.002, 0.0006),
                "white",
                visual=True,
            )
        box(w, "start_mark", (*START_C, 0.0008), (0.035, 0.035, 0.0008), "yellow", visual=True)
        b = w.add_body(name="bar", pos=(*START_C, BAR_HALF[2] + 0.002))
        b.add_freejoint(name="bar_free")
        box(b, "bar_core", (0, 0, 0), BAR_HALF, "orange", density=900, friction=(1.0, 0.01, 0.001))
        box(
            b,
            "bar_band",
            (0, 0, 0.0),
            (BAR_HALF[0] + 0.0004, BAR_HALF[1] + 0.0004, 0.012),
            "black",
            contype=0,
            conaffinity=0,
        )

    def bind(self):
        m = self.m
        self.bar = m.body("bar").id
        self.bar_q = m.jnt_qposadr[m.joint("bar_free").id]
        self.bar_geoms = self.body_geoms("bar")
        self.stage = 0  # 0 none, 1 left holds, 2 both hold (handoff), 3 right holds alone after handoff

    def randomize_state(self, rng):
        j = rng.uniform(-0.03, 0.03, 2) if self.randomize else np.zeros(2)
        yaw = rng.uniform(-0.4, 0.4) if self.randomize else 0.0
        a = self.bar_q
        self.d.qpos[a : a + 7] = [*(START_C + j), BAR_HALF[2] + 0.002, *qz(yaw)]
        self.stage = 0

    def _holds(self, arm):
        """Both fingers of the arm are touching the bar."""
        d, m = self.d, self.m
        hit = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.bar_geoms and g2 in arm.finger_geoms:
                hit.add(m.geom_bodyid[g2])
            elif g2 in self.bar_geoms and g1 in arm.finger_geoms:
                hit.add(m.geom_bodyid[g1])
        return len(hit) >= 2

    def on_substep(self):
        l, r = self._holds(self.arms[0]), self._holds(self.arms[1])
        if self.stage == 0 and l and self.d.xpos[self.bar][2] > BAR_HALF[2] + 0.02:
            self.stage = 1
        if self.stage == 1 and l and r:
            self.stage = 2
        if self.stage == 2 and r and not l:
            self.stage = 3

    def evaluate(self):
        p = self.d.xpos[self.bar]
        on_pad = abs(p[0] - PAD_C[0]) < PAD_HALF and abs(p[1] - PAD_C[1]) < PAD_HALF and p[2] < BAR_HALF[2] + 0.02
        free = not (self._holds(self.arms[0]) or self._holds(self.arms[1]))
        upright = abs(self.d.xmat[self.bar].reshape(3, 3)[2, 2]) > 0.9
        success = self.stage == 3 and on_pad and free and upright
        prog = [0.0, 0.25, 0.55, 0.8][self.stage] + (0.2 if success else 0.0)
        return {"success": bool(success), "progress": min(prog, 1.0), "stage": self.stage}

    def oracle_solve(self):
        self.stage = 3
        a = self.bar_q
        self.d.qpos[a : a + 7] = [*PAD_C, BAR_HALF[2] + 0.002, 1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def reach_targets(self):
        mid = np.array([0.5, 0.0, 0.30])
        return [
            ("left: pick bar", 0, np.array([*START_C, 0.10]), down_rot(0.0)),
            ("left: handoff point", 0, mid - np.array([0.06, 0, 0]), down_rot(0.0)),
            ("right: place on pad", 1, np.array([*PAD_C, 0.12]), down_rot(0.0)),
        ]

    @property
    def instruction(self):
        return "hand the bar from the left arm to the right arm and place it on the green pad"
