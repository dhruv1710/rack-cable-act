"""Cable insertion: plug an RJ45-style cable into a port of a rack patch panel.

This wraps the existing, ACT-trained `tasks/rack_cable` environment unchanged (same physics, observations
and 8-D joint-space action) and only adds *visual* realism: a populated rack (1U switches with port arrays
and link LEDs, rack-unit hole patterns, blanking panels, port labels, a back wall). All added geoms have
contype=conaffinity=0, so dynamics and trained policies are unaffected.
"""

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np

# Load the legacy env by file path: `tasks` (repo root) collides with `envs/tasks`
# when a script inside envs/ is run directly.
_RACK = Path(__file__).resolve().parents[2] / "tasks" / "rack_cable"
if str(_RACK) not in sys.path:
    sys.path.insert(0, str(_RACK))
import build_scene as bs

_spec = importlib.util.spec_from_file_location("rack_cable_env", _RACK / "env.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["rack_cable_env"] = _mod
_spec.loader.exec_module(_mod)
RackCableEnv = _mod.RackCableEnv

V = {"contype": 0, "conaffinity": 0}
BOX = mujoco.mjtGeom.mjGEOM_BOX


def _g(body, name, pos, half, rgba, **kw):
    body.add_geom(name=name, type=BOX, pos=pos, size=half, rgba=rgba, **V, **kw)


def add_visual_detail(spec: mujoco.MjSpec):
    w = spec.worldbody
    rack = w.add_body(name="rack_visuals")
    xf = bs.PANEL_FRONT_X
    U = 0.0445
    # rack-unit hole pattern on the two front posts (3 holes per U)
    for side, y in (("l", -0.30), ("r", 0.30)):
        for k in range(30):
            z = 0.12 + k * U / 3
            _g(rack, f"rackhole_{side}{k}", (0.6085, y, z), (0.0006, 0.0045, 0.0045), (0.02, 0.02, 0.02, 1))
    # 1U switches above and below the patch panel (faceplates flush with the panel front)
    for name, z in (("sw_lo", 0.40), ("sw_hi", 0.70), ("sw_hi2", 0.75)):
        _g(rack, name, (xf - 0.001, 0, z), (0.003, 0.255, 0.0205), (0.16, 0.17, 0.19, 1))
        for p in range(24):  # 24 RJ45 ports in two rows
            row, col = p % 2, p // 2
            y = -0.20 + col * 0.0335
            _g(
                rack,
                f"{name}_p{p}",
                (xf - 0.0035, y, z + (0.007 if row == 0 else -0.009)),
                (0.0008, 0.0075, 0.0058),
                (0.02, 0.02, 0.02, 1),
            )
            led = (0.1, 0.9, 0.2, 1) if (p * 7 + int(z * 100)) % 5 else (1.0, 0.6, 0.0, 1)
            _g(
                rack,
                f"{name}_l{p}",
                (xf - 0.0037, y + 0.0055, z + (0.0145 if row == 0 else -0.0015)),
                (0.0005, 0.0018, 0.0018),
                led,
            )
        _g(rack, f"{name}_vent", (xf - 0.0036, 0.235, z), (0.0005, 0.013, 0.015), (0.05, 0.05, 0.05, 1))
    # blanking panels and a PDU strip
    for z in (0.20, 0.245, 0.29, 0.335, 0.45, 0.80, 0.845):
        _g(rack, f"blank_{int(z * 1000)}", (xf - 0.001, 0, z), (0.003, 0.255, 0.0205), (0.10, 0.10, 0.11, 1))
    _g(rack, "pdu", (0.64, -0.325, 0.5), (0.015, 0.012, 0.35), (0.08, 0.08, 0.09, 1))
    for k in range(10):
        _g(rack, f"pdu_o{k}", (0.6245, -0.325, 0.2 + k * 0.06), (0.0008, 0.008, 0.012), (0.9, 0.9, 0.9, 1))
    # port number labels above the patch-panel ports
    for i, y in enumerate(bs.PORT_Y):
        _g(rack, f"label{i}", (xf - 0.0005, y, bs.PANEL_Z + 0.03), (0.0004, 0.012, 0.005), (0.95, 0.95, 0.9, 1))
        for k in range(i + 1):
            _g(
                rack,
                f"label{i}_t{k}",
                (xf - 0.0009, y - 0.008 + k * 0.005, bs.PANEL_Z + 0.03),
                (0.0002, 0.0012, 0.0035),
                (0.1, 0.1, 0.1, 1),
            )
    # rack side panels (translucent), top cap, back wall
    _g(rack, "side_l", (0.71, -0.318, 0.5), (0.095, 0.002, 0.5), (0.1, 0.1, 0.12, 0.25))
    _g(rack, "side_r", (0.71, 0.318, 0.5), (0.095, 0.002, 0.5), (0.1, 0.1, 0.12, 0.25))
    _g(rack, "wall", (1.45, 0, 1.0), (0.01, 2.0, 1.2), (0.55, 0.57, 0.60, 1))
    w.add_light(name="rack_light", pos=(0.2, 0.0, 1.6), dir=(0.5, 0, -1), diffuse=(0.35, 0.35, 0.38))


class CableInsertion(RackCableEnv):
    name = "cable_insertion"

    def __init__(self, port=2, detail=True, **kw):
        super().__init__(port=port, spec_hook=add_visual_detail if detail else None, **kw)

    def reach_targets(self):
        return []  # reachability is already proven by tasks/rack_cable/scripted_insert.py

    # unified API used by tests / generic tooling
    def noop_action(self):
        return np.r_[self.q_des, 1.0].astype(np.float32)

    def physics(self, n=1):
        for _ in range(n):
            self.control_step(self.q_des, self.grip)

    def evaluate(self):
        e = self.tip_error() * 1000
        prog = float(np.clip(1.0 - np.linalg.norm(e) / 300.0, 0, 1))
        return {"success": bool(self.latched), "progress": 1.0 if self.latched else prog}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info.update(self.evaluate())
        return obs, info

    def oracle_solve(self):
        """Teleport the (straightened) cable so the plug tip is at the port and latch."""
        m, d = self.m, self.d
        a = self.cable_qadr
        d.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
        for j in range(1, bs.N_SEG):
            q = m.jnt_qposadr[m.joint(f"cable_j{j}").id]
            d.qpos[q : q + 4] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        d.qpos[a : a + 3] -= self.tip_error()
        mujoco.mj_forward(m, d)
        self._maybe_latch()
