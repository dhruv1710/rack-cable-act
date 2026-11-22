"""Thermal patrolling: a wheeled inspection robot patrols a data-center cold aisle and flags overheating servers.

Scene: a 6 m aisle between two rows of eight 42U racks (20 x 2U servers each, front panels with vents,
handles and status LEDs), raised-floor tiles, ceiling light strips. The robot is a differential-drive base with
a mast carrying a pan/tilt head with an RGB camera and a 160x120 thermal camera.

Thermal model: every geom has a surface temperature (racks 24-38 C with a vertical gradient, 1-3 servers
"hot" at 58-78 C, a few "warm" decoys at 42-48 C that must NOT be flagged). The thermal image is rendered by
segmentation -> temperature lookup -> thermal diffusion blur + sensor noise -> iron colour map, so heat is
only visible through the thermal camera (the RGB image gives no direct cue).

Action: [v (m/s), w (rad/s), pan (rad), tilt (rad), flag in {0,1}] ; a rising edge of `flag` reports an anomaly at
whatever the thermal camera is centred on. Success: every hot server flagged, at most 1 false flag.
"""

import mujoco
import numpy as np
from gymnasium import spaces

from ..common.base_env import TaskEnv
from ..common.geom import box, cyl, qx

N_RACK, RACK_W, RACK_D, RACK_H = 8, 0.60, 1.00, 2.00
AISLE_HALF = 0.60  # rack fronts at y = +-0.60
X0 = 0.9
N_SLOT, SLOT_Z0, SLOT_PITCH = 20, 0.15, 0.0889
HOT_T, WARM_T = 55.0, 40.0
IMG_T = (120, 160)
CAM_Z = 1.20
AMBIENT = 21.0


def ironbow(t):
    """t in [0,1] -> uint8 RGB (black-purple-red-orange-yellow-white)."""
    anchors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[0, 0, 0], [60, 0, 110], [200, 40, 40], [250, 170, 20], [255, 255, 210]], float)
    return np.stack([np.interp(t, anchors, cols[:, c]) for c in range(3)], -1).astype(np.uint8)


def blur(a, k=2):
    for _ in range(k):
        a = (np.roll(a, 1, 0) + 2 * a + np.roll(a, -1, 0)) / 4
        a = (np.roll(a, 1, 1) + 2 * a + np.roll(a, -1, 1)) / 4
    return a


class ThermalPatrol(TaskEnv):
    name = "thermal_patrol"
    arm_cfgs = ()
    world_kw = {"with_table": False, "floor_z": 0.0, "floor_half": 12.0, "floor_repeat": 0.8333, "zfar": 30.0}
    max_seconds = 150.0
    success_hold = 0.2
    settle_seconds = 0.2
    overview = ((0.10, 0.0, 2.3), (3.5, 0.0, 0.9))

    # ---------------------------------------------------------------- build
    def build(self, spec):
        w = spec.worldbody
        self.server_geoms = []  # (geom name, rack idx, slot idx, centre xyz)
        for s, sy in enumerate((-1, 1)):
            for k in range(N_RACK):
                rid = s * N_RACK + k
                xc = X0 + RACK_W * k
                sy * (AISLE_HALF + 0.02)  # back of the front-panel plane
                rack = w.add_body(name=f"rack{rid}", pos=(xc, 0.0, 0.0))
                # carcass: collision box behind the front plane + visual posts/panels
                box(
                    rack,
                    f"r{rid}_body",
                    (0, sy * (AISLE_HALF + 0.02 + (RACK_D - 0.04) / 2), RACK_H / 2),
                    (RACK_W / 2 - 0.002, (RACK_D - 0.04) / 2, RACK_H / 2),
                    "charcoal",
                )
                for dx in (-1, 1):
                    box(
                        rack,
                        f"r{rid}_post{dx}",
                        (dx * (RACK_W / 2 - 0.012), sy * (AISLE_HALF + 0.012), RACK_H / 2),
                        (0.012, 0.012, RACK_H / 2),
                        "black",
                        visual=True,
                    )
                box(
                    rack,
                    f"r{rid}_top",
                    (0, sy * (AISLE_HALF + 0.02 + RACK_D / 2), RACK_H + 0.002),
                    (RACK_W / 2, RACK_D / 2, 0.002),
                    "black",
                    visual=True,
                )
                for sl in range(N_SLOT):
                    z = SLOT_Z0 + sl * SLOT_PITCH + 0.04
                    gy = sy * (AISLE_HALF + 0.005)
                    n = f"srv_{rid}_{sl}"
                    box(rack, n, (0, gy, z), (RACK_W / 2 - 0.035, 0.015, 0.0395), "dark_steel", visual=True)
                    for hx in (-1, 1):  # rack ears + handles
                        box(
                            rack,
                            f"{n}_h{hx}",
                            (hx * (RACK_W / 2 - 0.045), gy - sy * 0.016, z),
                            (0.006, 0.004, 0.012),
                            "alu",
                            visual=True,
                        )
                    box(rack, f"{n}_vent", (-0.06, gy - sy * 0.0155, z), (0.17, 0.0006, 0.026), "black", visual=True)
                    led = ["led_green", "led_amber"][(rid * 7 + sl * 3) % 11 == 0]
                    box(rack, f"{n}_led", (0.19, gy - sy * 0.0155, z + 0.015), (0.003, 0.0006, 0.003), led, visual=True)
                    self.server_geoms.append((n, rid, sl, np.array([xc, gy, z])))
        # aisle floor markings, walls, ceiling, light strips
        box(w, "end_wall_a", (-0.4, 0, 1.5), (0.02, 2.0, 1.5), "white", friction=(0.5, 0.01, 0.001))
        box(w, "end_wall_b", (X0 + RACK_W * N_RACK + 0.6, 0, 1.5), (0.02, 2.0, 1.5), "white")
        box(w, "ceiling", (3.5, 0, 2.7), (6.0, 3.0, 0.02), "white", visual=True)
        for k in range(5):
            box(w, f"light{k}", (0.8 + 1.4 * k, 0, 2.675), (0.5, 0.08, 0.005), "led_off", visual=True)
        box(w, "aisle_line_a", (3.5, -0.22, 0.0006), (3.2, 0.012, 0.0004), "yellow", visual=True)
        box(w, "aisle_line_b", (3.5, 0.22, 0.0006), (3.2, 0.012, 0.0004), "yellow", visual=True)
        for k in range(4):
            w.add_light(
                name=f"hall{k}",
                pos=(1.2 + 1.7 * k, 0, 2.6),
                dir=(0, 0, -1),
                diffuse=(0.55, 0.55, 0.52),
                castshadow=False,
            )
        # robot
        r = w.add_body(name="robot", pos=(0.0, 0.0, 0.0))  # slide joints = world x/y
        for jn, ax, ty in (
            ("base_x", (1, 0, 0), mujoco.mjtJoint.mjJNT_SLIDE),
            ("base_y", (0, 1, 0), mujoco.mjtJoint.mjJNT_SLIDE),
            ("base_yaw", (0, 0, 1), mujoco.mjtJoint.mjJNT_HINGE),
        ):
            r.add_joint(name=jn, type=ty, axis=ax, damping=2.0 if ty == mujoco.mjtJoint.mjJNT_SLIDE else 1.0)
        cyl(r, "chassis", (0, 0, 0.13), 0.26, 0.09, "alu", mass=18)
        cyl(r, "bumper", (0, 0, 0.07), 0.275, 0.03, "black", mass=1)
        for s in (-1, 1):
            cyl(
                r,
                f"wheel{s}",
                (0, s * 0.27, 0.075),
                0.075,
                0.025,
                "rubber",
                quat=qx(np.pi / 2),
                contype=0,
                conaffinity=0,
            )
        cyl(r, "mast", (0, 0, 0.72), 0.025, 0.50, "dark_steel", mass=2, contype=0, conaffinity=0)
        box(r, "battery", (-0.1, 0, 0.24), (0.12, 0.14, 0.03), "blue", visual=True)
        box(r, "status_strip", (0.25, 0, 0.17), (0.003, 0.14, 0.012), "led_green", visual=True)
        head = r.add_body(name="pan_link", pos=(0, 0, CAM_Z))
        head.add_joint(name="pan", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), range=(-1.7, 1.7), damping=0.3)
        cyl(head, "pan_collar", (0, 0, -0.03), 0.03, 0.02, "dark_steel", mass=0.2, contype=0, conaffinity=0)
        tilt = head.add_body(name="tilt_link", pos=(0, 0, 0.0))
        tilt.add_joint(name="tilt", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, -1, 0), range=(-0.9, 0.9), damping=0.3)
        box(tilt, "head_box", (0.03, 0, 0), (0.05, 0.09, 0.05), "charcoal", mass=1.0, contype=0, conaffinity=0)
        cyl(tilt, "lens_rgb", (0.082, 0.04, 0.0), 0.016, 0.006, "black", quat=qy_(np.pi / 2), contype=0, conaffinity=0)
        cyl(
            tilt,
            "lens_thermal",
            (0.082, -0.03, 0.0),
            0.022,
            0.008,
            "steel",
            quat=qy_(np.pi / 2),
            contype=0,
            conaffinity=0,
        )
        look_x = (0, -1, 0, 0, 0, 1)  # camera looks along +x, up = +z
        tilt.add_camera(name="rgb", pos=(0.09, 0.04, 0.0), xyaxes=look_x, fovy=55)
        tilt.add_camera(name="thermal", pos=(0.09, -0.03, 0.0), xyaxes=look_x, fovy=50)
        # actuators: velocity servos on the planar base, position servos on pan/tilt
        kv = 60.0
        for jn in ("base_x", "base_y", "base_yaw"):
            a = spec.add_actuator(name=f"v_{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT, target=jn)
            a.gainprm[0], a.biasprm[2], a.biastype = kv, -kv, mujoco.mjtBias.mjBIAS_AFFINE
            a.ctrlrange, a.ctrllimited = (-3, 3), 1
        for jn, kp in (("pan", 12.0), ("tilt", 12.0)):
            a = spec.add_actuator(name=f"p_{jn}", trntype=mujoco.mjtTrn.mjTRN_JOINT, target=jn)
            a.gainprm[0], a.biasprm[1], a.biasprm[2], a.biastype = kp, -kp, -0.8, mujoco.mjtBias.mjBIAS_AFFINE

    # ---------------------------------------------------------------- bind
    def extra_cameras(self):
        return {"head_rgb": "rgb"}

    def bind(self):
        m = self.m
        self.j = {n: m.joint(n) for n in ("base_x", "base_y", "base_yaw", "pan", "tilt")}
        self.qa = {n: m.jnt_qposadr[self.j[n].id] for n in self.j}
        self.act = {n: m.actuator(n).id for n in ("v_base_x", "v_base_y", "v_base_yaw", "p_pan", "p_tilt")}
        self.thermal_cam = m.camera("thermal").id
        self.srv_gid = np.array([m.geom(n).id for n, *_ in self.server_geoms])
        self.srv_rack = np.array([r for _, r, _, _ in self.server_geoms])
        self.srv_slot = np.array([s for _, _, s, _ in self.server_geoms])
        self.srv_pos = np.array([p for *_, p in self.server_geoms])
        self.geom_temp = np.full(m.ngeom, AMBIENT)
        self.srv_temp = np.zeros(len(self.srv_gid))
        self.hot = np.zeros(len(self.srv_gid), bool)
        self.flagged = np.zeros(len(self.srv_gid), bool)
        self.false_flags = 0
        self._last_flag = 0.0
        self._seg = mujoco.Renderer(m, *IMG_T)
        self._seg.enable_segmentation_rendering()
        self._rng_img = np.random.default_rng(0)

    @property
    def state_dim(self):
        return 6

    def make_action_space(self):
        return spaces.Box(
            np.array([-0.5, -1.2, -1.7, -0.9, 0.0], np.float32), np.array([0.8, 1.2, 1.7, 0.9, 1.0], np.float32)
        )

    def home_action(self):
        return np.zeros(5)

    def noop_action(self):
        return np.zeros(5, np.float32)

    def get_state(self):
        d, q = self.d, self.qa
        yaw = d.qpos[q["base_yaw"]]
        return np.array(
            [d.qpos[q["base_x"]], d.qpos[q["base_y"]], np.cos(yaw), np.sin(yaw), d.qpos[q["pan"]], d.qpos[q["tilt"]]],
            np.float32,
        )

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self._renderer is not None:
            self.observation_space["observation.images.thermal"] = spaces.Box(0, 255, (*IMG_T, 3), np.uint8)

    # ---------------------------------------------------------------- reset
    def randomize_state(self, rng):
        d = self.d
        if not self.randomize:
            rng = np.random.default_rng(0)
        d.qpos[self.qa["base_x"]] = 0.3 + (rng.uniform(0, 0.4))
        d.qpos[self.qa["base_y"]] = rng.uniform(-0.12, 0.12)
        d.qpos[self.qa["base_yaw"]] = rng.uniform(-0.25, 0.25)
        n = len(self.srv_gid)
        rack_base = 26 + rng.uniform(0, 8, 2 * N_RACK)
        grad = 0.07 * self.srv_slot + rng.normal(0, 0.8, n)
        T = rack_base[self.srv_rack] + grad
        order = rng.permutation(n)
        racks_used = set()
        hot_idx = []
        for i in order:
            if self.srv_rack[i] not in racks_used and len(hot_idx) < int(rng.integers(1, 4)):
                hot_idx.append(i)
                racks_used.add(self.srv_rack[i])
        warm_idx = [i for i in order if i not in hot_idx and self.srv_rack[i] not in racks_used][:4]
        for i in hot_idx:
            T[i] = rng.uniform(58, 78)
        for i in warm_idx:
            T[i] = rng.uniform(42, 48)
        self.srv_temp = T
        self.hot = np.zeros(n, bool)
        self.hot[hot_idx] = True
        self.flagged = np.zeros(n, bool)
        self.false_flags = 0
        self._last_flag = 0.0
        self.geom_temp[:] = AMBIENT
        m = self.m
        for g in range(m.ngeom):
            nm = m.geom(g).name
            if nm.startswith("r") and "_body" in nm:
                self.geom_temp[g] = 24.0
            elif nm in ("chassis", "battery"):
                self.geom_temp[g] = 29.0
            elif nm.startswith("light") or nm == "ceiling":
                self.geom_temp[g] = 24.0
        self.geom_temp[self.srv_gid] = T
        # neighbours above/below a hot server pick up some heat
        for i in hot_idx:
            for j in np.where((self.srv_rack == self.srv_rack[i]) & (np.abs(self.srv_slot - self.srv_slot[i]) == 1))[0]:
                self.geom_temp[self.srv_gid[j]] = max(
                    self.geom_temp[self.srv_gid[j]], T[i] * 0.5 + self.srv_temp[j] * 0.5
                )
        # LED stays consistent with nothing: RGB gives no thermal cue

    # ---------------------------------------------------------------- control
    def apply_action(self, a):
        d = self.d
        v, w, pan, tilt, flag = [float(x) for x in a]
        yaw = d.qpos[self.qa["base_yaw"]]
        d.ctrl[self.act["v_base_x"]] = v * np.cos(yaw)
        d.ctrl[self.act["v_base_y"]] = v * np.sin(yaw)
        d.ctrl[self.act["v_base_yaw"]] = w
        d.ctrl[self.act["p_pan"]] = pan
        d.ctrl[self.act["p_tilt"]] = tilt
        if flag > 0.5 and self._last_flag <= 0.5:
            self._report()
        self._last_flag = flag

    def _cam(self):
        d = self.d
        pos = d.cam_xpos[self.thermal_cam].copy()
        fwd = -d.cam_xmat[self.thermal_cam].reshape(3, 3)[:, 2]
        return pos, fwd

    def _report(self):
        pos, fwd = self._cam()
        v = self.srv_pos - pos
        dist = np.linalg.norm(v, axis=1)
        ang = np.arccos(np.clip((v @ fwd) / np.maximum(dist, 1e-6), -1, 1))
        ok = (dist < 2.0) & (dist > 0.3)
        if not ok.any():
            self.false_flags += 1
            return
        ang_m = np.where(ok, ang, 9.0)
        i = int(ang_m.argmin())
        if ang_m[i] < 0.08 and self.srv_temp[i] >= HOT_T and not self.flagged[i]:
            self.flagged[i] = True
        elif not (ang_m[i] < 0.08 and self.flagged[i]):
            self.false_flags += 1

    # ---------------------------------------------------------------- observations
    def thermal_temperature(self):
        self._seg.update_scene(self.d, camera="thermal")
        seg = self._seg.render()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        T = np.where(gid >= 0, self.geom_temp[np.maximum(gid, 0)], 20.0)
        T = blur(T, 2) + self._rng_img.normal(0, 0.35, T.shape)
        return T.astype(np.float32)

    def get_obs(self):
        obs = super().get_obs()
        T = self.thermal_temperature()
        obs["observation.images.thermal"] = ironbow(np.clip((T - 15.0) / 70.0, 0, 1))
        return obs

    # ---------------------------------------------------------------- eval
    def evaluate(self):
        n_hot = int(self.hot.sum())
        got = int((self.flagged & self.hot).sum())
        success = got == n_hot and self.false_flags <= 1
        prog = got / max(n_hot, 1) - 0.15 * self.false_flags
        return {
            "success": bool(success),
            "progress": float(np.clip(prog, 0, 1)),
            "hot_found": got,
            "hot_total": n_hot,
            "false_flags": self.false_flags,
        }

    def oracle_solve(self):
        self.flagged = self.hot.copy()
        self.false_flags = 0

    def reach_targets(self):
        return []

    @property
    def instruction(self):
        return "patrol the aisle with the thermal camera and flag every server that is overheating"

    def close(self):
        super().close()
        if getattr(self, "_seg", None) is not None:
            self._seg.close()
            self._seg = None


def qy_(a):
    return np.array([np.cos(a / 2), 0, np.sin(a / 2), 0])
