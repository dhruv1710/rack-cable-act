"""Drive replacement: swap the failed hot-swap 3.5" drive of a 2U 12-bay server.

Chassis: 4 columns x 3 rows of guide-railed bays (1.5 mm vertical / 3 mm lateral clearance), 12 drive carriers.
Each carrier has the real hot-swap mechanism:
  * a release BUTTON (spring-loaded slide); pushing it >4 mm unlocks the lever,
  * a spring-loaded LEVER (hinge) that pops open ~83 deg and ejects the drive ~5 mm when unlocked,
  * a detent LATCH that holds a seated drive until the lever is opened.
The failed drive has an amber fault LED. A spare drive (lever open) lies on a stand beside the server.
Procedure: press the button on the failed drive, grip the open lever, pull the drive clear, pick up the spare,
slide it into the empty bay until seated, and push the lever shut.
Success: the spare is seated and latched (lever closed) in the failed drive's bay, the failed drive is out of
the chassis, and the other drives were not disturbed.
"""

import mujoco
import numpy as np

from ..common import latch
from ..common.arms import forward_rot
from ..common.base_env import TaskEnv
from ..common.geom import CYL, box, qz

XF, ZB = 0.600, 0.120  # chassis front face x, shelf height (a horizontal Panda hand needs >=12 cm palm clearance)
ROW_P, COL_P, N_ROW, N_COL = 0.0295, 0.108, 3, 4
L, W, H = 0.147, 0.100, 0.0235  # carrier length, width, height
DEPTH = 0.1505  # rail length
LEVER_OPEN = 1.45
SPARE_C = np.array([0.44, 0.31])
STAND_H = 0.135
SEAT_TOL = 0.0035


def bay_center(r, c):
    return np.array([XF + L / 2, (1.5 - c) * COL_P, ZB + ROW_P * (r + 0.5) + 0.001])


class DriveReplacement(TaskEnv):
    name = "drive_replacement"
    max_seconds = 150.0
    success_hold = 0.5
    overview = ((0.22, -0.50, 0.42), (0.55, 0.02, 0.12))

    # ------------------------------------------------------------------ build
    def build(self, spec):
        w = spec.worldbody
        # shelf + chassis rails
        box(w, "shelf", (XF + 0.12, 0, ZB / 2), (0.19, 0.25, ZB / 2), "dark_steel")
        top = ZB + N_ROW * ROW_P
        for r in range(N_ROW + 1):
            box(
                w,
                f"plate{r}",
                (XF + DEPTH / 2, 0, ZB + ROW_P * r + 0.001),
                (DEPTH / 2, 2 * COL_P + 0.002, 0.001),
                "alu",
                friction=(0.3, 0.005, 0.0005),
            )
        for k in range(N_COL + 1):
            box(
                w,
                f"div{k}",
                (XF + DEPTH / 2, (2 - k) * COL_P, ZB + N_ROW * ROW_P / 2 + 0.001),
                (DEPTH / 2, 0.001, N_ROW * ROW_P / 2 + 0.001),
                "alu",
                friction=(0.3, 0.005, 0.0005),
            )
        box(
            w,
            "stop",
            (XF + DEPTH + 0.0015, 0, ZB + N_ROW * ROW_P / 2 + 0.001),
            (0.0015, 2 * COL_P + 0.002, N_ROW * ROW_P / 2 + 0.001),
            "steel",
        )
        # chassis styling (visual): ears, frame, back, power panel
        for s in (-1, 1):
            box(
                w,
                f"ear{s}",
                (XF - 0.002, s * (2 * COL_P + 0.012), ZB + N_ROW * ROW_P / 2),
                (0.002, 0.012, N_ROW * ROW_P / 2 + 0.002),
                "steel",
                visual=True,
            )
            w.add_geom(
                name=f"ear_screw{s}",
                type=CYL,
                pos=(XF - 0.0042, s * (2 * COL_P + 0.012), ZB + N_ROW * ROW_P / 2),
                size=(0.004, 0.0008, 0),
                material="dark_steel",
                quat=np.array([0.7071, 0, 0.7071, 0]),
                contype=0,
                conaffinity=0,
            )
        box(
            w,
            "chassis_back",
            (XF + DEPTH + 0.06, 0, ZB + N_ROW * ROW_P / 2 + 0.001),
            (0.06, 2 * COL_P + 0.014, N_ROW * ROW_P / 2 + 0.0015),
            "charcoal",
            visual=True,
        )
        box(
            w,
            "chassis_cover",
            (XF + DEPTH + 0.06, 0, top + 0.004),
            (0.06, 2 * COL_P + 0.014, 0.003),
            "steel",
            visual=True,
        )
        for k in range(8):
            box(
                w,
                f"cover_vent{k}",
                (XF + DEPTH + 0.06, -0.2 + 0.057 * k, top + 0.0071),
                (0.045, 0.012, 0.0003),
                "black",
                visual=True,
            )
        box(
            w, "power_btn", (XF - 0.0035, 2 * COL_P + 0.012, ZB + 0.01), (0.002, 0.006, 0.006), "led_green", visual=True
        )
        # spare-drive stand: two rails
        for s in (-1, 1):
            box(
                w,
                f"stand_rail{s}",
                (SPARE_C[0] + 0.02, SPARE_C[1] + s * 0.043, STAND_H / 2),
                (0.085, 0.006, STAND_H / 2),
                "dark_steel",
                friction=(0.5, 0.01, 0.001),
            )
        box(
            w,
            "stand_base",
            (SPARE_C[0] + 0.02, SPARE_C[1], 0.002),
            (0.09, 0.055, 0.002),
            "dark_steel",
            contype=0,
            conaffinity=0,
        )
        box(w, "esd_mat", (0.43, -0.31, 0.0008), (0.10, 0.075, 0.0008), "green", visual=True)
        # drives
        self.drive_names = [f"drive{i}" for i in range(N_ROW * N_COL)] + ["drive_new"]
        for n in self.drive_names:
            self._build_drive(spec, w, n)

    def _build_drive(self, spec, w, n):
        b = w.add_body(name=n, pos=(0.3, 0.0, 0.3))
        b.add_freejoint(name=f"{n}_free")
        kw = {"friction": (0.35, 0.015, 0.0005), "density": 1800, "condim": 4}
        box(b, f"{n}_body", (0, 0, 0), (L / 2, W / 2, H / 2), "steel", **kw)
        box(
            b,
            f"{n}_bezel",
            (-L / 2 + 0.0015, 0, 0),
            (0.0015, W / 2 - 0.0005, H / 2 - 0.0005),
            "charcoal",
            contype=0,
            conaffinity=0,
        )
        for k in range(5):  # air-vent perforations (visual)
            box(
                b,
                f"{n}_perf{k}",
                (-L / 2 - 0.0006, 0.002 + k * 0.0055 - 0.011, 0.0),
                (0.0003, 0.0015, 0.0075),
                "black",
                visual=True,
            )
        box(
            b,
            f"{n}_led",
            (-L / 2 - 0.0006, -W / 2 + 0.010, H / 2 - 0.0045),
            (0.0004, 0.0025, 0.0025),
            "led_green",
            visual=True,
        )
        box(b, f"{n}_label", (0.02, 0, H / 2 + 0.0003), (0.045, 0.035, 0.0003), "white", visual=True)
        # lever: hinge at the +y edge of the front face
        lv = b.add_body(
            name=f"lever{n[5:]}" if n != "drive_new" else "lever_new", pos=(-L / 2 - 0.0016, W / 2 - 0.006, 0)
        )
        lv.add_joint(
            name=f"{n}_lever_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0, 0, -1),
            range=(0, 1.55),
            stiffness=0.02,
            springref=LEVER_OPEN,
            damping=0.004,
        )
        box(
            lv,
            f"{n}_lever_plate",
            (0, -0.030, 0),
            (0.0015, 0.030, H / 2 - 0.0012),
            "orange",
            density=1200,
            friction=(1.2, 0.01, 0.001),
        )
        box(
            lv,
            f"{n}_lever_grip",
            (-0.0015, -0.056, 0),
            (0.0025, 0.003, H / 2 - 0.0012),
            "orange",
            density=1200,
            friction=(1.2, 0.01, 0.001),
        )
        box(lv, f"{n}_lever_hook", (0.0025, -0.058, 0), (0.0015, 0.004, 0.004), "orange", contype=0, conaffinity=0)
        # release button
        bt = b.add_body(
            name=f"button{n[5:]}" if n != "drive_new" else "button_new", pos=(-L / 2 - 0.0016, -W / 2 + 0.020, 0.0)
        )
        bt.add_joint(
            name=f"{n}_button_slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=(1, 0, 0),
            range=(0, 0.007),
            stiffness=1500,
            damping=15.0,
        )  # ~6 N to reach the 4 mm unlock point, ~10 N full stroke
        box(bt, f"{n}_button", (-0.0015, 0, 0), (0.0025, 0.005, 0.0065), "blue", density=3000)
        latch.add_weld(spec, f"{n}_weld", n)
        eq = spec.add_equality(name=f"{n}_lever_lock", type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{n}_lever_hinge")
        eq.active = False
        eq.solref = (0.004, 1.0)

    # ------------------------------------------------------------------ bind
    def bind(self):
        m = self.m
        self.bid = {n: m.body(n).id for n in self.drive_names}
        self.qa = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.drive_names}
        self.da = {n: m.jnt_dofadr[m.joint(f"{n}_free").id] for n in self.drive_names}
        self.lq = {n: m.jnt_qposadr[m.joint(f"{n}_lever_hinge").id] for n in self.drive_names}
        self.bq = {n: m.jnt_qposadr[m.joint(f"{n}_button_slide").id] for n in self.drive_names}
        self.weld = {n: m.equality(f"{n}_weld").id for n in self.drive_names}
        self.lock = {n: m.equality(f"{n}_lever_lock").id for n in self.drive_names}
        self.led_geom = {n: m.geom(f"{n}_led").id for n in self.drive_names}
        self.seats = {(r, c): bay_center(r, c) for r in range(N_ROW) for c in range(N_COL)}
        self.bay_of = dict.fromkeys(self.drive_names)  # which bay a drive is latched in
        self.fail_bay = (1, 1)
        self.fail_drive = "drive5"
        self.eject = {}

    # ------------------------------------------------------------------ reset
    def randomize_state(self, rng):
        d, m = self.d, self.m
        r, c = (int(rng.integers(0, N_ROW)), int(rng.integers(0, N_COL))) if self.randomize else (1, 1)
        self.fail_bay, self.fail_drive = (r, c), f"drive{r * N_COL + c}"
        for i in range(N_ROW * N_COL):
            n = f"drive{i}"
            rr, cc = divmod(i, N_COL)
            a = self.qa[n]
            d.qpos[a : a + 7] = [*self.seats[(rr, cc)], 1, 0, 0, 0]
            d.eq_active[self.lock[n]] = 1
            m.geom_matid[self.led_geom[n]] = m.material("led_amber" if n == self.fail_drive else "led_green").id
        n = "drive_new"
        j = rng.uniform(-0.01, 0.01, 2) if self.randomize else np.zeros(2)
        yaw = rng.uniform(-0.12, 0.12) if self.randomize else 0.0
        a = self.qa[n]
        d.qpos[a : a + 7] = [SPARE_C[0] + j[0], SPARE_C[1] + j[1], STAND_H + H / 2 + 0.002, *qz(yaw)]
        d.qpos[self.lq[n]] = LEVER_OPEN
        d.eq_active[self.lock[n]] = 0
        m.geom_matid[self.led_geom[n]] = m.material("led_off").id
        self.bay_of = dict.fromkeys(self.drive_names)
        self.eject = {}
        d.xfrc_applied[:] = 0

    def after_settle(self):
        for i in range(N_ROW * N_COL):
            n = f"drive{i}"
            latch.engage(self.m, self.d, self.weld[n], self.bid[n])
            self.bay_of[n] = divmod(i, N_COL)

    # ------------------------------------------------------------------ mechanism
    def _seat_of(self, p):
        best, bd = None, 1e9
        for k, s in self.seats.items():
            e = np.abs(p - s)
            if e[0] < SEAT_TOL and e[1] < 0.004 and e[2] < 0.004 and e.sum() < bd:
                best, bd = k, e.sum()
        return best

    def on_substep(self):
        d = self.d
        for n in self.drive_names:
            lq, bq = d.qpos[self.lq[n]], d.qpos[self.bq[n]]
            locked = d.eq_active[self.lock[n]]
            if locked and bq > 0.004:
                d.eq_active[self.lock[n]] = 0  # button unlocks the lever -> it springs open
            elif (not locked) and lq < 0.05 and bq < 0.002:
                d.eq_active[self.lock[n]] = 1  # lever pushed shut -> locks
            if n in self.eject:  # ejector spring pushes the drive out ~6 mm
                x = d.xpos[self.bid[n]][0]
                if lq > 0.3 and x > self.eject[n] - 0.006:
                    d.xfrc_applied[self.bid[n], 0] = -2.5
                else:
                    d.xfrc_applied[self.bid[n], 0] = 0.0
                    del self.eject[n]
            if d.eq_active[self.weld[n]]:
                if lq > 0.15:  # lever opened: unlatch, ejector engages
                    latch.release(d, self.weld[n])
                    self.bay_of[n] = None
                    self.eject[n] = float(d.xpos[self.bid[n]][0])
            elif lq < 0.05:
                k = self._seat_of(d.xpos[self.bid[n]])
                if (
                    k is not None
                    and all(v != k for v in self.bay_of.values())
                    and abs(d.xmat[self.bid[n]].reshape(3, 3)[0, 0]) > 0.995
                ):
                    latch.engage(self.m, d, self.weld[n], self.bid[n])
                    self.bay_of[n] = k

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        d = self.d
        fb, fd = self.fail_bay, self.fail_drive
        new_in = self.bay_of.get("drive_new") == fb
        old_out = self.bay_of.get(fd) is None and np.linalg.norm(d.xpos[self.bid[fd]] - self.seats[fb]) > 0.03
        others = all(
            self.bay_of.get(f"drive{i}") == divmod(i, N_COL) for i in range(N_ROW * N_COL) if f"drive{i}" != fd
        )
        lever_ok = d.qpos[self.lq["drive_new"]] < 0.06
        success = new_in and old_out and others and lever_ok
        lever_open = d.qpos[self.lq[fd]] > 0.3 or old_out
        prog = (
            0.2 * lever_open
            + 0.3 * old_out
            + 0.2 * (self.bay_of.get("drive_new") is not None and new_in)
            + 0.3 * success
        )
        return {
            "success": bool(success),
            "progress": float(min(prog, 1.0)),
            "failed_bay": fb,
            "old_removed": bool(old_out),
            "new_seated": bool(new_in),
        }

    def oracle_solve(self):
        d, m = self.d, self.m
        fb, fd = self.fail_bay, self.fail_drive
        latch.release(d, self.weld[fd])
        self.bay_of[fd] = None
        d.qpos[self.lq[fd]] = LEVER_OPEN
        a = self.qa[fd]
        d.qpos[a : a + 7] = [0.43, -0.31, 0.02, 1, 0, 0, 0]
        d.qvel[self.da[fd] : self.da[fd] + 6] = 0
        a = self.qa["drive_new"]
        d.qpos[a : a + 7] = [*self.seats[fb], 1, 0, 0, 0]
        d.qvel[self.da["drive_new"] : self.da["drive_new"] + 6] = 0
        d.qpos[self.lq["drive_new"]] = 0.0
        mujoco.mj_forward(m, d)

    def reach_targets(self):
        r, c = self.fail_bay
        s = self.seats[(r, c)]
        return [
            ("failed drive button", 0, np.array([XF - 0.045, s[1] - 0.030, s[2]]), forward_rot(0.0, 0.0)),
            ("open lever grip", 0, np.array([XF - 0.045, s[1] - 0.045, s[2]]), forward_rot(0.0, 0.0)),
            (
                "spare drive",
                0,
                np.array([SPARE_C[0] - 0.02, SPARE_C[1], STAND_H + H / 2 + 0.002]),
                forward_rot(0.0, np.pi / 2),
            ),
            (
                "bay mouth (bottom row)",
                0,
                np.array([XF - 0.04, bay_center(0, 3)[1], bay_center(0, 3)[2]]),
                forward_rot(0.0, np.pi / 2),
            ),
            (
                "bay mouth (top row)",
                0,
                np.array([XF - 0.04, bay_center(2, 0)[1], bay_center(2, 0)[2]]),
                forward_rot(0.0, np.pi / 2),
            ),
        ]

    @property
    def instruction(self):
        r, c = self.fail_bay
        return f"replace the failed drive (amber LED) in row {r + 1}, column {c + 1} with the spare drive"
