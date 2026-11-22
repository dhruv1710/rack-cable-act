"""Scripted reference policies for tasks beyond the ones in envs/tests/.

Status: bottle_binning is partial (about 3 of 8 seeds). It is not used for the published videos.

Each reads privileged simulator state (object poses), solves inverse kinematics and drives the arm through
``Driver``. They exist to show each task is solvable, to produce demonstration videos and, later, demonstration data.
"""

from __future__ import annotations

import numpy as np

from .common.arms import down_rot, solve_ik
from .tasks.bottle_binning import CRATE_C
from .tests.scripted import Driver


def gentle_grasp(drv, seconds=1.2, hold=0.4):
    """Close the gripper gradually. An instant close squirts small round or hex parts out of the fingers."""
    n = int(seconds * 25)
    for i in range(1, n + 1):
        drv.hold(1 / 25, grip=1.0 - i / n)
    drv.hold(hold, grip=0.0)


def reachable_R(env, drv, positions, yaw, extra=()):
    """The hand is symmetric under a 180 degree turn: return the first yaw (of yaw, yaw+pi, yaw-pi) for which IK
    reaches every position, or None. Executing an unsolved IK result throws objects around, so callers skip instead."""
    for y in (yaw, yaw + np.pi, yaw - np.pi, *(yaw + e for e in extra)):
        R = down_rot(y)
        q0 = drv.joint_state()
        if all(solve_ik(env.m, env.d, drv.arm, pos, R, q_init=q0, iters=300, restarts=40)[1] for pos in positions):
            return R
    return None


COMP_HALF = 0.05  # half-width of a compartment's landing zone
CRATE_SLOTS = [(-0.07, -0.05), (0.07, -0.05), (-0.07, 0.05), (0.07, 0.05)]


def _on_belt(env, b):
    p = env.d.xpos[b]
    return p[2] > 0.0 and abs(p[1] - CRATE_C[1]) > 0.2 and abs(p[0]) < 1.0


def _clearance(d, b, pool):
    """Distance from bottle b to its nearest neighbour in pool."""
    return min((np.linalg.norm(d.xpos[b][:2] - d.xpos[o][:2]) for o in pool if o != b), default=9.0)


def bottle_binning(env, verbose=False, tries_per_bottle=2):
    """Pick the most isolated bottle left on the belt (standing: pinch the neck, lying: grasp the body) and drop it in
    its own crate slot. The open hand is only ~1 cm wider than a bottle on each side, so picking in order of
    clearance avoids shoving the neighbours. A bottle that did not end up in the crate is retried once."""
    d = env.d
    drv = Driver(env)
    tries = dict.fromkeys(env.bid, 0)
    done = 0
    for _ in range(3 * len(env.bid)):
        left = [b for b in env.bid if _on_belt(env, b) and tries[b] < tries_per_bottle]
        if not left:
            break
        b = max(left, key=lambda c, pool=left: _clearance(d, c, pool))
        tries[b] += 1
        p = d.xpos[b].copy()
        up = abs(d.xmat[b].reshape(3, 3)[2, 2])
        if up > 0.9:  # standing
            yaw, pre, grasp = 0.0, np.array([p[0], p[1], 0.32]), np.array([p[0], p[1], 0.168])
        else:  # lying: close across the body at its centre, hand yawed to the bottle axis
            ax = d.xmat[b].reshape(3, 3)[:, 2]
            yaw = float(np.arctan2(ax[1], ax[0]))
            pre, grasp = p + np.array([0, 0, 0.12]), p.copy()
        sx, sy = CRATE_SLOTS[done % len(CRATE_SLOTS)]
        tx, ty = CRATE_C[0] + sx, CRATE_C[1] + sy
        R = reachable_R(env, drv, [pre, grasp, np.array([tx, ty, 0.40])], yaw)
        if R is None:
            if verbose:
                print(f"  bottle at {np.round(p, 2)}: unreachable")
            continue
        drv.move(pre, R, 1.8, grip=1.0)
        drv.hold(0.5)
        drv.move_linear(grasp, R, 2.2)
        drv.hold(0.5)
        gentle_grasp(drv)
        drv.move_linear(grasp + np.array([0, 0, 0.22]), R, 1.2)
        drv.move(np.array([tx, ty, 0.40]), R, 2.0)
        drv.move_linear(np.array([tx, ty, 0.36]), R, 0.8)
        drv.release(0.7)
        drv.move_linear(np.array([tx, ty, 0.42]), R, 0.8)
        if env._in_crate(d.xpos[b]):
            done += 1
        elif verbose:
            print(f"  bottle at {np.round(p, 2)} not in the crate after the pick")
    drv.hold(2.0)
    return env.evaluate()


# --------------------------------------------------------------------------------------------------- thermal patrol
def thermal_patrol(env, thr=50.0, verbose=False):
    """Perception-guided patrol: stop at each rack, look at both rack rows at several tilts, and when the thermal
    image shows a hot blob, centre it with pan/tilt feedback and raise the flag. Uses the thermal image only
    (temperatures, not the hot-server list); warm decoys (< 48 C) stay below the threshold."""
    from .tasks.thermal_patrol import RACK_W, X0

    m, d = env.m, env.d
    H, W = 120, 160
    f = (H / 2) / np.tan(np.radians(m.cam_fovy[env.thermal_cam]) / 2)  # focal length in pixels
    st = {"pan": 0.0, "tilt": 0.0}

    def step(v=0.0, w=0.0, flag=0.0, n=1):
        a = np.array([v, w, st["pan"], st["tilt"], flag], dtype=np.float32)
        for _ in range(n):
            env.step(a)

    def base():
        return d.qpos[env.qa["base_x"]], d.qpos[env.qa["base_yaw"]]

    def face_forward():
        for _ in range(200):
            _, yaw = base()
            if abs(yaw) < 0.01:
                break
            step(0.0, float(np.clip(-3.0 * yaw, -0.8, 0.8)))
        step(n=10)

    def drive_to(x_goal):
        for _ in range(2000):
            x, yaw = base()
            err = x_goal - x
            if abs(err) < 0.01:
                break
            step(float(np.clip(1.2 * err, -0.5, 0.5)), float(np.clip(-3.0 * yaw, -0.5, 0.5)))
        step(n=10)

    def look(pan, tilt, settle=25):
        st["pan"], st["tilt"] = pan, tilt
        step(n=settle)

    def square_on(rack_x):
        """Move the base so the camera (offset from the base origin, and swung sideways by the pan) sits at rack_x."""
        for _ in range(3):
            dx = rack_x - env._cam()[0][0]
            if abs(dx) < 0.006:
                break
            drive_to(base()[0] + dx)

    def blob():
        """Bounding-box middle of the hottest region of the thermal image (neighbours of a hot server pick up some
        heat, so only pixels within 6 C of the peak count). The box middle is robust to the blur and heat gradient."""
        T = env.thermal_temperature()
        peak = float(T.max())
        if peak < thr:
            return None
        hot = max(thr, peak - 6.0) <= T
        if hot.sum() < 4:
            return None
        ys, xs = np.nonzero(hot)
        return (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, int(hot.sum())

    def centre_and_flag():
        """Servo pan and tilt until the blob is centred, then raise the flag."""
        sign = {"pan": -1.0, "tilt": 1.0}  # image axes vs joint axes, corrected on the fly if the error grows
        prev = None
        for _ in range(30):
            b = blob()
            if b is None:
                return False
            cx, cy, _n = b
            ex, ey = np.arctan((cx - W / 2) / f), np.arctan((cy - H / 2) / f)
            err = float(np.hypot(ex, ey))
            if err < 0.015:
                step(flag=1.0, n=2)
                step(flag=0.0, n=3)
                return True
            if prev is not None and err > prev[0] * 1.15:
                sign[prev[1]] = -sign[prev[1]]
            axis = "pan" if abs(ex) >= abs(ey) else "tilt"
            lim = 1.7 if axis == "pan" else 0.9
            st[axis] = float(np.clip(st[axis] + sign[axis] * (ex if axis == "pan" else ey) * 0.8, -lim, lim))
            prev = (err, axis)
            step(n=12)
        return False

    face_forward()
    n_flags = 0
    k0 = max(0, int(np.ceil((base()[0] - X0) / RACK_W)))
    for k in range(k0, 8):
        rack_x = X0 + RACK_W * k
        drive_to(rack_x)
        for pan in (np.pi / 2, -np.pi / 2):
            look(pan, -0.75)
            square_on(rack_x)
            for tilt in (-0.75, -0.25, 0.2, 0.6):
                look(pan, tilt)
                if blob() is not None:
                    n_flags += centre_and_flag()
                    if verbose:
                        print(f"  rack {k} pan {pan:+.2f} tilt {tilt:+.2f}: flagged ({n_flags} so far)")
    look(0.0, 0.0, 10)
    step(n=25)
    return env.evaluate()


# ------------------------------------------------------------------------------------------------- parts sorting
def _clearance_xy(pos, a, pool):
    return min((np.linalg.norm(pos[a][:2] - pos[b][:2]) for b in pool if b != a), default=9.0)


def _blockers(p, yaw, others, reach=0.058, half_w=0.022):
    """Neighbours that the open fingers would hit when descending at p with the hand yawed to `yaw` (the fingers
    close along (-sin yaw, cos yaw) and open about 4 cm either side of the part)."""
    u, v = np.array([-np.sin(yaw), np.cos(yaw)]), np.array([np.cos(yaw), np.sin(yaw)])
    n = 0
    for q in others:
        dq = q[:2] - p[:2]
        n += abs(dq @ u) < reach and abs(dq @ v) < half_w
    return n


def _grip_options(t, M):
    """Candidate hand yaws (and the grip point offset) for a part, best first by the part's own geometry."""
    ax = M[:, 2]
    if t == "nut":
        y0 = float(np.arctan2(M[1, 0], M[0, 0]))
        return [y0 + k * np.pi / 3 for k in range(3)]  # any pair of opposite flats
    if abs(ax[2]) < 0.5:  # lying: along the axis onto the flat ends (a cylinder rolls away from the first finger)
        y0 = float(np.arctan2(ax[1], ax[0]))
        return [y0 + np.pi / 2, y0] if t == "standoff" else [y0]
    return [k * np.pi / 8 for k in range(4)]  # standing round part: any yaw


def parts_sorting(env, verbose=False, attempts=3):
    """Pick a loose part whose grip is not blocked by neighbours, carry it to the compartment whose label matches its
    type, drop it. A part that was not lifted, or did not land in its compartment, goes back on the list."""
    d = env.d
    drv = Driver(env)
    slot = dict.fromkeys(range(3), 0)
    todo = list(env.part_names)
    tries = dict.fromkeys(todo, 0)
    while todo:
        pos = {n: d.xpos[env.part_bid[n]].copy() for n in todo}
        plan = {}
        for n in todo:
            opts = _grip_options(env.part_type[n], d.xmat[env.part_bid[n]].reshape(3, 3))
            others = [pos[o] for o in todo if o != n]
            plan[n] = min(((_blockers(pos[n], y, others), i, y) for i, y in enumerate(opts)))
        # least blocked first, then most isolated
        n = min(todo, key=lambda a: (plan[a][0], -_clearance_xy(pos, a, todo)))
        todo.remove(n)
        tries[n] += 1
        p = pos[n]
        if p[2] < -0.05:  # fell off the bench
            continue
        t = env.part_type[n]
        yaw = plan[n][2]
        gz = p[2] if t == "bolt" else max(p[2], 0.009)
        c = env.mapping.index(t)
        cc = env.comp_center(c)
        k = min(slot[c], 2)
        tgt = np.array([cc[0] - 0.035 + 0.035 * k, cc[1]])
        pre, grasp, over = np.array([p[0], p[1], 0.12]), np.array([p[0], p[1], gz]), np.array([tgt[0], tgt[1], 0.12])
        R = reachable_R(env, drv, [pre, grasp, over], yaw)
        if R is None:
            if verbose:
                print(f"  skip {n}: unreachable")
            continue
        drv.move(pre, R, 1.6, grip=1.0)
        drv.hold(0.4)
        drv.move_linear(grasp, R, 1.4)
        drv.hold(0.3)
        gentle_grasp(drv)
        drv.move_linear(pre, R, 1.2)
        lifted = d.xpos[env.part_bid[n]][2] > 0.06
        if lifted:
            drv.move(over, R, 2.2)
            drv.move_linear(np.array([tgt[0], tgt[1], 0.065]), R, 0.8)
        drv.release(0.5)
        drv.move_linear(over if lifted else pre, R, 0.8)
        q = d.xpos[env.part_bid[n]]
        landed = abs(q[0] - cc[0]) < 0.06 and abs(q[1] - cc[1]) < COMP_HALF and q[2] < 0.04
        if landed:
            slot[c] += 1
        elif tries[n] < attempts:
            todo.append(n)
            if verbose:
                print(f"  retry {n} (lifted={lifted})")
    drv.hold(1.5)
    return env.evaluate()
