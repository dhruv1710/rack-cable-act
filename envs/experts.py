"""Scripted reference policies for tasks beyond the ones in envs/tests/.

Status: bottle_binning is partial (about 3 of 8 seeds). It is not used for the published videos.

Each reads privileged simulator state (object poses), solves inverse kinematics and drives the arm through
``Driver``. They exist to show each task is solvable, to produce demonstration videos and, later, demonstration data.
"""

from __future__ import annotations

import itertools

import numpy as np

from .common.arms import Q_HOME, down_rot, solve_ik
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


# ------------------------------------------------------------------------------------------------- path planning
def plan_joint_path(drv, poses, n_seeds=24, small=0.5, seed=0, allow_flip=True, big_steps="any"):
    """Joint-space path through a list of Cartesian waypoints (position, rotation).

    A waypoint has several IK solutions on different arm configurations, and a warm start from the previous waypoint
    often stays on one that dead-ends. So: collect distinct IK solutions per waypoint (warm start plus random seeds,
    including the equivalent pose with the hand turned 180 degrees about its approach axis, which the symmetric gripper
    allows), then pick the chain from the current joint state with the fewest large joint steps and, among those, the
    smallest steps (dynamic programming). big_steps limits where a large step may occur: "any", "first" (only from the
    current state to the first waypoint, e.g. a wrist flip done high above everything) or "none".
    Returns (list of q, number of large steps, largest step); the list is None if no chain satisfies big_steps."""
    env = drv.env
    rng = np.random.default_rng(seed)
    q_now = drv.joint_state()
    flip = np.diag([-1.0, -1.0, 1.0])  # turn about the tool z axis (approach direction)
    layers, prev_best = [], q_now
    for pos, R in poses:
        sols = []
        for Rt in (R, R @ flip) if allow_flip else (R,):
            q, ok = solve_ik(env.m, env.d, drv.arm, pos, Rt, q_init=prev_best, iters=250)
            if ok:
                sols.append(q)
            tries = 0
            while tries < 3 * n_seeds and (tries < n_seeds or len(sols) < 4):  # thin regions need many more seeds
                tries += 1
                q0 = np.clip(Q_HOME + rng.normal(0, 0.7, 7), drv.arm.lo + 0.05, drv.arm.hi - 0.05)
                q, ok = solve_ik(env.m, env.d, drv.arm, pos, Rt, q_init=q0, iters=250)
                if ok and all(np.abs(q - o).max() > 0.05 for o in sols):
                    sols.append(q)
        if not sols:
            return None, 99, float("inf")
        layers.append(sols)
        prev_best = min(sols, key=lambda c: np.abs(c - prev_best).max())

    def cost(a, b_, allowed):
        st = np.abs(a - b_).max()
        if st <= small:
            return st
        return 10.0 + st if allowed else 1e6  # a large step costs far more than any run of small ones

    acc = [np.array([cost(c, q_now, big_steps in ("any", "first")) for c in layers[0]])]
    back = []
    for i in range(1, len(layers)):
        step = np.array([[cost(c, p_, big_steps == "any") for p_ in layers[i - 1]] for c in layers[i]])
        tot = step + acc[-1][None, :]
        arg = tot.argmin(1)
        acc.append(tot[np.arange(len(layers[i])), arg])
        back.append(arg)
    j = int(acc[-1].argmin())
    final_cost = float(acc[-1][j])
    chain = [layers[-1][j]]
    for i in range(len(layers) - 1, 0, -1):
        j = int(back[i - 1][j])
        chain.append(layers[i - 1][j])
    chain.reverse()
    if final_cost >= 1e6:
        return None, 99, float("inf")
    steps = [np.abs(chain[0] - q_now).max()] + [np.abs(chain[i] - chain[i - 1]).max() for i in range(1, len(chain))]
    return chain, sum(st > small for st in steps), float(max(steps))


def run_joint_path(drv, qs, seconds, grip=None, max_speed=1.2):
    """Execute a joint path (list of q) from the current state. Each segment takes its share of `seconds`, but never
    less than the time the largest joint step needs at max_speed (rad/s), so a wrist flip is done slowly."""
    if grip is not None:
        drv.grip = grip
    qs = [drv.joint_state(), *qs]
    base = seconds / max(len(qs) - 1, 1)
    for a, b_ in itertools.pairwise(qs):
        n = max(1, int(max(base, np.abs(b_ - a).max() / max_speed) * 50))
        for i in range(1, n + 1):
            drv.env.step(drv._act(a + (b_ - a) * (i / n)))


def carry_path(drv, start, goal, R, seconds, n=30, **kw):
    """Straight-line Cartesian carry executed on a planned joint path. Returns (large steps, largest step)."""
    poses = [(start + (goal - start) * s, R) for s in np.linspace(0, 1, n)[1:]]
    qs, n_big, worst = plan_joint_path(drv, poses, **kw)
    if qs is not None:
        run_joint_path(drv, qs, seconds)
    return n_big, worst


# ------------------------------------------------------------------------------------------------- plate racking
def plate_racking(env, verbose=False):
    """Grip the plate by its near rim with the hand horizontal, carry it flat and high across to the target slot with
    the hand yawed (straight ahead, |y| < 0.07 m, the wrist has no IK solution), square the hand up over the slot, roll
    the wrist to stand the plate on edge, measure the plate relative to the hand and lower it between the tines."""
    from .common.arms import forward_rot
    from .tasks.plate_racking import BASE_T, PLATE_R, RACK_C

    d, drv = env.d, Driver(env)
    p0 = d.xpos[env.plate].copy()
    flat = forward_rot(0.0, np.pi / 2)
    rim = np.array([p0[0] - PLATE_R + 0.012, p0[1], p0[2]])
    drv.grip = 1.0
    stage = rim + np.array([-0.16, 0, 0.12])
    for target, secs in ((stage, 2.5), (rim + np.array([-0.08, 0, 0.0]), 1.5), (rim, 1.5)):
        carry_path(drv, drv.arm.tcp_pos(d).copy(), target, flat, secs, n=14)
    drv.hold(0.5)
    gentle_grasp(drv, seconds=1.0)
    carry_path(drv, drv.arm.tcp_pos(d).copy(), rim + np.array([0, 0, 0.12]), flat, 1.4, n=8)

    def held():
        return np.linalg.norm(d.xpos[env.plate] - drv.arm.tcp_pos(d)) < 0.13

    sy = env.slot_y(env.target)
    plate_dx = d.xpos[env.plate][0] - drv.arm.tcp_pos(d)[0]  # plate centre ahead of the hand
    top = np.array([RACK_C[0] - plate_dx, sy, 0.34])
    # The goal region may only be reachable on the other arm configuration. The symmetric hand allows a 180 degree wrist
    # flip: plan "lift high, then carry" as one chain in which the only large step is the first one (the flip, done high
    # above everything with the plate still gripped).
    here = drv.arm.tcp_pos(d).copy()
    hi = here + np.array([0.0, 0.0, 0.14])
    poses = [(hi, flat)] + [(hi + (top - hi) * s_, flat) for s_ in np.linspace(0, 1, 30)[1:]]
    qs, n_big, worst = plan_joint_path(drv, poses, big_steps="first")
    if qs is None:
        return env.evaluate(), None
    run_joint_path(drv, qs, 7.0)
    if verbose:
        print(f"  carry: {n_big} large joint step(s), largest {worst:.2f} rad")
    if verbose:
        print("  over slot, flat: plate", np.round(d.xpos[env.plate], 3), "held", held())
    # stand the plate on edge: roll the wrist 90 degrees in place, either way, planned as one continuous joint path
    pos = drv.arm.tcp_pos(d).copy()
    best = None
    for end in (0.0, np.pi):
        poses = [(pos, forward_rot(0.0, r)) for r in np.linspace(np.pi / 2, end, 8)[1:]]
        qs, n_big, worst = plan_joint_path(drv, poses, big_steps="none")
        if qs is not None and (best is None or (n_big, worst) < best[:2]):
            best = (n_big, worst, qs, end)
    if best is None:
        return env.evaluate(), None
    run_joint_path(drv, best[2], 3.0)
    drv.hold(0.6)
    vert = forward_rot(0.0, best[3])
    if verbose:
        print(f"  rolled to edge: {best[0]} large step(s), plate {np.round(d.xpos[env.plate], 3)} held {held()}")
    off = d.xpos[env.plate] - drv.arm.tcp_pos(d)  # plate centre relative to the hand
    goal_xy = np.array([RACK_C[0], sy])
    z_hover, z_final = BASE_T + PLATE_R + 0.07, BASE_T + PLATE_R + 0.0015

    def go(target, seconds, n=14, big="none"):
        return carry_path(drv, drv.arm.tcp_pos(d).copy(), target, vert, seconds, n=n, big_steps=big)

    tcp = np.array([*(goal_xy - off[:2]), z_hover - off[2]])
    go(tcp, 3.0)
    drv.hold(0.6)
    for _ in range(3):  # centre the plate on the slot within ~1 mm before it goes between the tines
        tcp[:2] += goal_xy - d.xpos[env.plate][:2]
        go(tcp, 1.0, n=4)
        drv.hold(0.5)
    if verbose:
        print("  on edge over slot: plate", np.round(d.xpos[env.plate], 4), "goal", np.round(goal_xy, 4))
    for z in np.linspace(z_hover, z_final, 7)[1:]:
        tcp[:2] += 0.5 * (goal_xy - d.xpos[env.plate][:2])
        tcp[2] = z - off[2]
        go(tcp, 0.9, n=3)
    drv.hold(0.6)
    drv.release(0.8)
    for off in ((-0.05, 0, 0), (-0.03, 0, 0.03), (0, 0, 0.06), (-0.05, 0, 0.05), (0, 0, 0.12), (-0.08, 0, 0.08)):
        if not env._held():
            break  # fingers are clear of the plate
        go(drv.arm.tcp_pos(d) + np.array(off), 1.5, n=5, big="first")  # plate released: one bigger step is safe
        drv.hold(0.3)
    drv.hold(1.5)
    return env.evaluate(), None


# --------------------------------------------------------------------------------------------- planned Cartesian moves
def _log_rot(R):
    """Rotation vector of a rotation matrix (angle < ~170 degrees)."""
    ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return w * ang


def _exp_rot(w):
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def pmove(drv, pos, R, seconds, n=12, big="none"):
    """Move the TCP in a straight line to `pos` while turning to `R`, on a planned continuous joint path (see
    plan_joint_path). big: where a large joint step is allowed ("none", "first", "any"). Returns True if it moved."""
    d, arm = drv.env.d, drv.arm
    p0, R0 = arm.tcp_pos(d).copy(), d.site_xmat[arm.tcp].reshape(3, 3).copy()
    w = _log_rot(R0.T @ R)
    poses = [(p0 + (np.asarray(pos) - p0) * s, R0 @ _exp_rot(w * s)) for s in np.linspace(0, 1, n)[1:]]
    qs, _nb, _worst = plan_joint_path(drv, poses, big_steps=big)
    if qs is None:
        return False
    run_joint_path(drv, qs, seconds)
    return True


def pmove_safe(drv, pos, R, seconds, n=8):
    """pmove that tries a continuous path first and only then one with a large first step (used for retreats)."""
    return pmove(drv, pos, R, seconds, n=n) or pmove(drv, pos, R, seconds, n=n, big="first")


def pmove_via(drv, points, R, seconds, n_per=8, big="first"):
    """Move through several points with the hand held at R: the arm jumps to the first point (where a large joint step
    is allowed, e.g. to turn the hand into place), then follows straight legs through the rest. One joint-space chain
    for the whole route, chosen globally so the first pose is the configuration that connects to the later ones.
    Returns True if it moved."""
    poses = [(np.asarray(points[0]), R)]
    for a, b in itertools.pairwise(points):
        poses += [(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s_, R) for s_ in np.linspace(0, 1, n_per)[1:]]
    qs, _nb, _worst = plan_joint_path(drv, poses, big_steps=big)
    if qs is None:
        return False
    run_joint_path(drv, qs, seconds)
    return True


# ------------------------------------------------------------------------------------------------------- handoff
def object_handoff(env, verbose=False):
    """Left arm takes the bar by its top (a vertical hand's palm stops the fingertips ~4.5 cm below a bar top), squares
    it to the world axes while carrying, the right arm takes it from the other side 12.5 cm above its bottom, the left
    lets go, and the right sets the bar on the pad. The right hand is horizontal, so its TCP must be >= 12 cm above the
    bench at the pad: that is why the bar is 20 cm tall."""
    from .common.arms import forward_rot
    from .tasks.object_handoff import BAR_HALF, PAD_C

    d = env.d
    L, Rr = Driver(env, 0), Driver(env, 1)
    bar = env.bar
    p = d.xpos[bar].copy()
    yaw = float(np.arctan2(d.xmat[bar].reshape(3, 3)[1, 0], d.xmat[bar].reshape(3, 3)[0, 0]))
    z_top = p[2] + BAR_HALF[2]
    z_grip_l = z_top - 0.03
    hp = np.array([0.45, 0.09, 0.42])  # off the right arm's sagittal plane (y = 0): IK has no solution there
    pre_l, grasp_l = np.array([p[0], p[1], z_grip_l + 0.16]), np.array([p[0], p[1], z_grip_l])
    RL = reachable_R(env, L, [pre_l, grasp_l], yaw, extra=(np.pi / 2,))
    if RL is None:
        return env.evaluate()

    def log(tag):
        if verbose:
            print(f"  {tag:28s} stage={env.stage} bar={np.round(d.xpos[bar], 3)}")

    pmove(L, pre_l, RL, 3.0, n=14, big="first")
    L.hold(0.4, grip=1.0)
    pmove(L, grasp_l, RL, 2.0, n=10)
    L.hold(0.4)
    gentle_grasp(L, seconds=1.0)
    pmove(L, pre_l, RL, 1.5, n=8)
    log("left grasped and lifted")
    Rh = reachable_R(env, L, [hp], 0.0, extra=(np.pi / 2, -np.pi / 2))
    if Rh is None:
        return env.evaluate()
    pmove(L, hp, Rh, 4.0, n=18, big="first")
    L.hold(0.6)
    log("left at handoff point")
    RR = forward_rot(np.pi, 0.0)
    b = d.xpos[bar].copy()
    grasp_r = np.array([b[0], b[1], b[2] - BAR_HALF[2] + 0.125])
    pmove_via(Rr, [grasp_r + np.array([0.20, 0.0, 0.06]), grasp_r + np.array([0.08, 0.0, 0.0]), grasp_r], RR, 7.0)
    Rr.hold(0.4)
    gentle_grasp(Rr, seconds=1.0)
    log("right grasped (both hold)")
    L.release(0.6)
    log("left released")
    pmove(L, hp + np.array([-0.12, 0.0, 0.0]), Rh, 1.5, n=8)
    off = Rr.arm.tcp_pos(d) - d.xpos[bar]
    goal = np.array([PAD_C[0], PAD_C[1], BAR_HALF[2] + 0.004]) + off
    pmove(Rr, goal + np.array([0.0, 0.0, 0.12]), RR, 3.5, n=14)
    pmove(Rr, goal, RR, 1.8, n=8)
    Rr.hold(0.4)
    Rr.hold(1.2, grip=0.58)  # loosen just enough for the bar to slip down and seat flat on the pad under its weight
    Rr.release(0.8)
    log("right released on pad")
    pmove(Rr, goal + np.array([0.10, 0.0, 0.05]), RR, 1.5, n=6, big="first")
    Rr.hold(1.0)
    return env.evaluate()


# ------------------------------------------------------------------------------------------------------ pouring
def pouring_beads(env, verbose=False, tilt_max=2.35, back=0.03, z_over=0.15):
    """Grip the cup, carry it low over the bowl, tilt it about the world y axis until it is empty, set it back down on
    the table where it started and let go. Every move is a planned continuous joint path (see plan_joint_path)."""
    from .tasks.pouring_beads import rot_y

    d = env.d
    drv = Driver(env)
    R0 = down_rot(0.0)
    c = d.xpos[env.cup].copy()
    pmove(drv, np.array([c[0], c[1], 0.30]), R0, 3.0, n=12, big="first")
    drv.hold(0.4, grip=1.0)
    pmove(drv, np.array([c[0], c[1], 0.055]), R0, 2.0, n=10)
    drv.hold(0.4)
    gentle_grasp(drv, seconds=1.0)
    pmove(drv, np.array([c[0], c[1], 0.20]), R0, 1.5, n=8)
    over = np.array([env.bowl_c[0] - back, env.bowl_c[1], z_over])
    pmove(drv, np.array([over[0], over[1], 0.20]), R0, 2.5, n=12)
    pmove(drv, over, R0, 1.2, n=6)
    for k in range(1, 9):  # slow tilt about the TCP
        pmove(drv, over, rot_y(tilt_max * k / 8) @ R0, 0.6, n=3)
    for _ in range(20):  # wait until the cup is empty (<= 2 beads left)
        drv.hold(0.5)
        if env.beads_in_cup() <= 2:
            break
    drv.hold(1.0)
    n_in, n_sp = env._counts()
    if verbose:
        print(f"  after pour: in bowl {n_in}/100, spilled {n_sp}")
    Rt = rot_y(tilt_max) @ R0
    up = np.array([over[0] - 0.02, over[1], 0.27])
    pmove(drv, up, Rt, 1.8, n=8)  # rise clear of the bowl while still tipped
    for k in range(7, -1, -1):  # untilt in the air
        pmove(drv, up, rot_y(tilt_max * k / 8) @ R0, 0.5, n=3)
    pmove(drv, np.array([c[0], c[1], 0.20]), R0, 2.0, n=12)
    pmove(drv, np.array([c[0], c[1], 0.058]), R0, 1.5, n=8)
    drv.hold(0.5)
    drv.hold(0.8, grip=0.6)  # loosen so the cup seats on the table before letting go
    drv.release(0.6)
    pmove_safe(drv, np.array([c[0], c[1], 0.25]), R0, 1.5, n=8)
    drv.hold(1.0)
    return env.evaluate()
