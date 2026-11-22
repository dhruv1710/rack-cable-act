"""Force-driven mechanism tests (no arm): verify the custom latches/levers/clearances behave like the real thing."""

import sys

import mujoco
import numpy as np

import envs


def push(env, body, force, seconds):
    for _ in range(int(seconds / env.m.opt.timestep)):
        env.d.xfrc_applied[body, :3] = force
        mujoco.mj_step(env.m, env.d)
        env.on_substep()
    env.d.xfrc_applied[body] = 0


def drive_replacement():
    env = envs.make("drive_replacement", render=False)
    env.reset(seed=3)
    m, d = env.m, env.d
    fd, fb = env.fail_drive, env.fail_bay
    seat = env.seats[fb]
    idx = fd[5:]
    ok = {}
    ok["start: drive latched, lever locked"] = bool(d.eq_active[env.weld[fd]]) and bool(d.eq_active[env.lock[fd]])
    push(env, m.body(f"button{idx}").id, [3, 0, 0], 0.05)  # light touch: must NOT unlock (<4 mm)
    ok["light touch does not unlock"] = bool(d.eq_active[env.lock[fd]])
    push(env, m.body(f"button{idx}").id, [15, 0, 0], 0.3)  # firm press
    for _ in range(300):
        mujoco.mj_step(m, d)
        env.on_substep()
    lever = d.qpos[env.lq[fd]]
    x = d.xpos[env.bid[fd]][0] - seat[0]
    ok[f"button press opens lever (angle {lever:.2f} rad)"] = lever > 1.2
    ok[f"drive unlatched (ejector spring moved it {x * 1000:+.1f} mm)"] = not d.eq_active[env.weld[fd]]
    # pull the drive out by the lever grip
    lev = m.body(f"lever{idx}").id
    push(env, lev, [-6, 0, 0], 1.6)
    for _ in range(250):
        mujoco.mj_step(m, d)
        env.on_substep()
    x = d.xpos[env.bid[fd]][0] - seat[0]
    ok[f"drive slides out freely ({x * 1000:.0f} mm travel)"] = x < -0.20
    ok["neighbours stay latched"] = all(
        env.bay_of.get(f"drive{i}") == divmod(i, 4) for i in range(12) if f"drive{i}" != fd
    )
    # insert the spare: place at the bay mouth aligned, push in, then close the lever
    a = env.qa["drive_new"]
    d.qpos[a : a + 7] = [seat[0] - 0.09, seat[1] + 0.0008, seat[2] + 0.0006, 1, 0, 0, 0]
    d.qvel[env.da["drive_new"] : env.da["drive_new"] + 6] = 0
    mujoco.mj_forward(m, d)
    nb = env.bid["drive_new"]
    push(env, nb, [4, 0, 0], 1.4)
    for _ in range(300):
        mujoco.mj_step(m, d)
        env.on_substep()
    x = d.xpos[nb][0] - seat[0]
    ok[f"spare slides into the bay ({x * 1000:+.1f} mm from seat)"] = abs(x) < 0.004
    ok["not latched while lever is open"] = not d.eq_active[env.weld["drive_new"]]
    push(env, m.body("lever_new").id, [3, -6, 0], 1.0)  # swing the lever shut, pressing its face against the bezel
    for _ in range(300):
        mujoco.mj_step(m, d)
        env.on_substep()
    ok[f"closing the lever latches the drive (angle {d.qpos[env.lq['drive_new']]:.2f})"] = bool(
        d.eq_active[env.weld["drive_new"]]
    )
    ev = env.evaluate()
    ok["task success registered"] = ev["success"]
    return ok


def fuse_holder():
    env = envs.make("fuse_holder", render=False)
    env.reset(seed=4)
    m, d = env.m, env.d
    t = env.target
    old = env.holder_fuse[t]
    ok = {}
    ok["start: fuses latched, lids closed"] = (
        all(env.holder_of[env.holder_fuse[i]] == i for i in range(4)) and abs(d.qpos[env.lid_q[t]]) < 0.02
    )
    # open the lid by pushing the thumb tab toward the robot / down: apply force on lid body
    lid = m.body(f"lid{t}").id
    push(env, lid, [-0.4, 0, -0.1], 0.8)
    for _ in range(200):
        mujoco.mj_step(m, d)
        env.on_substep()
    ang = d.qpos[env.lid_q[t]]
    ok[f"lid pushed open ({ang:.2f} rad) and stays open"] = ang > 1.0
    # pull the blown fuse straight up: must resist a small force, release under >6 N
    fb = env.fuse_bid[old]
    push(env, fb, [0, 0, 0.0], 0.01)
    z0 = d.xpos[fb][2]
    push(env, fb, [0, 0, 1.5], 0.25)
    ok["fuse retained under 1.5 N tug"] = env.holder_of[old] == t and abs(d.xpos[fb][2] - z0) < 0.002
    push(env, fb, [0, 0, 12.0], 0.2)
    for _ in range(300):
        mujoco.mj_step(m, d)
        env.on_substep()
    ok["fuse releases under a firm pull"] = env.holder_of[old] is None and d.xpos[fb][2] > z0 + 0.02
    # seat a new fuse: lower it into the cradle (a bit offset) and push down
    n = "fuse_new0"
    a = env.fuse_qadr[n]
    seat = env.seat[t]
    d.qpos[a : a + 7] = [seat[0] + 0.0012, seat[1] - 0.0012, seat[2] + 0.025, 1, 0, 0, 0]
    d.qvel[m.jnt_dofadr[m.joint(f"{n}_free").id] :][:6] = 0
    mujoco.mj_forward(m, d)
    push(env, env.fuse_bid[n], [0, 0, -0.03], 0.6)  # gentle downward push (a hand, not a hammer)
    for _ in range(200):
        mujoco.mj_step(m, d)
        env.on_substep()
    ok[f"new fuse self-centres and latches (err {np.linalg.norm(d.xpos[env.fuse_bid[n]] - seat) * 1000:.1f} mm)"] = (
        env.holder_of[n] == t
    )
    # close the lid by pushing it back
    push(env, lid, [0.1, 0, 1.0], 1.2)  # lift the lid back up
    for _ in range(300):
        mujoco.mj_step(m, d)
        env.on_substep()
    ok[f"lid closes (angle {d.qpos[env.lid_q[t]]:.2f})"] = d.qpos[env.lid_q[t]] < 0.08
    ok["task success registered"] = env.evaluate()["success"]
    return ok


TESTS = {"drive_replacement": drive_replacement, "fuse_holder": fuse_holder}

if __name__ == "__main__":
    allok = True
    for n in sys.argv[1:] or TESTS:
        print(f"== {n}")
        for k, v in TESTS[n]().items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
            allok &= bool(v)
    sys.exit(0 if allok else 1)
