"""Soft retention latch: a world-weld that locks a body at its current pose and releases under load.

Models spring-clip / detent retention (fuse in its clip, drive carrier in its bay). Call `engage` when
the body is seated; `check_release` every physics step frees it when the pulling force exceeds `release_n`.
"""

import mujoco
import numpy as np


def add_weld(spec, name, body_name, solref=(0.004, 1.0)):
    eq = spec.add_equality(
        name=name, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=body_name, name2="world"
    )
    eq.active = False
    eq.solref = solref
    return eq


def engage(m, d, eq_id, body_id):
    """Weld `body_id` to the world at its present pose.

    The weld point (anchor, in world coordinates since body2 = world) is the body's own origin, so the lock
    pivots about the body instead of the world origin (an origin-anchored weld lets light bodies swing about
    a point metres away). Orientation target is tried in both quaternion conventions; the lower violation wins.
    """
    p, q = d.xpos[body_id].copy(), d.xquat[body_id].copy()
    qi = np.zeros(4)
    mujoco.mju_negQuat(qi, q)
    best = None
    for relq in (qi, q):
        m.eq_data[eq_id, :] = 0.0
        m.eq_data[eq_id, 0:3] = p
        m.eq_data[eq_id, 6:10] = relq
        m.eq_data[eq_id, 10] = 1.0
        d.eq_active[eq_id] = 1
        mujoco.mj_forward(m, d)
        sel = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq_id)
        err = np.abs(d.efc_pos[sel]).max() if sel.any() else np.inf
        if best is None or err < best[0]:
            best = (err, relq.copy())
    m.eq_data[eq_id, 6:10] = best[1]
    d.eq_active[eq_id] = 1
    mujoco.mj_forward(m, d)


def release(d, eq_id):
    d.eq_active[eq_id] = 0


def holding_force(m, d, eq_id) -> float:
    sel = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq_id)
    if not sel.any():
        return 0.0
    return float(np.linalg.norm(d.efc_force[sel][:3]))
