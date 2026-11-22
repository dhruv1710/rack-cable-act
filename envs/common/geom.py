"""Small geometry helpers on top of MjSpec."""

import mujoco
import numpy as np

BOX, CYL, CAP, SPH = (
    mujoco.mjtGeom.mjGEOM_BOX,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_SPHERE,
)


def qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def qaxis(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(ang / 2), axis * np.sin(ang / 2)]


def qz(ang):
    return qaxis((0, 0, 1), ang)


def qy(ang):
    return qaxis((0, 1, 0), ang)


def qx(ang):
    return qaxis((1, 0, 0), ang)


def box(body, name, pos, half, material="steel", quat=None, visual=False, **kw):
    if visual:
        kw.update(contype=0, conaffinity=0)
    g = body.add_geom(name=name, type=BOX, pos=pos, size=half, material=material, **kw)
    if quat is not None:
        g.quat = quat
    return g


def cyl(body, name, pos, r, half_h, material="steel", quat=None, visual=False, **kw):
    if visual:
        kw.update(contype=0, conaffinity=0)
    g = body.add_geom(name=name, type=CYL, pos=pos, size=(r, half_h, 0), material=material, **kw)
    if quat is not None:
        g.quat = quat
    return g


def hollow_cylinder(
    body,
    prefix,
    center,
    r_in,
    height,
    wall=0.003,
    n=16,
    material="ceramic",
    flare=0.0,
    base=True,
    base_thick=0.004,
    **kw,
):
    """Open-topped container (cup/bowl) built from n box walls + a base disc.

    `flare` tilts the wall outward (rad) so the mouth is wider than the base (a bowl). `center` is the
    centre of the base's underside; walls rise from there. Returns the list of geom names.
    """
    cx, cy, cz = center
    names = []
    L = height / np.cos(flare)
    r_mid = r_in + wall / 2 + (height / 2) * np.tan(flare)
    half_w = (r_in + wall) * np.tan(np.pi / n) * 1.08 + (height / 2) * np.tan(flare) * np.tan(np.pi / n)
    for i in range(n):
        phi = 2 * np.pi * i / n
        pos = (cx + r_mid * np.cos(phi), cy + r_mid * np.sin(phi), cz + base_thick + height / 2)
        q = qmul(qz(phi), qy(+flare))  # +flare leans the wall outward as it rises (mouth wider than base)
        body.add_geom(
            name=f"{prefix}_w{i}", type=BOX, pos=pos, size=(wall / 2, half_w, L / 2), quat=q, material=material, **kw
        )
        names.append(f"{prefix}_w{i}")
    if base:
        body.add_geom(
            name=f"{prefix}_base",
            type=CYL,
            pos=(cx, cy, cz + base_thick / 2),
            size=(r_in + wall, base_thick / 2, 0),
            material=material,
            **kw,
        )
        names.append(f"{prefix}_base")
    return names


def add_hex_mesh(spec, name, across_flats, height):
    """Convex hex-prism mesh (centred at origin, axis z)."""
    R = across_flats / np.sqrt(3)
    ang = np.arange(6) * np.pi / 3
    top = np.c_[R * np.cos(ang), R * np.sin(ang), np.full(6, height / 2)]
    bot = top * np.array([1, 1, -1])
    spec.add_mesh(name=name, uservert=np.r_[top, bot].flatten().tolist())
    return name


def yaw_of(R):
    """Yaw (rad) of a 3x3 rotation matrix about z."""
    return float(np.arctan2(R[1, 0], R[0, 0]))
