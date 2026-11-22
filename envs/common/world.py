"""Shared world: lighting, procedural materials, floor, and a robot bench (table top at z = 0)."""

import mujoco

# name -> rgba. Everything is a plain colour so tasks can mix freely; textured ones are defined below.
MATERIALS = {
    "steel": (0.62, 0.64, 0.67, 1),
    "dark_steel": (0.28, 0.29, 0.31, 1),
    "alu": (0.78, 0.79, 0.81, 1),
    "black": (0.05, 0.05, 0.06, 1),
    "charcoal": (0.13, 0.14, 0.16, 1),
    "white": (0.92, 0.92, 0.9, 1),
    "red": (0.78, 0.10, 0.08, 1),
    "green": (0.12, 0.60, 0.25, 1),
    "blue": (0.10, 0.28, 0.75, 1),
    "yellow": (0.95, 0.78, 0.10, 1),
    "orange": (0.92, 0.42, 0.08, 1),
    "purple": (0.45, 0.20, 0.60, 1),
    "brass": (0.78, 0.60, 0.22, 1),
    "copper": (0.72, 0.42, 0.25, 1),
    "rubber": (0.07, 0.07, 0.08, 1),
    "wood": (0.58, 0.42, 0.27, 1),
    "cardboard": (0.66, 0.52, 0.36, 1),
    "foam": (0.12, 0.12, 0.14, 1),
    "glass": (0.80, 0.90, 0.95, 0.35),
    "clear_blue": (0.55, 0.75, 0.95, 0.45),
    "amber": (0.95, 0.55, 0.05, 1),
    "ceramic": (0.95, 0.95, 0.93, 1),
    "led_green": (0.1, 1.0, 0.2, 1),
    "led_red": (1.0, 0.1, 0.05, 1),
    "led_amber": (1.0, 0.6, 0.0, 1),
    "led_off": (0.15, 0.15, 0.15, 1),
    "plastic_grey": (0.55, 0.56, 0.58, 1),
    "crate_blue": (0.10, 0.25, 0.55, 1),
    "hot_pink": (0.95, 0.25, 0.55, 1),
    "tan": (0.82, 0.70, 0.50, 1),
}


def base_xml(
    name="scene",
    table_x=(-0.35, 1.15),
    table_hy=0.65,
    zfar=20.0,
    floor_z=-0.75,
    with_table=True,
    floor_half=6.0,
    floor_repeat=12,
):
    """XML for the shared world. Table top is z=0; the floor is `floor_z` below it."""
    mats = "\n".join(
        f'    <material name="{k}" rgba="{" ".join(str(c) for c in v)}" specular="0.3" shininess="0.4"/>'
        for k, v in MATERIALS.items()
    )
    cx, hx = (table_x[0] + table_x[1]) / 2, (table_x[1] - table_x[0]) / 2
    table = ""
    if with_table:
        h = -floor_z
        table = f"""
    <geom name="bench_top" type="box" pos="{cx} 0 -0.02" size="{hx} {table_hy} 0.02" material="bench" friction="0.8 0.01 0.001"/>
    <geom name="bench_apron" type="box" pos="{cx} 0 -0.075" size="{hx - 0.02} {table_hy - 0.02} 0.035" material="dark_steel" contype="0" conaffinity="0"/>
    <geom name="bench_leg1" type="box" pos="{table_x[0] + 0.04} {-table_hy + 0.04} {floor_z / 2 - 0.02}" size="0.03 0.03 {h / 2 - 0.02}" material="dark_steel" contype="0" conaffinity="0"/>
    <geom name="bench_leg2" type="box" pos="{table_x[0] + 0.04} {table_hy - 0.04} {floor_z / 2 - 0.02}" size="0.03 0.03 {h / 2 - 0.02}" material="dark_steel" contype="0" conaffinity="0"/>
    <geom name="bench_leg3" type="box" pos="{table_x[1] - 0.04} {-table_hy + 0.04} {floor_z / 2 - 0.02}" size="0.03 0.03 {h / 2 - 0.02}" material="dark_steel" contype="0" conaffinity="0"/>
    <geom name="bench_leg4" type="box" pos="{table_x[1] - 0.04} {table_hy - 0.04} {floor_z / 2 - 0.02}" size="0.03 0.03 {h / 2 - 0.02}" material="dark_steel" contype="0" conaffinity="0"/>"""
    return f"""<mujoco model="{name}">
  <compiler autolimits="true" angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="960" fovy="45"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.26 0.26 0.28" diffuse="0.28 0.28 0.30" specular="0 0 0"/>
    <map znear="0.01" zfar="{zfar}" force="0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.78 0.84 0.92" rgb2="0.32 0.36 0.44" width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker" mark="edge" rgb1="0.46 0.47 0.49" rgb2="0.42 0.43 0.45" markrgb="0.30 0.30 0.32" width="512" height="512"/>
    <material name="floor" texture="floor_tex" texrepeat="{floor_repeat} {floor_repeat}" texuniform="true" reflectance="0.08"/>
    <texture name="bench_tex" type="2d" builtin="flat" mark="random" random="0.06" rgb1="0.34 0.37 0.41" markrgb="0.26 0.28 0.32" width="512" height="512"/>
    <material name="bench" texture="bench_tex" texrepeat="3 3" reflectance="0.05"/>
    <texture name="mat_tex" type="2d" builtin="flat" mark="random" random="0.1" rgb1="0.10 0.14 0.12" markrgb="0.2 0.25 0.22" width="256" height="256"/>
    <material name="mat" texture="mat_tex" texrepeat="4 4"/>
{mats}
  </asset>
  <worldbody>
    <light name="key" pos="0.4 -0.6 2.2" dir="0 0.15 -1" diffuse="0.55 0.55 0.52" specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="fill" pos="1.2 1.0 1.8" dir="-0.3 -0.3 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 {floor_z}" size="{floor_half} {floor_half} 0.05" material="floor"/>{table}
  </worldbody>
</mujoco>"""


def new_spec(**kw) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(base_xml(**kw))
