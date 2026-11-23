# MuJoCo findings

Last verified: 2026-11-23. Versions: MuJoCo 4.0.1, Python 3.12 (`.venv`), Gymnasium. Models are built in code with
`MjSpec`; the Panda comes from Menagerie. Physics 500 Hz, control 50 Hz.

## Gotchas
- **Rolling friction needs `condim=6`.** The friction triple is (slide, torsion, roll), but with the default `condim=3`
  the torsion and roll terms are ignored. Lying cylinders then roll on forever (parts sorting: standoffs drifted across
  the bench and were swept away before being picked; bolts rolled out of compartments). Fix: `condim=6` on the geoms.
- **Executing an unsolved IK result throws objects around.** Check IK success first and skip instead of executing.
- **Close grippers gradually.** An instant close squirts small round or hex parts out. Close over about 1.2 s.
- **Commanded height is not a safe descent.** Pushing a held fuse to a commanded height drove it through the floor.
  Descend in 1 mm steps against the measured height.
- **Body origin is not the centre.** Bottles have their origin at the base end, axis along local z. Grasp at
  `origin + axis * half_length`. Always check where a body's origin is before using `xpos` as a grasp point.
- **Thin plates (1 mm) work as surfaces** but report negative contact distances of 1-2 mm when something presses on them.
- **Hysteresis and force measure.** A weld detent was reliable when its release was judged by torque, not force, with
  a latch window and hysteresis.
- **Ball-joint cables bow sideways with tiny slack.** About 3 mm of slack over 0.4 m gave a 1.6 cm lateral bow. Pull taut.

## Retention and latches
`envs/common/latch.py` provides weld-equality latches (`add_weld`, `engage`, `release`, `holding_force`). Use them for
spring clips, drive detents, lid detents. Engage only inside a valid pose window (a detent engaged beyond a hinge limit
held the part wrongly).

## Grip actuator
`grip_ctrl(g)`: 0 = closed and squeezing, 1 = open. Squeeze is about 23 N on top of the position spring. A lower grip
value did not help a bottle that slipped (the bottle was not lifted at all).

## Debug recipes
- Wrap `env.step` to log a body's pose and its contacts every N steps; the first frame where something changes is the bug.
- Contact list: loop `d.contact[:d.ncon]`, map `m.geom_bodyid` to `m.body(...).name`, print the pair and `c.dist`.
- A scripted move that returns "ok" can still stop short (the arm lags the target by up to 2 cm at the end of a fast
  move). Print where the hand actually is.
- Measure over many seeds; failures often cluster on one cause that a single seed hides.
