# Franka Panda: control, IK, planning

Last verified: 2026-11-23. Code: `envs/common/arms.py`, `envs/experts.py`, `envs/tests/scripted.py` (`Driver`).

## Orientation helpers
- `down_rot(yaw)`: vertical hand, gripper pointing down. Empirical rule: pass the object's long-axis angle and the pads
  straddle the object (bottle, plug, lying standoff all worked with `yaw = axis angle`). Adding pi/2 failed (pads landed on
  the plug). The docstring wording is easy to misread; trust the test.
- `forward_rot(yaw, roll)`: hand horizontal, pointing +x. `roll=0`: pads close along y (wide palm horizontal).
  `roll=pi/2`: pads close vertically (palm 20 cm tall vertically).
- The gripper is symmetric under a 180 degree turn about its approach axis. Fine for loose objects. For a held object the
  turn flips it over (a spare drive arrived upside down). Carry with `allow_flip=False`.

## IK
- `solve_ik` is damped least squares; its `ok` flag is conservative.
- A horizontal hand within about |y| < 0.07 of the arm's plane has few IK solutions. A map at x = 0.40-0.52 found solutions
  only at z of about 0.16-0.22 and only a few per pose.
- The vertical hand at x = 0.80 cannot reach z below about 0.042. At x = 0.74 it reaches the tray (z 0.022).
- The configuration used to grasp can be unconnected to the one needed later. A carried object cannot change arm
  configuration, so plan grasp and carry as one chain.

## Planning (`envs/experts.py`)
- `plan_joint_path(drv, poses, big_steps=...)`: DP over IK solution layers (warm start from every previous solution plus
  random seeds). `big_steps`: `"any"`, `"first"` (only the jump from the current state to the first waypoint), `"none"`.
- `plan_via`, `pmove`, `pmove_safe`, `pmove_via`, `run_joint_path` (per-segment speed cap, `stop` callback), `carry_path`,
  `reachable_R`. `pmove*` return False when planning fails and the arm then does not move: check the return value.
- Use `big="first"` for the initial approach (nothing held yet) and `"none"` while holding something.

## Tracking and contact
- The arm lags the target at the end of a fast move (up to 2 cm). Hold 0.3-0.6 s and send a correcting move before
  descending onto something tight.
- A hand pressed down by contact sinks further if each step is relative to the current hand pose. Integrate the target
  from the measured object error, not from where the hand is.

## Hand geometry
- A vertical hand's palm stops the fingertips about 4.5 cm below an object's top.
- Open fingers span 8 cm total; a 7.4 cm mug leaves 3 mm per side.
- Pad block is about 1.6 cm thick: too thick to follow a 1.5 mm clearance into a drive bay.
- link7 is about 6 cm wide and hits mounting plates and posts; tilt the hand or stand behind the obstacle.
