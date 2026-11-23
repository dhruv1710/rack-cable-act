# Manipulation environments (MuJoCo)

Eleven task environments sharing one robot, one world and one Gymnasium interface. Everything runs on CPU
(Apple Silicon is fine). Physics 500 Hz, control 50 Hz.

```python
import envs
env = envs.make("fuse_holder")           # see envs.list_tasks()
obs, info = env.reset(seed=0)            # info["instruction"] is the language command
obs, r, terminated, truncated, info = env.step(env.noop_action())
```

| key | |
|---|---|
| `obs["observation.state"]` | per arm: 7 joint angles + gripper opening (0..1). 8 dims per arm |
| `obs["observation.images.<cam>"]` | uint8 RGB, default 192x256. `overview`, `wrist` (`wrist_left`/`wrist_right` for two arms) |
| action | per arm: 7 absolute joint targets (rad) + gripper (0 = squeeze closed, 1 = open) |
| `info` | `success`, `progress` (0..1), task metrics, `instruction`, `is_success` |
| reward | 1.0 on success, else 0 (`progress` is available for dense shaping) |

Success must hold for 0.3-0.6 s before the episode terminates. Episodes truncate at `env.max_seconds`.

## Layout

```
envs/
  common/world.py      lighting, materials, bench (table top = z 0), floor
  common/arms.py       Franka Panda (Menagerie) attach, IK, gripper model
  common/base_env.py   TaskEnv: obs/action spaces, reset/settle/step, success hold
  common/latch.py      retention latch (weld that releases under load)
  common/geom.py       boxes, hollow cups/bowls, hex-prism meshes
  tasks/*.py           one file per task: build(), randomize_state(), evaluate(), oracle_solve()
  registry.py          envs.make / envs.list_tasks
  smoke.py             build + reset + no-op stability + render + oracle success, all tasks
  tests/               reach.py, mechanisms.py, grasp_check.py, pick_place.py, pour_test.py
```

Shared parts, one compiled model per task: tasks add only their own bodies to the shared world (the "separate
scenes from shared assets" layout), so nothing from other tasks costs physics time.

## Tasks

| task | what the robot does | success | notes |
|---|---|---|---|
| `cable_insertion` | plug an RJ45-style cable into a port of a patch panel | plug latched in the port | wraps the existing `tasks/rack_cable` env (same physics/obs/action as the trained ACT policy) and only adds visual detail: switches with port arrays and LEDs, rack-unit holes, labels |
| `cable_routing` | thread a patch cable through three U-channel cable clips on a tray | cable passes through every clip, plug beyond the last | 40-segment ball-joint cable with a moulded plug; clips have flared lead-ins; clip positions randomised |
| `fuse_holder` | replace the blown 10x38 fuse in a DIN-rail holder | fresh fuse latched in the target holder, lid closed, blown fuse removed, other holders untouched | hinged lids, spring-clip retention (~6 N pull-out), foam caddy, scrap bin |
| `parts_sorting` | sort 3 bolts, 3 nuts, 3 standoffs into labelled compartments | every part in the compartment whose label matches | hex bolts/nuts from convex hex-prism meshes; type-to-compartment mapping randomised, so labels must be read |
| `object_handoff` | left Panda hands a bar to the right Panda, which places it on a pad | handoff (both grippers hold at once, then giver releases) and bar resting on the pad | two arms, two wrist cameras, 16-D action; handoff tracked from finger contacts |
| `bottle_binning` | move four PET bottles from a conveyor into a crate | all bottles inside the crate | bottles standing or lying, built from body/shoulder/neck/ring/cap with liquid mass |
| `plate_racking` | stand a melamine plate in the commanded coloured slot of a wire rack | plate upright in that slot, supported by the rack, at rest | plate on a stand, picked flat by the rim, rotated vertical, lowered between tines (1.5 mm per side clearance) |
| `pouring_beads` | pour 100 beads from a cup into a bowl, put the cup back | >=85 % of beads in bowl, <=8 % on the table, cup upright and released | cup/bowl are real open containers (16/20 wall panels); beads are rigid spheres |
| `relative_placement` | "put the can to the left of the mug" (left/right/front/behind/on top) | relation holds geometrically and the object is at rest | 6 objects, relation and objects sampled per episode; "on top" only for stable stacks |
| `thermal_patrol` | drive a wheeled robot along a data-center aisle and flag overheating servers | all hot servers flagged, at most 1 false flag | 16 racks x 20 servers; heat only visible in the 160x120 thermal camera; `observation.images.thermal`; action `[v, w, pan, tilt, flag]` |
| `drive_replacement` | swap the failed (amber LED) 3.5" drive in a 2U 12-bay server | spare latched in the failed bay with lever closed, failed drive out, others untouched | real hot-swap mechanism (see below) |

### Drive-replacement mechanism

Each carrier has a spring-loaded release button (about 6 N to reach the 4 mm unlock point), a spring-loaded
lever that pops open about 83 degrees when unlocked, a detent latch that holds a seated drive, and an ejector
spring that pushes an unlatched drive about 6 mm out of the bay. Bays have 1.5 mm vertical and 3 mm lateral
clearance. The spare lies on a stand with its lever open; insert it, then push the lever shut to latch.

## What makes them realistic (and what they check)

- Real dimensions: 10x38 mm fuses, M10 hardware, 500 ml bottles, 200 mm plate, 3.5" LFF carrier (147x102x26 mm),
  2U chassis, 42U-style racks with 2U servers, Cat6 cable diameter.
- Real masses and friction; torsional friction (`condim 4`) where an object is pinched and rotated; stiff
  contacts on thin walls so gripper pads do not sink in.
- Gripper models a commanded grasp force (about 23 N preload) like a Franka hand, instead of a bare position spring.
- Thermal camera is rendered through segmentation, per-surface temperature, thermal diffusion blur, noise and a colour map.
- Heights follow what the Panda can reach: a horizontal hand (fingers vertical) has a 20 cm-wide palm behind the
  TCP, so such grasps (plate rim, spare drive) are at shelf height, not on the bench. Standing bottles are pinched by the
  neck so the palm clears the cap.

## Tests

```bash
.venv/bin/python -m envs.smoke                      # build, reset, stability, render, oracle success: all 11 tasks
.venv/bin/python -m envs.tests.reach                # IK reachability of key poses (random-restart IK)
.venv/bin/python -m envs.tests.mechanisms           # fuse + drive mechanisms driven by applied forces (no arm)
.venv/bin/python -m envs.tests.grasp_check          # scripted Panda grasp + lift of each task's key objects
.venv/bin/python -m envs.tests.handoff_test         # two-arm scripted handoff
.venv/bin/python -m envs.tests.pour_test 3          # scripted pour
.venv/bin/python -m envs.tests.pick_place relative_placement 10
```

`smoke` writes contact sheets to `out/envs/`.

### What has been verified (last run)

| check | result |
|---|---|
| build, reset, 100 no-op steps without NaN, no spurious success, render | 11 / 11 tasks |
| success check fires on a solved state (`oracle_solve`) | 11 / 11 |
| deterministic reset with `randomize=False` | 11 / 11 |
| key poses reachable by IK | all arm tasks |
| fuse mechanism under forces (lid, 6 N retention, pull-out, reseat, close) | 7 / 7 checks |
| drive mechanism under forces (button, unlock, eject, pull, insert, latch) | 10 / 10 checks |
| Panda grasp + lift of key objects (bottles standing/lying, bead-filled cup, nut, bolt, 3 standoffs, fuse, bar, plate rim, spare drive, can, mug, apple, block) | 16 / 16 |
| scripted bimanual handoff, bar to pad | 1 / 8 (seed 0 only; contact-based stage tracking confirmed) |
| scripted pour (100 beads into the bowl, cup returned) | 4 / 8 episodes succeed |
| scripted pick-and-place for relative placement | 8 / 12 episodes |

The scripted solutions are test drivers, not expert policies: they use privileged state and joint-space/linear
IK without collision avoidance.

## Realism issues the tests found and fixed

Worth knowing because they show up as policy failures if reintroduced:

- Weld latches must be anchored at the body, not the world origin (otherwise light parts swing about a point metres away).
- Latches must ignore the release force for a moment after engaging and only engage at low speed.
- Gripper pads sank 9 mm into 2.5 mm cup walls with default contacts: pads and thin-walled objects use stiff contacts.
- A position-servo gripper holds a thin object with almost no force: closed commands add a constant ~23 N preload.
- Pinched objects need torsional friction (`condim 4`), or they pivot out of the fingers.
- A flared container helper leaned its walls inward (bowl wider at the base than at the mouth, leaking beads): fixed and now checked.
- Contact friction is the max of the two geoms, so high cup-wall friction (for grip) also stopped beads sliding out.
- A 0.46 kg porcelain plate cannot be held by a 9 mm rim pinch: the plate is melamine (about 0.21 kg).
- Panda reach limits shape layouts: a forward ~2 rad tip needs TCP x <= 0.48 m; horizontal-hand grasps need >= 12 cm clearance.

## Adding a task

Subclass `TaskEnv` in `envs/tasks/`, implement `build(spec)`, `bind()`, `randomize_state(rng)`, `evaluate()`,
`oracle_solve()` and `instruction`, register it in `envs/registry.py`, then run `python -m envs.smoke <name>`.

## Running a pi policy (zero-shot, no fine-tuning)

`envs/pi_run.py` drives an env with an openpi policy server over a websocket, using the DROID (`pi05_droid`)
interface: exterior + wrist image, 7 joint positions, gripper position, language prompt in; 7 joint velocities +
gripper out. Velocities are integrated into the env's absolute joint targets (0.2 rad per step at full velocity,
15 Hz policy, held for 3 env steps). The model needs a GPU; this Mac only runs the sim.

```bash
# on a GPU box (>= ~16 GB VRAM), inside an openpi checkout
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
# on the Mac (port-forward 8000 if needed)
.venv/bin/python -m envs.pi_run relative_placement --host <gpu-host> --port 8000 --episodes 5
.venv/bin/python -m envs.pi_run relative_placement --dry      # offline wiring check, no server
```

Needs `pip install --no-deps "git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client" msgpack dm-tree`.
Videos go to `out/pi/`. Single-arm tasks only. Expect ~0 % success: the base model has not seen this simulator.

### Zero-shot baseline: pi05_droid, no fine-tuning

`modal run models/pi/modal_baseline.py` (one L4 container: openpi server + sim workers on EGL). 3 episodes per task,
seeds 0-2, full episode length, replan every 8 actions, ~0.44 s per inference.

| task | success | mean progress |
|---|---|---|
| cable_insertion | 0/3 | 0.00 |
| cable_routing | 0/3 | 0.25 |
| fuse_holder | 0/3 | 0.25 |
| parts_sorting | 0/3 | 0.00 |
| bottle_binning | 0/3 | 0.00 |
| plate_racking | 0/3 | 0.00 |
| pouring_beads | 0/3 | 0.00 |
| relative_placement | 0/3 | 0.57 |
| drive_replacement | 0/3 | 0.00 |

Nonzero progress on cable_routing, fuse_holder and relative_placement is the starting state's partial credit (e.g.
objects already near their goal), not task progress by the policy; a still arm scores 0.52 on relative_placement.

## Scripted policies

Reference solutions that use inverse kinematics on a planned joint path (`envs/experts.py`, `envs/tests/*`,
`tasks/rack_cable/scripted_insert.py`). `python -m envs.record_scripted <task|all> <n_seeds>` runs them with rendering
and writes an mp4 per successful episode to `out/scripted/`.

| task | seeds solved | note |
|---|---|---|
| cable_insertion | 10 / 10 | random cable start (`scripted_insert.py --randomize`) |
| thermal_patrol | 6 / 6 | perception-guided: thermal image only, no hot-server list |
| plate_racking | 10 / 10 | slots 2 to 4 came up in these seeds |
| object_handoff | 10 / 10 | 20 cm bar; `experts.object_handoff` |
| parts_sorting | 8 / 12 | misses are single parts |
| bottle_binning | 8 / 12 | misses are one bottle short |
| relative_placement | 8 / 12 | `tests/pick_place.py`; misses are mostly the bowl and the tall box. A rewrite on planned paths scored 6 / 12 and was dropped |
| pouring_beads | 21 / 24 | `experts.pouring_beads`; beads always reach the bowl, the 3 misses are the cup tipping over on the return descent |
| fuse_holder | 12 / 12 | `experts.fuse_holder`; holder pitch widened to 50 mm, lid stop at 83 degrees |
| drive_replacement | 3 / 9 | seeds 4, 5, 6; see below |
| others | none | cable_routing (attempted, see below) |

Findings from writing them:

- The Panda hand is symmetric under a 180 degree turn, so pick the yaw whose IK converges (`experts.reachable_R`) and
  skip a pick rather than execute an unsolved IK result. The `ok` flag of `solve_ik` is conservative: the arm often
  tracks a target that it reports as not converged.
- A waypoint has several IK solutions on different arm configurations, and a warm start from the previous waypoint
  often dead-ends on one. `experts.plan_joint_path` collects solutions per waypoint (random seeds, plus the equivalent
  pose with the hand turned 180 degrees) and picks the chain with the fewest large joint steps by dynamic
  programming. A large step is allowed only where it is safe (`big_steps="first"`, e.g. a wrist flip high above the
  scene). This is what made plate racking and the handoff work.
- With the hand pointing straight ahead there is a band about 7 cm either side of the arm's sagittal plane where the
  warm-started IK has no solution. Layouts (and the handoff point) avoid it.
- The arm lags its target, so hold 0.5 s at the pre-grasp pose before a tight approach.
- Light or small parts (28 g bottles, 12 mm cylinders) are squirted out by an instant close: `gentle_grasp` closes over
  about 1.2 s. To seat an object on release, loosen the grip first and let it settle (plate, bar, cup).
- A lying cylinder rolls away from the first finger when gripped across its axis: grip its flat end faces instead.
- A vertical hand's palm stops the fingertips about 4.5 cm below the top of a tall object, and a horizontal hand needs
  its TCP 12 cm above the bench. A handoff needs a bar tall enough for both grasps (20 cm).
- The wrist trails the hand by about 15 cm, so a stand behind the target can be hit when placing (plate racking).
- The open fingers span about 8 cm: choose the grip yaw whose closing axis has no neighbour within about 6 cm
  (`experts._blockers`).
- A servo that centres on a hot server's blob must use the camera's offset from the base origin (8 cm on the patrol
  robot) and the bounding-box middle of the hottest region. A brightness-weighted centroid is pulled off by neighbours.

### Environment changes made while writing the policies

Lying bottles used a random yaw on the belt although a bottle is 19.5 cm long and the belt 17 cm wide, so they
overlapped and leaned on each other at reset. The belt is now 22 cm wide with two lanes along the belt. A lane holds
one lying bottle (aligned with the belt) plus one standing bottle, or two standing bottles. The pi0.5 baseline ran
before this change, on the earlier layout.

- **Plate racking:** the source stand and the rack are now on the same side of the arm (a carry across the base is
  unreachable) and the stand is behind the rack (the wrist would hit its posts).
- **Object handoff:** the bar is 20 cm tall (was 15 cm): a vertical hand grips only the top few centimetres and the
  horizontal hand needs its grasp point 12 cm up when it sets the bar down.
- **Cable routing:** the clips were staggered 7 cm sideways, so the path anchor, clip 0, 1, 2 and plug was about 0.62 m
  against a 0.54 m cable. The stagger is now 2 cm and the cable 0.65 m. A 0.65 m cable lying straight put the plug
  beyond the arm's reach (x = 0.97; the original 0.5 m cable also reached x = 0.86), so it now starts as a serpentine
  beside the clip row (plug at x = 0.74 to 0.77).

Cable routing, attempted and not solved:

- Gripping the 6 mm cable on the tray fails: the vertical hand's pads cannot get below the cable's centreline without
  touching the tray, so they close on nothing and the cable stays where it is.
- Handling the plug only (grip it, carry it along the clip row above the walls, lower it beyond the last clip) does not
  work either: the plug slips out of the grip as it drags the long cable, and it is 13.6 mm wide, wider than the 12.5 mm
  channels, so it has to go over them. Routing needs the cable pulled taut after it is threaded, which was not found.

The pi0.5 baseline ran before these changes, on the earlier layouts.

Fuse holder (`experts.fuse_holder`, 12 / 12 seeds, all four holders): pinch the lid tab and pull it open, grip the blown
fuse above the 15 mm cradle walls and lift it out of the clip into the scrap bin, pick the fresh fuse that is furthest
from the arm's swing (the wrist sweeps the near caddy slot), lower it into the cradle and open the pads the instant the
clip latches, then close the lid. What it took:

- The lid is opened by pinching its tab with a half-open horizontal hand (pad gap about 16 mm) and pulling toward the
  robot. The lid is balanced over its hinge, so gravity finishes opening it once the detent lets go.
- Seating needs a vertical hand tilted about 7 degrees (the wrist would hit the mounting plate behind the holders). The
  fuse is lowered in 1 mm steps from its *measured* height (it slides in the pads during the carry, and driving past the
  seat pushes it through the 6 mm floor), and the gripper starts opening on the same control step the clip latches: the
  weld frees the clip above 6 N, so any hand motion with the pads still closed breaks it.
- The fingers must not be fully open near the neighbours: 5 cm apart, a fully open hand (+-4 cm) reaches the next fuse.
- Closing: the lid hangs 83 degrees open. The pad starts 16 mm below the plate (the pad block is 1.6 cm thick), follows the
  lid's arc upward while closing up on its surface, then a firm horizontal push; the detent only latches inside its
  valid range (it fought the hinge limit when driven past it).

Environment changes: the hinged lids had no latch and tipped open by themselves within about 30 s with the arm idle. Each lid
now has a closed-position detent (weld latch that releases at about 0.4 N at the tab and re-latches when pushed shut after
being opened), a light hinge, and a stop at 83 degrees. The holders are 50 mm apart (were 34 mm): the Panda hand body
is 6 cm wide and caught the neighbours. The pi0.5 baseline ran before these changes.

Drive replacement, partial (`experts.drive_press_button`, `drive_pull_out`, `drive_pick_spare`; 6 seeds): the release button is
pressed and the lever unlocks 6 / 6, the drive is pulled clear by its lever 6 / 6 (neighbours stay latched 6 / 6), and the
spare is picked up from its stand 5 / 6. Inserting the spare is not solved (`drive_insert_spare` does not work).

- The hand must roll so the pads close sideways when pressing or pulling: with the pads closing vertically the hand's 20 cm
  palm is vertical and catches the shelf in front of the chassis.
- The button needs 4 mm and is 1 cm wide; a pressing fingertip stalls at about 1.5 mm for most aim points, so the press
  retries with different aim points until the lever unlocks.
- The planner treats a hand pose and the same pose turned 180 degrees about the approach axis as equivalent. That is wrong
  for a held object: it can turn the drive upside down. Carrying needs `allow_flip=False`.
- Why the insertion fails: a drive held by two pads rotates about the pad contact as soon as the hand rolls, so the carry
  needs one fixed hand orientation along a continuous joint path. The planner finds none from the stand to the bays. A
  joint-space jump tilts the hand and the drive slips out of the pads (its yaw also drifts when gripped by the front edge).
  Even with a good carry the drive's 1.5 mm vertical clearance is thinner than the pads, so the last 2.4 cm needs a second
  tool (a fingertip push), then the lever has to be swung shut to latch it. Moving the spare stand in front of the bays
  would probably remove the carry problem.

Update: `drive_replacement` now runs end to end (`experts.drive_replacement`). Insertion aligns the carried drive on its
measured pose (position integrator plus yaw correction; a pitch correction diverged and was dropped), slides it in as far as
the vertical pads fit, re-grips the front edge, and finishes with a closed-finger push (hand rolled so the pads are the thin
way up, since the vertical pads stall 2 cm short of the seat). `drive_close_lever` sweeps the open lever shut along its arc.
Success 3 / 9 (seeds 4, 5, 6). The other seeds fail in carry planning (no continuous orientation-constant path to bays in
columns 0 to 2; allowing a joint jump did not help), and seed 1 failed at the spare pick-up.

Pouring fix: the cup swung in the grip after the pour and tipped over during the descent back to the table (it slid into the
robot base). Settling for 0.6 s, planning the return without wrist flips and descending in 16 waypoints over 3 s raised the
scripted pour from 5 / 8 to 21 / 24 (seeds 0 to 23). Beads reached the bowl 24 / 24.
