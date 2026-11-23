# Tasks: what the code does not say

Last verified: 2026-11-23. Constants live in `envs/tasks/<task>.py`; this file holds approaches, reasons and pitfalls.

## cable_routing
Original layout (0.65 m serpentine cable, clips staggered +-2 cm) is **not solved**. Current layout: straight 0.40 m cable
(32 segments), anchor and the three clips on one line (clip offsets +-2 mm), cable starts angled about 0.2 rad away from
the row. Policy `experts.cable_routing`: grip the plug (`down_rot(yaw = plug axis angle)`), lift to z 0.12, pull to
x 0.745 so the taut cable runs above the clip walls, lower slowly to the tray so it drops into the channels, then pull
2 cm further along the tray. Without that last pull the cable bows 1.6 cm and misses clip 2. Pulling past x 0.755 makes
the plug slip. The cable length was chosen so the taut plug (x about 0.74) is reachable at tray height.

## fuse_holder
Open a lid by pinching its tab with a half-open horizontal hand (pad gap about 16 mm) and pulling toward the robot; lids
have weld detents. Lift the blown fuse above the cradle walls into the scrap bin. Take the fresh fuse furthest from the
arm's swing (the wrist sweeps the near slot). Descend by measured height. Open the pads the instant the clip latches.
Close the lid with a pad arc and keep pressing until it latches. Holders are 50 mm apart (closer than that hit the hand).

## drive_replacement
Bay geometry: shelf at z 0.12, chassis front x 0.60, 3 rows x 4 columns. Spare lies on a stand at (0.44, 0.31).
Press the release button (needs more than 4 mm; one centimetre wide; retry aim points), pull the open lever out. Use
the hand rolled so the pads close sideways (`roll=0`) for the button and lever: the vertical palm hits the shelf. Pick the
spare with `roll=pi/2` at its centre. Insert: align on the drive's measured pose, slide until the pads stop at the opening
(they are thicker than the 1.5 mm vertical clearance), release, re-grip, push, then a closed-finger push with `roll=0`,
then sweep the lever shut along its arc (`drive_close_lever`). Current result and the carry problem: `knowledge.md`.

## parts_sorting
Nine parts, three compartments, mapping randomised. Grip cylinders across their flat ends (gripped across they roll).
Pick the least blocked part first (`_blockers`), retry parts not lifted or not landed. Rolling friction (`condim=6`) was
added after the baseline run.

## bottle_binning
Standing: pinch the neck. Lying: grasp at mid-body (`origin + axis * H_BODY/2`; the origin is at the base end). Close
gently (28 g bottles squirt out). Crate slots: with a lying bottle present it gets a lane along the crate's long axis and
standing bottles use the other lane; with none, four 2x2 slots.

## pouring_beads
Grip the cup, carry it low over the bowl, tilt about the world y axis through the TCP up to 2.35 rad, wait until at most 2
beads remain, untilt in the air, return. The return needs: a 0.6 s settle, no wrist flips, and a slow 16-waypoint descent,
or the cup swings and tips.

## relative_placement
Solver is `solve_relative_placement` in `envs/tests/pick_place.py` (direct IK moves, no collision avoidance). The bowl is
gripped by its near rim wall. The mug is 7.4 cm across and the open gripper is 8 cm.

## plate_racking
Source and rack are on the same side of the arm (a carry across the base is unreachable). The source stand is behind the
rack because link7 hits the posts. Rack clearance 1.5 mm per side. The planner uses a wrist flip high above the obstacles.

## object_handoff
The bar is 20 cm tall: a vertical hand grips only the top few centimetres. Left arm grips the top; right arm grips the
bottom plus 0.125 m. The right arm's reachable plane forces the handover point y = 0.09. Release in stages for a tilted bar.

## thermal_patrol
Perception only: bounding-box blob centring on the thermal image with pan/tilt feedback. The camera is offset 8 cm from
the base, so correct with `square_on`. Hot threshold 50 C; decoys stay under 48 C. Action is `[v, w, pan, tilt, flag]`.
