# Failed approaches and open problems

Last verified: 2026-11-23.

## Open problems
- **drive_replacement carry (3/9).** The spare is gripped level and must stay level to the bay. The pick configuration is
  often not connected to one that reaches the bay, and carrying cannot change configuration. Work in progress is saved as
  `git stash@{0}` ("drive-replacement carry WIP: fixes seeds 1,2; breaks seeds 4,5"). It plans the pick and the carry as
  one chain (`drive_pick_spare(..., bay=...)`), keeps the slide's y/z targets integrated from the drive's measured error
  (a relative target ratcheted the hand downward), and fixes the pitch correction sign. Result: seeds 1, 2, 6 succeed,
  4 and 5 broke (cause not investigated). Seed 3 (bay y +0.16) stalls in the slide IK; seeds 0 and 7 never grip the drive.
- **cable_routing original layout:** not solved (see tasks.md for the working simplified layout).
- **relative_placement (16/24):** mug rim collisions. A settle step and a smaller mug did not help (13/24, 12/24).
- **parts_sorting (16/24):** some bolts land 5-8 cm short or in a neighbouring compartment; cause not established.
- **bottle_binning (10/12):** one bottle is knocked off the belt (seeds 3, 4).
- **pouring_beads (21/24):** cup tips on the return descent.

## Tried and did not work (do not retry blindly)
- Moving the spare-drive stand (to the centre, to the side): 0/6, no change in the carry problem.
- Yawing the hand +-0.3/0.5 rad mid-carry and carrying 3 or 6 cm above the seat: no continuous path on any tested seed,
  including bay (2,0) outside the thin IK band.
- Allowing a joint jump during the carry (`big="first"/"any"`): flings the drive off the pads.
- Pitch-correcting a carried drive with the wrong sign: diverged (1 -> 2 -> 4 -> 9 degrees).
- Lower grip force for a lying bottle: the bottle was not lifted. Perpendicular hand yaw: 4/12.
- Dropping parts deeper into compartments: no change.
- Settle-then-descend plus a smaller mug for relative placement: not better.
- A planner-based rewrite of relative placement: worse than the original direct solver.
- Gripping the 6 mm cable flat on the tray: pads cannot get under it. Dragging a long serpentine cable by the plug:
  the plug slips.
- Rerunning a failed Pages deploy: same failure; split the workflow instead (see `shared/publishing.md`).
