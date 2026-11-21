# Server-rack cable insertion environment (MuJoCo)

## Intent
A MuJoCo environment where a Franka Panda picks up a cable by its connector and plugs it
into a port on a server-rack patch panel. Purpose: a testbed for running robot policies
(first pi0.5 via openpi) on wire tasks, and later for generating demonstrations.

## Goals
- Runs on the user's Apple Silicon Mac (CPU physics, offscreen rendering).
- Gymnasium-style `reset()` / `step()`; language instruction per episode
  ("plug the cable into port N").
- Two cameras: front and wrist. Randomized cable start pose and target port.
- Success check: plug seated in the target port within tolerance.
- Policy-agnostic interface so an openpi client (running on a cloud GPU) can drive it.

## Non-goals
- Photoreal rendering, detailed server models, sim-to-real transfer guarantees.
- Running the policy on the Mac (it runs on a remote GPU server).

## Anti-goals
- A scene so stiff or brittle that no policy or scripted controller can ever succeed.

## Approaches considered
1. MuJoCo (chosen): native cable composite, runs on Mac, existing LIBERO-style pi0.5 pipeline.
2. Isaac Sim: better rendering, but no native cable, needs a paid RTX box for every iteration.
   Revisit if camera realism or GR00T 2 tooling becomes the priority.

## Design (calls made by Claude, correctable)
- Robot: Franka Panda from MuJoCo Menagerie (user's choice).
- Rack: simplified 19" rack with a patch panel holding a row of ports (boxes/primitives).
- Cable: chain of 16 capsule segments with spring-damped ball joints and a rigid plug
  (grip block + tip) on the last segment. CHANGED from the original plan: MuJoCo 4.0.1
  removed the `composite` cable element, so the chain is built in `tasks/rack_cable/build_scene.py`.
- Insertion: socket with small clearance; success = plug within a few mm / degrees of the
  port and inserted past a depth threshold. Optional latch to lock the plug once seated.
- Interface: `RackCableEnv` (Gymnasium), observations = 2 images + joint state, actions =
  end-effector deltas + gripper (joint-space option later).
- Milestones: (1) scene loads and renders, (2) scripted pick-and-insert succeeds,
  (3) env wrapper + randomization, (4) openpi client connection.

## Status / changes made while building (milestone 2 done)
- Milestone 1 (scene loads, renders) and 2 (scripted pick-and-insert, `tasks/rack_cable/scripted_insert.py`)
  done: all 4 ports succeed, tip within ~2 mm, 24.7/25 mm inserted.
- Socket clearance is 1.5 mm/side (`CLEARANCE`), not the 0.5 mm of a real RJ45, and each socket has a
  30 deg x 14 mm lead-in funnel. A rigid scripted arm cannot hold sub-mm alignment.
- Plug tip is 40 mm long and the body 32 mm: the grip must stay clear of the funnel when seated.
- A weld "latch" (equality `latch`) locks the plug once seated; without it the hanging cable pulls it
  out after release. It is switched on by the script, not by physics.
- Gripper force scaled 10x (Menagerie pinch is ~2 N); plug has torsional friction (condim 6).
- Cable-robot collision is disabled via contact bit masks (cable contype 2, conaffinity 0).
- NOT yet done: randomized cable start pose, Gymnasium wrapper, openpi client (milestones 3, 4).
  The script uses privileged sim state (plug pose); a learned policy will not have it.

## Left to the builder
Exact cable segment count/stiffness, tolerances, camera poses, controller details.
