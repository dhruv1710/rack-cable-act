# Rack cable: findings

Last verified: 2026-11-23 (from the published page; (unverified) in this session).

- 50 scripted demonstrations (54,879 frames at 50 Hz). ACT (LeRobot), 17,500 steps, batch 8, one L4, about $0.995.
- Two data bugs cost the most time: the video encoder silently dropped frames when its queue filled (video shorter than the
  data), and the expert paused about 10 s at one pose, which the policy could not learn to leave.
- After that the policy grasped but missed the final insertion. Re-planning more often did not help. Temporal ensembling did
  (query every step, average overlapping predictions; no retraining).
- Limits: simulation only; success is a simulated latch; 52/52 supports only about 93%+ reliability at 95% confidence;
  insertion lands within about 2.5 mm against a 2 mm / 2.5 mm tolerance; one port, one fixed camera pose.
- Next ideas on the page: move the camera, change lighting and physics; 60+ seeds; real or teleoperated data; try pi0.5.
