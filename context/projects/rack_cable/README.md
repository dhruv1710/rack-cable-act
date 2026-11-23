# Rack cable: ACT policy for patch-panel insertion

Last verified: 2026-11-23. Facts below come from the published page (`public/index.html`), not re-checked this session.
Page: `public/index.html` (the site root). Model: `public/rack_cable/model`. Videos: `public/rack_cable/videos`.
Eval: `python models/act/eval_act.py --checkpoint public/rack_cable/model --temporal-ensemble 0.01 --episodes 10`.

A Franka Panda in MuJoCo picks up a cable plug and inserts it into a patch-panel port from two cameras, outputting joint
targets; no port coordinates or object poses are given. A latch engages when the plug is within 2 mm sideways and 2.5 mm
of fully seated; that latch is the success check.

## Status
| Setup | Success |
|---|---|
| Default ACT, 50 actions per prediction | 4/20 |
| Re-plan every 10, 25 or 100 steps | 0/8 each |
| Temporal ensembling | 20/20 |
| Temporal ensembling, all runs | 52/52 |
| Temporal ensembling, 1.25x wider cable start | 10/10 |
