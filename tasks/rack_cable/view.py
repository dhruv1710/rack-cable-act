"""Open the rack + cable + Franka scene in the interactive MuJoCo viewer.

Run on macOS with:  .venv/bin/mjpython tasks/rack_cable/view.py
"""
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent))
from build_scene import build_model

m, _ = build_model()
d = mujoco.MjData(m)
if m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.viewer.launch(m, d)
