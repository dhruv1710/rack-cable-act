"""Zero-shot pi0.5 (pi05_droid) baseline on the MuJoCo task envs, all inside one Modal container.

    modal run models/pi/modal_baseline.py                       # 9 single-arm tasks x 3 episodes
    modal run models/pi/modal_baseline.py --episodes 1 --tasks relative_placement,bottle_binning
    modal volume get groot-pi-results /baseline out/pi_baseline

One L4 container runs the openpi policy server (localhost) and a few sim worker processes against it
(MUJOCO_GL=egl). Cost cap: L4 + 4 cores + 16 GiB ~ $1.2/h, hard timeout 60 min -> <= ~$1.20.
"""
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

_parents = Path(__file__).resolve().parents
ROOT = _parents[2] if len(_parents) > 2 else Path("/repo")  # inside the container the script sits at /root
TIMEOUT_MIN = int(os.environ.get("TIMEOUT_MIN", "60"))
TASKS = ["cable_insertion", "cable_routing", "fuse_holder", "parts_sorting", "bottle_binning",
         "plate_racking", "pouring_beads", "relative_placement", "drive_replacement"]

app = modal.App("groot-pi-baseline")
cache_vol = modal.Volume.from_name("groot-pi-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("groot-pi-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "curl", "build-essential", "clang", "ffmpeg",
                 "libegl1", "libgl1", "libglvnd0", "libglib2.0-0", "libosmesa6")
    .pip_install("uv")
    # openpi in its own uv-managed venv (jax[cuda12], pinned deps); LFS skipped, weights come from GCS at runtime
    .run_commands(
        "git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git /openpi",
        "cd /openpi && GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=300 UV_PYTHON_PREFERENCE=only-managed uv sync --no-dev --python 3.12 "
        "--no-install-package gym-aloha --no-install-package mujoco --no-install-package dm-control "
        "--no-install-package labmaze --no-install-package pynput --no-install-package evdev",
    )
    # sim side: system python
    .pip_install("mujoco==4.0.1", "gymnasium==1.4.0", "numpy", "pillow", "imageio[ffmpeg]", "msgpack", "dm-tree", "websockets")
    .run_commands('pip install --no-deps "git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"')
    .env({"MUJOCO_GL": "egl", "OPENPI_DATA_HOME": "/cache/openpi", "PYTHONPATH": "/repo",
          "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"})
    .add_local_dir(ROOT / "envs", "/repo/envs", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "tasks" / "rack_cable", "/repo/tasks/rack_cable", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "third_party/mujoco_menagerie/franka_emika_panda",
                   "/repo/third_party/mujoco_menagerie/franka_emika_panda")
)


def _wait_port(port, timeout, proc):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited with code {proc.returncode}")
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return time.time() - t0
        time.sleep(3)
    raise TimeoutError("policy server did not come up")


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=TIMEOUT_MIN * 60,
              volumes={"/cache": cache_vol, "/results": res_vol})
def sweep(episodes: int = 3, tasks: str = ",".join(TASKS), parallel: int = 3, seed: int = 0,
          checkpoint: str = "gs://openpi-assets/checkpoints/pi05_droid", config: str = "pi05_droid"):
    import json

    out = Path("/results/baseline")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").unlink(missing_ok=True)
    server_log = open(out / "server.log", "w")
    server = subprocess.Popen(
        ["/openpi/.venv/bin/python", "scripts/serve_policy.py", "policy:checkpoint", f"--policy.config={config}",
         f"--policy.dir={checkpoint}"], cwd="/openpi", stdout=server_log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "", "GIT_LFS_SKIP_SMUDGE": "1"})
    try:
        up = _wait_port(8000, 25 * 60, server)
        print(f"server up after {up:.0f}s", flush=True)
        cache_vol.commit()

        def run(task):
            t0 = time.time()
            p = subprocess.run(
                ["python", "-m", "envs.pi_run", task, "--port", "8000", "--episodes", str(episodes), "--seed", str(seed),
                 "--out", str(out / "videos"), "--json", str(out / "results.jsonl")],
                cwd="/repo", capture_output=True, text=True)
            (out / f"{task}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
            print(f"{task}: exit {p.returncode} in {time.time() - t0:.0f}s", flush=True)
            if p.returncode:
                print(p.stderr[-1500:], flush=True)

        with ThreadPoolExecutor(parallel) as ex:
            list(ex.map(run, [t for t in tasks.split(",") if t]))
    finally:
        server.terminate()
        res_vol.commit()

    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()] if (out / "results.jsonl").exists() else []
    print("\ntask                 success  mean_progress  episodes")
    for t in tasks.split(","):
        r = [x for x in rows if x["task"] == t]
        if r:
            print(f"{t:20s} {sum(x['success'] for x in r)}/{len(r)}      {sum(x['progress'] for x in r) / len(r):.2f}          {len(r)}")
        else:
            print(f"{t:20s} NO RESULTS (see {t}.log)")
    return rows


@app.local_entrypoint()
def main(episodes: int = 3, tasks: str = ",".join(TASKS), parallel: int = 3):
    sweep.remote(episodes=episodes, tasks=tasks, parallel=parallel)
