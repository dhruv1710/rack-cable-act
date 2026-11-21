"""Train ACT on Modal (L4) under a hard budget.

    modal volume create groot-act-data
    modal volume put groot-act-data out/rack_cable/datasets/rack_act_p2 /rack_act_p2
    modal run models/act/modal_train.py            # trains until TRAIN_MINUTES, then stops and commits
    modal volume get groot-act-out /act_run1 out/modal_act_run1

Cost cap (L4 $0.000222/s + 2 cores + 12 GiB ~ $0.99/h): TRAIN_MINUTES=44 -> ~$0.73, hard function timeout 50 min.
ACT has a constant LR (no schedule), so stopping at a time limit is harmless; checkpoints every SAVE_FREQ steps.
"""
import os
import subprocess
import threading
import time

import modal

TRAIN_MINUTES = float(os.environ.get("TRAIN_MINUTES", "44"))
TIMEOUT_MIN = int(os.environ.get("TIMEOUT_MIN", "50"))

app = modal.App("groot-act")
data_vol = modal.Volume.from_name("groot-act-data", create_if_missing=True)
out_vol = modal.Volume.from_name("groot-act-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "build-essential", "linux-libc-dev", "libgl1", "libglib2.0-0")
    # CUDA 12.8 wheels: runs on Modal's drivers; lerobot 0.7.0 accepts torch >=2.7,<2.16
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("lerobot[training,core-scripts]==0.7.0")
)


@app.function(image=image, gpu="L4", cpu=2, memory=12288, timeout=TIMEOUT_MIN * 60,
              volumes={"/data": data_vol, "/out": out_vol})
def train(steps: int = 40000, batch: int = 8, run: str = "act_run1", init_from: str = "", save_freq: int = 2500):
    import torch
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    data_vol.reload()
    cmd = [
        "lerobot-train",
        "--dataset.repo_id=local/rack_act_p2", "--dataset.root=/data/rack_act_p2",
        "--dataset.video_backend=pyav",
        *(["--policy.type=act", "--policy.chunk_size=200", "--policy.n_action_steps=50"] if not init_from
          else [f"--policy.path={init_from}"]),            # fine-tune from earlier weights (fresh optimizer)
        "--policy.device=cuda", "--policy.push_to_hub=false",
        f"--output_dir=/out/{run}", f"--job_name={run}",
        f"--batch_size={batch}", f"--steps={steps}", "--num_workers=3",
        f"--save_freq={save_freq}", "--log_freq=100",
    ]
    stop = threading.Event()

    def committer():                     # persist checkpoints while training runs
        while not stop.wait(120):
            try:
                out_vol.commit()
            except Exception as e:       # keep training even if a commit hiccups
                print("commit failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t0 = time.time()
    deadline = TRAIN_MINUTES * 60
    last = ""
    for line in p.stdout:
        line = line.replace("\r", "\n")
        if "step:" in line or "Error" in line or "Traceback" in line or "error" in line:
            print(line.rstrip()[:300], flush=True)
        last = line
        if time.time() - t0 > deadline:
            print(f"== time budget reached ({TRAIN_MINUTES} min): stopping training", flush=True)
            p.terminate()
            break
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
    stop.set()
    out_vol.commit()
    print(f"done in {(time.time() - t0) / 60:.1f} min, exit={p.returncode}", flush=True)


@app.local_entrypoint()
def main(steps: int = 40000, batch: int = 8, run: str = "act_run1", init_from: str = "", save_freq: int = 2500):
    train.remote(steps=steps, batch=batch, run=run, init_from=init_from, save_freq=save_freq)
