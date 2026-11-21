"""Generate expert demonstrations as a LeRobot dataset (for ACT / pi0.5 fine-tuning).

    .venv/bin/python tasks/rack_cable/gen_dataset.py --episodes 100 --workers 6 --port 2 \
        --out out/rack_cable/datasets/rack_act_p2

Each worker runs the scripted expert on a randomized cable pose and saves a temporary .npz;
the main process validates it (only SUCCESSFUL episodes are kept) and appends it to the dataset.
Recorded at the env rate (50 Hz): observation = {overview cam, wrist cam, state(8)}, action = joint targets(7)+gripper.
"""
import argparse
import contextlib
import io
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    # Main process only (spawned workers skip this). On Linux with MUJOCO_GL=osmesa, loading
    # torch/triton AFTER MuJoCo's OSMesa segfaults, so import lerobot (-> torch) first.
    import lerobot.datasets.lerobot_dataset  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from env import CAMERAS, FPS, STATE_NAMES, RackCableEnv  # noqa: E402
from scripted_insert import Demo  # noqa: E402

ACTION_NAMES = [f"joint{i}_target" for i in range(1, 8)] + ["gripper_open"]


def run_episode(job):
    seed, port, tmp_dir, img_hw = job
    t0 = time.time()
    env = RackCableEnv(port=port, img_hw=img_hw, randomize=True, render=True)
    env.reset(seed=seed)
    rec = {"state": [], "action": [], **{k: [] for k in CAMERAS}}

    def cb(q_des, grip01):
        obs = env.get_obs()
        rec["state"].append(obs["observation.state"])
        rec["action"].append(np.r_[q_des, grip01].astype(np.float32))
        for k in CAMERAS:
            rec[k].append(obs[k])

    demo = Demo(env, record_cb=cb)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            ok = bool(demo.run())
        except Exception as e:  # keep the pool alive; report the seed
            env.close()
            return seed, False, None, 0, time.time() - t0, f"exception: {e!r}"
    env.close()
    n = len(rec["action"])
    path = None
    if ok:
        path = str(Path(tmp_dir) / f"ep_{seed:06d}.npz")
        np.savez(path, state=np.stack(rec["state"]), action=np.stack(rec["action"]),
                 **{k.replace(".", "_"): np.stack(rec[k]) for k in CAMERAS})
    return seed, ok, path, n, time.time() - t0, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100, help="number of SUCCESSFUL episodes to keep")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--port", type=int, default=2)
    ap.add_argument("--out", default="out/rack_cable/datasets/rack_act_p2")
    ap.add_argument("--repo-id", default="local/rack_act_p2")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--img-h", type=int, default=192)
    ap.add_argument("--img-w", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    tmp = Path(args.out + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    H, W = args.img_h, args.img_w
    vid = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    features = {
        **{k: dict(vid) for k in CAMERAS},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create(args.repo_id, fps=FPS, features=features, root=out,
                               robot_type="franka_panda_sim", use_videos=True, streaming_encoding=True,   # encode while adding frames: ~10x faster ingest
                               encoder_queue_maxsize=2000,   # default 30 DROPS frames when the encoder lags -> video shorter than data
                               
                               # AV1 (default) encodes ~75 s/episode on one core; H.264 is far faster.
                               # g=2 keeps a keyframe every 2 frames so random access stays cheap.
                               rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, g=2, preset="ultrafast"))
    task = f"plug the cable into port {args.port}"
    max_attempts = args.max_attempts or int(args.episodes * 1.6) + 4
    jobs = [(args.seed0 + i, args.port, str(tmp), (H, W)) for i in range(max_attempts)]

    kept = attempts = 0
    t_start = time.time()
    fail_log = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for seed, ok, path, n, dt, err in pool.imap_unordered(run_episode, jobs):
            attempts += 1
            if ok and kept < args.episodes:
                with np.load(path) as zf:     # npz access is lazy: load each array ONCE, not per frame
                    z = {k: zf[k] for k in zf.files}
                for i in range(n):
                    ds.add_frame({
                        "observation.images.overview": z["observation_images_overview"][i],
                        "observation.images.wrist": z["observation_images_wrist"][i],
                        "observation.state": z["state"][i],
                        "action": z["action"][i],
                        "task": task,
                    })
                ds.save_episode()
                kept += 1
            else:
                fail_log.append((seed, err or "expert failed"))
            if path:
                Path(path).unlink(missing_ok=True)
            print(f"[{time.time() - t_start:6.0f}s] seed {seed}: {'OK ' if ok else 'FAIL'} ({n} frames, {dt:.0f}s)"
                  f" | kept {kept}/{args.episodes}, attempts {attempts}/{max_attempts}", flush=True)
            if kept >= args.episodes:
                pool.terminate()
                break
    ds.finalize()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ndone: kept {kept} episodes from {attempts} attempts "
          f"(expert success rate {kept / max(attempts, 1):.0%}); dataset at {out}")
    if fail_log:
        print("failures:", fail_log[:10])


if __name__ == "__main__":
    main()
