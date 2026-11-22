"""Zero-shot rollout of an openpi (pi0 / pi0.5) policy in one of the MuJoCo task envs.

The model runs on a remote GPU (``scripts/serve_policy.py`` from openpi); this process only runs the sim and talks
to it over a websocket. Interface = the DROID checkpoints (``pi05_droid``), the only public pi checkpoints trained
on a Franka Panda:

    request   observation/exterior_image_1_left  224x224x3 uint8   <- env camera ``overview``
              observation/wrist_image_left       224x224x3 uint8   <- env camera ``wrist``
              observation/joint_position         (7,)              <- obs state[:7]
              observation/gripper_position       (1,)  0 open .. 1 closed   (env: 1 - opening)
              prompt                             str               <- info["instruction"]
    response  actions  (T, 8): 7 normalised joint velocities in [-1, 1] + gripper position (0 open, 1 closed)

Env actions are absolute joint targets, so each velocity step is integrated: ``q_target = q + v * MAX_DELTA``
(DROID's controller scales velocity by a 0.2 rad per-step limit; 15 Hz policy rate, held for 3 env steps at 50 Hz).

Run from the repo root:

    .venv/bin/python -m envs.pi_run relative_placement --host <gpu-host> --port 8000
    .venv/bin/python -m envs.pi_run relative_placement --dry          # no server: zero-velocity policy, checks wiring

Expect near-zero success: the DROID checkpoints have never seen this simulator, camera placement or these tasks.
This measures how a base model behaves, it is not an evaluation of the envs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import envs

MAX_DELTA = 0.2  # rad per policy step at full velocity (DROID joint-velocity controller)
POLICY_HZ = 15
CONTROL_HZ = 50
ARM_TASKS_SINGLE = [t for t in envs.list_tasks() if t not in ("object_handoff", "thermal_patrol")]


def resize_pad(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Letterbox to size x size, keeping aspect ratio (same as openpi's image_tools.resize_with_pad)."""
    from openpi_client import image_tools

    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


class DryPolicy:
    """Stand-in server: holds still. Lets the sim loop, camera mapping and video writer be tested offline."""

    def infer(self, obs: dict) -> dict:
        assert obs["observation/joint_position"].shape == (7,)
        return {"actions": np.zeros((10, 8), dtype=np.float32)}


def build_request(obs: dict, instruction: str) -> dict:
    state = np.asarray(obs["observation.state"], dtype=np.float32)
    return {
        "observation/exterior_image_1_left": resize_pad(obs["observation.images.overview"]),
        "observation/wrist_image_left": resize_pad(obs["observation.images.wrist"]),
        "observation/joint_position": state[:7],
        "observation/gripper_position": np.array([1.0 - state[7]], dtype=np.float32),
        "prompt": instruction,
    }


def to_env_action(chunk_row: np.ndarray, state: np.ndarray) -> np.ndarray:
    vel = np.clip(chunk_row[:7], -1.0, 1.0)
    grip_closed = 1.0 if chunk_row[7] > 0.5 else 0.0  # DROID binarises the gripper command
    return np.concatenate([state[:7] + vel * MAX_DELTA, [1.0 - grip_closed]]).astype(np.float32)


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path.with_suffix(".mp4"), frames, fps=fps)
        print("video:", path.with_suffix(".mp4"))
    except Exception:
        from PIL import Image

        ims = [Image.fromarray(f) for f in frames[:: max(1, len(frames) // 150)]]
        ims[0].save(path.with_suffix(".gif"), save_all=True, append_images=ims[1:], duration=1000 // fps)
        print("video:", path.with_suffix(".gif"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=ARM_TASKS_SINGLE)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dry", action="store_true", help="use a zero-velocity stand-in policy (no server needed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=None, help="cap episode length (default: env.max_seconds)")
    ap.add_argument("--replan", type=int, default=8, help="actions executed per inference (open-loop horizon)")
    ap.add_argument("--prompt", default=None, help="override the task instruction")
    ap.add_argument("--out", default="out/pi")
    ap.add_argument("--json", default=None, help="append one JSON result line per episode to this file")
    args = ap.parse_args()

    if args.dry:
        policy = DryPolicy()
    else:
        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        print("server metadata:", policy.get_server_metadata())

    env = envs.make(args.task)
    hold = round(CONTROL_HZ / POLICY_HZ)
    max_s = args.seconds or env.max_seconds
    n_policy_steps = int(max_s * POLICY_HZ)
    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        prompt = args.prompt or info["instruction"]
        print(f"\nepisode {ep}: {prompt!r}")
        frames, chunk, k, n_infer, t_infer = [], None, 0, 0, 0.0
        done = False
        for _ in range(n_policy_steps):
            if chunk is None or k >= args.replan or k >= len(chunk):
                t0 = time.time()
                chunk = np.asarray(policy.infer(build_request(obs, prompt))["actions"])
                t_infer += time.time() - t0
                n_infer += 1
                k = 0
            action = to_env_action(chunk[k], np.asarray(obs["observation.state"]))
            k += 1
            for _ in range(hold):
                obs, _, terminated, truncated, info = env.step(action)
                frames.append(obs["observation.images.overview"])
                done = terminated or truncated
                if done:
                    break
            if done:
                break
        ok = bool(info.get("is_success", False))
        successes += ok
        print(
            f"  success={ok} progress={info.get('progress', 0):.2f} sim_time={len(frames) / CONTROL_HZ:.1f}s "
            f"inferences={n_infer} mean_latency={t_infer / max(n_infer, 1) * 1000:.0f} ms"
        )
        if args.json:
            rec = {"task": args.task, "seed": args.seed + ep, "success": ok, "progress": float(info.get("progress", 0)),
                   "sim_seconds": len(frames) / CONTROL_HZ, "inferences": n_infer, "prompt": prompt}
            with open(args.json, "a") as f:
                f.write(json.dumps(rec) + "\n")
        save_video(frames, Path(args.out) / f"{args.task}_seed{args.seed + ep}", fps=30)
    print(f"\n{successes}/{args.episodes} episodes successful")


if __name__ == "__main__":
    main()
