"""Roll out a trained ACT checkpoint in the rack environment and report success rate.

    .venv/bin/python models/act/eval_act.py \
        --checkpoint out/rack_cable/act/checkpoints/best/pretrained_model --episodes 20 --video-dir out/rack_cable/act/evals/act

Success = the port latch engaged (plug tip seated and aligned) before the time limit.
Seeds are disjoint from the training seeds (gen_dataset.py uses 1000+), so this is a fair test.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "rack_cable"))
from env import CAMERAS, RackCableEnv  # noqa: E402


def to_batch(obs, device):
    b = {"observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0)}
    for k in CAMERAS:
        img = torch.from_numpy(obs[k]).permute(2, 0, 1).float() / 255.0
        b[k] = img.unsqueeze(0)
    return {k: v.to(device) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to .../pretrained_model")
    ap.add_argument("--port", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=50_000)
    ap.add_argument("--max-seconds", type=float, default=45.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--rand-scale", type=float, default=1.0,
                    help="scale the cable start-pose randomization (1.0 = training range)")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                    help="ACT temporal ensembling (query every step, exp-weighted average of overlapping chunks); "
                         "e.g. 0.01 as in the ACT paper")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override how many steps of each predicted chunk run before re-planning (default: checkpoint's)")
    args = ap.parse_args()

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device).eval()
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps   # read by policy.reset(); chunk_size stays 200
    if args.temporal_ensemble is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensemble, policy.config.chunk_size)
    policy.config.device = device     # checkpoints trained on CUDA carry device="cuda" in their config
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    print(f"loaded ACT from {args.checkpoint} on {device}; chunk_size={policy.config.chunk_size} "
          f"n_action_steps={policy.config.n_action_steps}")

    img_hw = tuple(policy.config.input_features[list(CAMERAS)[0]].shape[1:])
    env = RackCableEnv(port=args.port, img_hw=img_hw, randomize=not args.no_randomize,
                       render=True, max_seconds=args.max_seconds, rand_scale=args.rand_scale)
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed0 + ep)
        policy.reset()
        frames, t0, ok, trunc = [], time.time(), False, False
        while True:
            with torch.no_grad():
                action = policy.select_action(pre(to_batch(obs, device)))
                action = post(action)
            a = action.squeeze(0).detach().cpu().numpy()
            obs, _, term, trunc, info = env.step(a)
            if args.video_dir:
                frames.append(np.concatenate([obs[k] for k in CAMERAS], axis=1))
            if term or trunc:
                ok = bool(info["is_success"])
                break
        e = info["tip_error_mm"]
        results.append(ok)
        print(f"ep {ep:2d} seed {args.seed0 + ep}: {'SUCCESS' if ok else 'fail   '} "
              f"t={info['time']:5.1f}s  tip-port err (mm) {np.round(e, 1)}  [{time.time() - t0:.0f}s wall]", flush=True)
        if args.video_dir and frames:
            import imageio
            imageio.mimsave(Path(args.video_dir) / f"ep{ep:02d}_{'ok' if ok else 'fail'}.mp4", frames,
                            fps=50, macro_block_size=1)
    rate = float(np.mean(results)) if results else 0.0
    print(f"\nsuccess rate: {sum(results)}/{len(results)} = {rate:.0%}")


if __name__ == "__main__":
    main()
