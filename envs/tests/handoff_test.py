"""Scripted bimanual handoff (see envs.experts.object_handoff): the left arm takes the bar by its top, the right arm
takes it 12.5 cm above its bottom, the left lets go, and the right sets the bar on the pad.

    .venv/bin/python -m envs.tests.handoff_test [n_seeds]
"""

import sys

import envs

from ..experts import object_handoff

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    wins = 0
    for seed in range(n):
        env = envs.make("object_handoff", render=False)
        env.reset(seed=seed)
        ev = object_handoff(env, verbose=True)
        wins += ev["success"]
        print(f"seed {seed}: success={ev['success']} stage={ev['stage']} progress={ev['progress']:.2f}")
    print(f"{wins}/{n} scripted handoffs succeeded")
