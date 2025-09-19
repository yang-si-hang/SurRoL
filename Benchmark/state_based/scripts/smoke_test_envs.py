# Phase 0: Non-interactive smoke test
# This verifies that envs are registered and step-able.
import gym
import numpy as np

# Per-script registration (Option B). If you edited Gym globally, this import is still safe.
import surrol.gym  # registers SurRoL tasks with Gym

def run_smoke(env_id, steps=10):
    print(f"\n[Smoke] Creating {env_id}")
    env = gym.make(env_id)
    obs = env.reset()
    print(f"Obs type={type(obs)}")
    for t in range(steps):
        # Sample a random action in the valid range
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        if done:
            obs = env.reset()
    env.close()
    print(f"[OK] {env_id} ran {steps} steps.")

if __name__ == "__main__":
    run_smoke("ECMReach-v0", steps=10)       # non-contact camera reach
    run_smoke("NeedleReachRL-v0", steps=10)  # simple reach task