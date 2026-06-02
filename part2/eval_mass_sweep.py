import os
import json
import random

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import panda_gym  # noqa: F401 - required so Panda envs are registered
import torch

from helpers.parse_arguments import parse_args_eval


def evaluate(model_path: str,
             n_episodes: int,
             env_type: str,
             masses: list[float] = None,
             deterministic: bool = True,
             use_vecnormalize: bool = True) -> None:

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    # Auto-detect VecNormalize stats (.pkl) from model path
    vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if not vecnorm_path.endswith("_vecnormalize.pkl"):
        vecnorm_path = model_path + "_vecnormalize.pkl"

    if use_vecnormalize and os.path.exists(vecnorm_path):
        print(f"VecNormalize stats loaded from {vecnorm_path}")
    elif use_vecnormalize:
        print(f"[WARNING] {vecnorm_path} not found, evaluation in progress without normalization...")
    else:
        print("VecNormalize disabled (--no-vecnormalize)")

    mean_returns = []
    stds_returns = []
    min_returns = []
    max_returns = []
    success_rates = []
    seed = 42

    masses = masses if masses is not None else [1.0]
    for mass in masses:
        # for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = DummyVecEnv(
            [lambda: gym.make(id="PandaPush-v3",
                              type=env_type,
                              seed=int(mass),
                              reward_type="dense")]
        )

        if use_vecnormalize and os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

        if "ppo" in model_path:
            model = PPO.load(model_path, env=env)
        else:
            model = SAC.load(model_path, env=env)

        # assign mass, manage both VecNormalize and plain DummyVecEnv as the outer wrapper
        base_vec = env.unwrapped if isinstance(env, VecNormalize) else env
        inner = base_vec.envs[0]
        sim = inner.unwrapped.task.sim
        object_body_id = sim._bodies_idx["object"]
        sim.physics_client.changeDynamics(bodyUniqueId=object_body_id,
                                          linkIndex=-1,
                                          mass=float(mass)) # new mass assigned
        episode_returns = []
        successes = []

        for episode in range(1, n_episodes + 1):

            obs = env.reset()
            done = False
            episode_return = 0.0
            step = 0
            infos = None

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = env.step(action)
                episode_return += float(reward[0])
                step += 1
                done = bool(dones[0])

            episode_returns.append(episode_return)

            if infos and "is_success" in infos[0]:
                successes.append(float(infos[0]["is_success"]))

        returns = np.array(episode_returns, dtype=np.float32)

        mean_return = float(returns.mean())
        std_return = float(returns.std())
        min_return = float(returns.min())
        max_return = float(returns.max())
        success_rate = float(np.mean(successes)) if successes else 0.0

        mean_returns.append(mean_return)
        stds_returns.append(std_return)
        min_returns.append(min_return)
        max_returns.append(max_return)
        success_rates.append(success_rate)

        print(f"\n**** Evaluation summary, mass = {mass} ****")
        print(f"Episodes: {n_episodes}")
        print(f"Mean return: {mean_return:.3f}")
        print(f"Std return: {std_return:.3f}")
        print(f"Min return: {min_return:.3f}")
        print(f"Max return: {max_return:.3f}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Mass check: {sim.physics_client.getDynamicsInfo(object_body_id, -1)[0]}")

    env.close()

    print(f"\n==== TOTAL SUMMARY ====")
    print(f"Number of masses evaluated: {len(masses)}")
    print(f"Masses evaluated: {masses}")
    print(f"Mean success rate: {np.array(success_rates).mean():.2%}")
    print(f"=======================")


    # save stats for plotting
    out_json = model_path.replace(".zip", "_sweep_stats.json")
    with open(out_json, "w") as f:
        json.dump({
            "masses": list(map(float, masses)),
            "mean_returns": mean_returns,
            "std_returns": stds_returns,
            "min_returns": min_returns,
            "max_returns": max_returns,
            "success_rates": success_rates,
        }, f, indent=4)
    print(f"\nStatistics saved to {out_json}")

if __name__ == "__main__":
    args = parse_args_eval()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        env_type=args.env_type,
        masses=args.masses,
        deterministic=not args.stochastic,
        use_vecnormalize=not args.no_vecnormalize
    )
