#run comand: python .\part2\eval_sb3.py --model-path ./part2/models/ppo_push_none_source_700k.zip --episodes 30 --stochastic --render --env-type target
"""
# with normalization (default)
python .\part2\eval_sb3.py --model-path ./part2/models/sac_push_none_source_300k.zip --episodes 30 --render

# without normalization
python .\part2\eval_sb3.py --model-path ./part2/models/sac_push_none_source_300k.zip --episodes 30 --render --no-vecnormalize
"""

import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import panda_gym  # noqa: F401 - required so Panda envs are registered

from helpers.parse_arguments import parse_args_eval


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str, use_vecnormalize: bool = True) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"
    env = DummyVecEnv([lambda: gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")])

    # Auto-detect VecNormalize stats (.pkl) from model path
    vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if not vecnorm_path.endswith("_vecnormalize.pkl"):
        vecnorm_path = model_path + "_vecnormalize.pkl"
    if use_vecnormalize and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize stats caricate da {vecnorm_path}")
    elif use_vecnormalize:
        print(f"[WARNING] {vecnorm_path} non trovato, eval senza normalizzazione")
    else:
        print("VecNormalize disabilitata (--no-vecnormalize)")

    if "ppo" in model_path:
        model = PPO.load(model_path, env=env)
    else:
        model = SAC.load(model_path, env=env)

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        print(f"\n{'='*40}")
        print(f"  Episode {episode} / {n_episodes}")
        print(f"{'='*40}")
        if render:
            time.sleep(0.5)

        obs = env.reset()
        done = False
        episode_return = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)
            episode_return += float(reward[0])
            step += 1
            done = bool(dones[0])

            if step % 20 == 0:
                print(f"  step={step:4d}  cumulative_reward={episode_return:7.3f}", flush=True)
            if render:
                time.sleep(0.2)

        success = infos[0].get("is_success", None) if infos else None
        success_str = f"  success={bool(success)}" if success is not None else ""
        print(f"\n  Episode {episode} done — steps={step}  return={episode_return:.3f}{success_str}")
        print(f"{'='*40}")
        if render:
            time.sleep(0.5)

        episode_returns.append(episode_return)

        if infos and "is_success" in infos[0]:
            successes.append(float(infos[0]["is_success"]))

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")

if __name__ == "__main__":
    args = parse_args_eval()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        use_vecnormalize=not args.no_vecnormalize,
    )
