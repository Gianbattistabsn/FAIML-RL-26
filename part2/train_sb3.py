#run comand: python .\part2\train_sb3.py --env-type source --sampling-strategy none --timesteps 500000
#run comand: python .\part2\train_sb3.py --env-type source --sampling-strategy none --timesteps 5000


import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import DDPG
from rand_wrapper import RandomizationWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import time
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    if args.sampling_strategy != "none":
        env = RandomizationWrapper(env, sampling_strategy = args.sampling_strategy)


    save_name = f"part2/models/sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    vec_env = DummyVecEnv([lambda : env])


    use_gpu = False
    if use_gpu:
        device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    else:
        device = "cpu"

    model = SAC(
        policy = "MultiInputPolicy",
        env = vec_env,
        device = device,
        verbose = 1,
        learning_rate = 1e-3,
        buffer_size = int(args.timesteps / 2),
        batch_size = 256,
        tensorboard_log = f"{save_name}/logs",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq = int(args.timesteps / 2),
        save_path = f"{save_name}/checkpoints",
        name_prefix = "model",
    )

    model.learn(
        total_timesteps = args.timesteps,
        callback = checkpoint_cb,
        progress_bar = True,
    )


    model.save(save_name)
    print(f"Model saved to {save_name}.zip")



    render = True

    render_env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type=args.env_type,
        reward_type="dense",
    )


    n_episodes = 5

    for ep in range(n_episodes):  
        state, info = render_env.reset()  # Reset environment to initial state
        done = False
        cumsum = 0.0

        while not done:  # Until the episode is over
            time.sleep(0.02)

            action, _ = model.predict(state, deterministic = True)
            state, reward, terminated, truncated, _ = render_env.step(action)  # Step the simulator to the next timestep
            cumsum += reward
            done = terminated or truncated

            if render:
                render_env.render()
        
        print(f"\nreturn for episode {ep+1} = {cumsum:.2f}\n")

    render_env.close()


if __name__ == "__main__":
    main()