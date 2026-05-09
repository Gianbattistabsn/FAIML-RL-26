#run comand: python .\part2\train_sb3.py --env-type source --sampling-strategy none --timesteps 1500000
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
from stable_baselines3.common.vec_env import VecNormalize

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

    use_PPO = input("use PPO? [y/n]\n>")
    if (use_PPO == 'y'):
        use_PPO = True
        alg = "ppo"
    else:
        use_PPO = False
        alg = "sac"


    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    if args.sampling_strategy != "none":
        env = RandomizationWrapper(env, sampling_strategy = args.sampling_strategy)


    save_name = f"part2/models/{alg}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"


    load = input("want to load model? [y/n]\n>")
    if (load == 'y'):
        load = True
    else:
        load = False



    if load:
        if use_PPO:
            model = PPO.load(f"{save_name}.zip")
        else:
            model = SAC.load(f"{save_name}.zip")


    else:
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
        print(torch.version.cuda)
        
        use_gpu = True
        if use_gpu:
            device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
        else:
            device = "cpu"

        if use_PPO:
            vec_env = make_vec_env(
                lambda: gym.make(
                    "PandaPush-v3",
                    render_mode="rgb_array",
                    type=args.env_type,
                    reward_type="dense",
                ),
                n_envs=4
            )

            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
            vec_env.training = True
            vec_env.norm_reward = True

            model = PPO(
                policy="MultiInputPolicy",
                env=vec_env,
                device=device,
                verbose=1,

                learning_rate=1e-4,
                n_steps=1024,
                batch_size=256,
                n_epochs=10,

                gamma=0.99,
                gae_lambda=0.95,

                clip_range=0.2,
                ent_coef=0.005,

                normalize_advantage=True,
                tensorboard_log=f"{save_name}/logs",
            )

            checkpoint_cb = CheckpointCallback(
                save_freq = 200000,
                save_path = f"{save_name}/checkpoints",
                name_prefix = "model",
            )

            model.learn(
                total_timesteps = args.timesteps,
                callback = checkpoint_cb,
                progress_bar = True,
            )


        else:
            vec_env = make_vec_env(
                lambda: gym.make(
                    "PandaPush-v3",
                    render_mode="rgb_array",
                    type=args.env_type,
                    reward_type="dense",
                ),
                n_envs=4,   # good default for SAC on robotics
            )

            # Optional but HIGHLY recommended for stability
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,  # keep reward stable for SAC
            )

            model = SAC(
                learning_rate=3e-4,
                buffer_size=int(1e6),
                batch_size=256,
                gamma=0.99,          # keep default
                tau=0.005,           # keep default
                ent_coef = "auto",        # keep fixed
                train_freq = 1,
                gradient_steps = 1,
                learning_starts = 10_000,

                policy="MultiInputPolicy",
                env=vec_env,
                device=device,
                verbose=1,
                tensorboard_log=f"{save_name}/logs",
            )

            checkpoint_cb = CheckpointCallback(
                save_freq = 200000,
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







    
    render = input("want to render? [y/n]\n>")
    if (render == 'y'):
        render = True

        render_env = gym.make(
            "PandaPush-v3",
            render_mode="human" if render else "rgb_array",
            type=args.env_type,
            reward_type="dense",
        )

        while (render):
            n_episodes = int(input("insert n_episodes:\n>"))

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
            
            render = input("want to render? [y/n]\n>")
            if (render == 'y'):
                render = True
            else:
                render = False


        render_env.close()


if __name__ == "__main__":
    main()