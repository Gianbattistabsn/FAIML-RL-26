import os
import time
import random

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from helpers.parse_arguments import parse_args_train
from helpers.wandb_config import WandbMetricsCallback, get_wandb_config
from helpers.load_model import load_model
from helpers.make_model import make_model

"""
with normalization
python part2/train_sb3.py --env-type source --sampling-strategy none --timesteps 300000

without normalization
python part2/train_sb3.py --env-type source --sampling-strategy none --timesteps 300000 --no-vecnormalize

with UDR domain randomization
python part2/train_sb3.py --env-type source --sampling-strategy udr --timesteps 300000 --mass-range 0.5 2.0

with ADR domain randomization
python part2/train_sb3.py --env-type source --sampling-strategy adr --timesteps 300000 --mass-range 0.5 2.0 --adr-delta 0.2 --adr-buffer-size 20 --adr-perf-low -25.0 --adr-perf-high -10.0 --adr-boundary-prob 0.8
"""

def main() -> None:
    """
    Main function, trains a new model from scratch or loads an existing one.
    Offers the possibility to render at the end.
    """
    # get args
    args = parse_args_train()

    # seed environment
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # algorithm selection
    if input("use PPO? [y/n]\n>") == 'y':
        alg = "ppo"
    else:
        alg = "sac"

    # model's path (for saving if we will train it, or loading if we already have it)
    save_name = f"part2/models/{alg}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"

    # if the model being requested (algorithm + timesteps) was already trained, it can be loaded
    if input("Want to load existing model? [y/n]\n>") == 'y':
        model = load_model(args, save_name, alg)
    else:
        # make model
        model, n_envs, hparams, device = make_model(alg, save_name, args)

        # callbacks
        callbacks = [CheckpointCallback(
            save_freq=200000,
            save_path=f"{save_name}/checkpoints",
            name_prefix="model")
        ]

        # wandb setup
        if not args.no_wandb:

            # get wandb stuff
            config, run_tags = get_wandb_config(alg, args, device, hparams, n_envs, model)

            # init run
            run = wandb.init(
                project=args.wandb_project,
                entity=args.entity,
                name=args.run_name or save_name.replace("/", "_"),
                config=config,
                tags=run_tags,
                sync_tensorboard=False,
                save_code=True,
            )
            # add wandb callbacks
            wandb_cb = WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"wandb/{run.id}",
                verbose=2,
            )
            step_cb = WandbMetricsCallback(log_freq=500, sampling_strategy=args.sampling_strategy)
            callbacks += [wandb_cb, step_cb]

        # train model
        model.learn(
            total_timesteps = args.timesteps,
            callback = callbacks,
            progress_bar = True,
        )


        # save model as .zip (+ .pkl if vecnorm)
        model.save(save_name)
        vec_env = model.get_vec_normalize_env()
        if not args.no_vecnormalize and isinstance(vec_env, VecNormalize):
            vec_env.save(f"{save_name}_vecnormalize.pkl")
        print(f"Model saved to {save_name}.zip")

        # end wandb run
        if not args.no_wandb:
            wandb.finish()

    # Decide whether to render some episodes or not, regardless of model being trained or loaded
    if input("want to render? [y/n]\n>") == 'y':
        # setup environment
        env_type_render = args.env_type
        render_env = DummyVecEnv([
            lambda: gym.make(
                id="PandaPush-v3",
                render_mode="human",
                type=env_type_render,
                reward_type="dense"
            )
        ])

        # load the normalized env if needed
        vecnorm_path = f"{save_name}_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            render_env = VecNormalize.load(vecnorm_path, render_env)
            render_env.training = False    # rendering is not treated as training, no updates happen
            render_env.norm_reward = False  # rewards are not normalized during rendering (previously done in training)
            print(f"VecNormalize loaded from {vecnorm_path}")
        else:
            print(f"[WARNING] {vecnorm_path} not found, rendering without normalization")

        # start rendering
        render = True
        while render:
            n_episodes = int(input("insert n_episodes:\n>"))

            for ep in range(n_episodes):
                print(f"\n{'='*40}")
                print(f"  Episode {ep+1} / {n_episodes}")
                print(f"{'='*40}")
                time.sleep(1)

                obs = render_env.reset()
                done = False
                cumsum = 0.0
                step = 0

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = render_env.step(action)
                    cumsum += float(reward[0])
                    step += 1
                    done = bool(dones[0])

                    if step % 20 == 0:
                        print(f"  step={step:4d}  cumulative_reward={cumsum:7.3f}", flush=True)

                    time.sleep(0.2)

                success = infos[0].get("is_success", None) if infos else None
                success_str = f"  success={bool(success)}" if success is not None else ""
                print(f"\n  Episode {ep+1} done — steps={step}  return={cumsum:.3f}{success_str}")
                print(f"{'='*40}")
                time.sleep(1)

            # Ask whether to keep rendering or not
            render = input("\nwant to render again? [y/n]\n>")
            render = (render == 'y')

        render_env.close()


if __name__ == "__main__":
    main()
