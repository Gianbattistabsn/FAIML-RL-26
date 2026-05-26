# standard library
import os
import time

# external
import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

# internal
from parse_arguments import parse_args_train
from rand_wrapper import RandomizationWrapper
from wandb_callback import WandbMetricsCallback
import random
import numpy as np



# first time only:
# >pip install wandb
# >wandb login
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
    args = parse_args_train()

    # Algorithm selection
    use_PPO = input("use PPO? [y/n]\n>")
    if (use_PPO == 'y'):
        use_PPO = True
        alg = "ppo"
    else:
        use_PPO = False
        alg = "sac"


    training_seed = 42

    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    



    # Model's path (for saving if we will train it, or loading if we already have it)
    save_name = f"part2/models/{alg}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"


    # If the model being requested (algorithm + timesteps) was already trained, it can be loaded
    load = input("want to load model? [y/n]\n>")
    if (load == 'y'):
        load = True
    else:
        load = False



    if load:

        # Creation of one vectorized environment
        load_vec_env = make_vec_env(
            lambda: gym.make("PandaPush-v3", render_mode="rgb_array",
                             type=args.env_type, reward_type="dense"),
            n_envs=1,
        )

        # Loading of its normalization if the model was trained with normalization on
        vecnorm_path = f"{save_name}_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            load_vec_env = VecNormalize.load(vecnorm_path, load_vec_env)
            load_vec_env.training = False
            load_vec_env.norm_reward = False
            print(f"VecNormalize stats loaded from {vecnorm_path}")
        else:
            print(f"[WARNING] {vecnorm_path} not found, model loaded without normalization")

        if use_PPO:
            model = PPO.load(f"{save_name}.zip", env=load_vec_env)
        else:
            model = SAC.load(f"{save_name}.zip", env=load_vec_env)




    # Model has to be trained
    else:
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
        print(torch.version.cuda)
        
        # Check for GPU availability
        use_gpu = True
        if use_gpu:
            device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
        
        else:
            device = "cpu"
        
        
        print("using", device)


        # Parameters configuration, only worry about the algorithm you want to train right now
        ppo_hparams = {
            "learning_rate": 5e-4,
            "n_steps": 4096,
            "batch_size": 512,
            "n_epochs": 8,
            "clip_range": 0.15,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "ent_coef": 0.005,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "seed": training_seed
        }
        # Choose the number of environments to use during PPO training for paralellization
        ppo_n_envs = 8

        sac_hparams = {
            "learning_rate": 3e-4,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "ent_coef": "auto",
            "gamma": 0.9812,
            "tau": 0.005,
            "learning_starts": 10_000,
            "train_freq": 1,
            "gradient_steps": 4,
            "seed": training_seed
        }
        # Choose the number of environments to use during SAC training for paralellization
        sac_n_envs = 4

        



        # wandb configuration
        base_config = {
            "algorithm": alg,
            "env_type": args.env_type,
            "sampling_strategy": args.sampling_strategy,
            "timesteps": args.timesteps,
            "device": device,
            "mass_range_min": args.mass_range[0],
            "mass_range_max": args.mass_range[1],
            "seed": args.seed,
            "adr_delta": args.adr_delta,
            "adr_buffer_size": args.adr_buffer_size,
            "adr_perf_low": args.adr_perf_low,
            "adr_perf_high": args.adr_perf_high,
            "adr_boundary_prob": args.adr_boundary_prob,
            "training_seed": training_seed
        }

        if use_PPO:
            base_config.update({
                **ppo_hparams,
                "n_envs": ppo_n_envs,
                "vec_normalize": not args.no_vecnormalize,
                "norm_reward": not args.no_vecnormalize,
            })
        else:
            base_config.update({
                **sac_hparams,
                "n_envs": sac_n_envs,
                "vec_normalize": not args.no_vecnormalize,
                "norm_reward": False,
            })

        if not args.no_wandb:
            ##############################
            run_tags = [
                alg,                             # "ppo" / "sac"
                args.sampling_strategy,          # "none" / "udr" / "adr"
                args.env_type,                   # "source" / "target"
                f"seed{args.seed}",              # "seed42" — group runs of same seed
                f"ts{args.timesteps // 1000}k",  # "ts300k" — compare run lengths
                "vecnorm" if not args.no_vecnormalize else "no-vecnorm"
            ]
            if args.sampling_strategy == "udr":
                run_tags += [f"udr_range{args.mass_range[0]:.2f}-{args.mass_range[1]:.2f}"]
            elif args.sampling_strategy == "adr":
                run_tags += [f"delta{args.adr_delta:.2f}", f"bp{args.adr_boundary_prob:.2f}"]
                run_tags += [f"adr_start{(float(args.mass_range[1]) + float(args.mass_range[0])) /2:.2f}"]
            ##############################
            run = wandb.init(
                project=args.wandb_project,
                entity=args.entity,
                name=args.run_name or save_name.replace("/", "_"),
                config=base_config,
                tags=run_tags,
                sync_tensorboard=False,
                save_code=True,
            )
            wandb_cb = WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"wandb/{run.id}",
                verbose=2,
            )
            step_cb = WandbMetricsCallback(log_freq=500, sampling_strategy=args.sampling_strategy)
        else:
            wandb_cb = None
            step_cb = None




        # Custom 'make_env' function to properly use domain randomization if selected
        def make_env():
            env = gym.make(
                "PandaPush-v3",
                render_mode="rgb_array",
                type=args.env_type,
                reward_type="dense",
            )
            env = RandomizationWrapper(
                env,
                mode=args.sampling_strategy,
                mass_range=tuple(args.mass_range),
                seed=args.seed,
                verbose=args.verbose_wrapper,
                adr_delta=args.adr_delta,
                adr_buffer_size=args.adr_buffer_size,
                adr_perf_low=args.adr_perf_low,
                adr_perf_high=args.adr_perf_high,
                adr_boundary_prob=args.adr_boundary_prob,
            )
            return env






        # Actual model configuration
        if (use_PPO):

            # Creation of multiple vectorized environments to parallelize training
            vec_env = make_vec_env(make_env, n_envs=ppo_n_envs, seed = training_seed)

            # Enabling environment normalization if requested
            if not args.no_vecnormalize:
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
                vec_env.training = True
                vec_env.norm_reward = True


            # Model creation with selected parameters
            model = PPO(
                policy="MultiInputPolicy",
                env=vec_env,
                device=device,
                verbose=1,
                tensorboard_log=f"{save_name}/logs",

                **ppo_hparams
            )

            checkpoint_cb = CheckpointCallback(
                save_freq = 200000,
                save_path = f"{save_name}/checkpoints",
                name_prefix = "model",
            )

            callbacks = [checkpoint_cb] + ([wandb_cb, step_cb] if wandb_cb else [])

            # Model training
            model.learn(
                total_timesteps = args.timesteps,
                callback = callbacks,
                progress_bar = True,
            )


        # Same structure as PPO above
        else:
            vec_env = make_vec_env(make_env, n_envs=sac_n_envs, seed = training_seed)

            if not args.no_vecnormalize:
                vec_env = VecNormalize(
                    vec_env,
                    norm_obs=True,
                    norm_reward=False,
                )

            model = SAC(
                policy="MultiInputPolicy",
                env=vec_env,
                device=device,
                verbose=1,        
                tensorboard_log=f"{save_name}/logs",

                **sac_hparams
            )

            checkpoint_cb = CheckpointCallback(
                save_freq = 200000,
                save_path = f"{save_name}/checkpoints",
                name_prefix = "model",
            )

            callbacks = [checkpoint_cb] + ([wandb_cb, step_cb] if wandb_cb else [])

            model.learn(
                total_timesteps = args.timesteps,
                callback = callbacks,
                progress_bar = True,
            )


        # Saving model as .zip file
        model.save(save_name)
        if not args.no_vecnormalize and isinstance(vec_env, VecNormalize):
            vec_env.save(f"{save_name}_vecnormalize.pkl")
        print(f"Model saved to {save_name}.zip")

        if not args.no_wandb:
            wandb.finish()







    # Decide wether to render some episodes or not, regardless of model being trained or loaded
    render = input("want to render? [y/n]\n>")
    if (render == 'y'):

        # Setting up the proper environment
        env_type_render = args.env_type
        render_env = DummyVecEnv([lambda: gym.make(
            "PandaPush-v3",
            render_mode="human",
            type=env_type_render,
            reward_type="dense",
        )])


        # If the model was trained with env normalization, load the normalized env
        vecnorm_path = f"{save_name}_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            render_env = VecNormalize.load(vecnorm_path, render_env)
            render_env.training = False    # rendering is not treated as training, no updates happen
            render_env.norm_reward = False  # rewards are not normalized during rendering (previously done in training)
            print(f"VecNormalize loaded from {vecnorm_path}")
        else:
            print(f"[WARNING] {vecnorm_path} not found, rendering without normalization")


        # Start rendering
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

            # Ask wether to keep rendering or not
            render = input("\nwant to render again? [y/n]\n>")
            render = (render == 'y')

        render_env.close()


if __name__ == "__main__":
    main()