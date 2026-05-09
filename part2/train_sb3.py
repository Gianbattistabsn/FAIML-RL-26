# first time only: pip install wandb
# wandb login
# -------------- WITH WANDB ---------
#run comand: python .\part2\train_sb3.py --env-type source --sampling-strategy none --timesteps 500000
#run comand: python .\part2\train_sb3.py --env-type source --sampling-strategy none --timesteps 50000
import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import DDPG
from rand_wrapper import RandomizationWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

import time
import torch
import wandb
from wandb.integration.sb3 import WandbCallback


class WandbMetricsCallback(BaseCallback):
    """
    Logga su W&B:
      - reward/length/success rate per episodio (tracciati step-by-step, no Monitor wrapper necessario)
      - tutte le loss e metriche di training dal logger interno di SB3
    """
    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        # buffer episodio corrente per ogni env
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_successes: list[float] = []
        self._cur_rewards: list[float] = []
        self._cur_lengths: list[int] = []

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._cur_rewards = [0.0] * n
        self._cur_lengths = [0] * n

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones", [])
        infos   = self.locals.get("infos", [])

        for i, (r, done, info) in enumerate(zip(rewards, dones, infos)):
            self._cur_rewards[i] += float(r)
            self._cur_lengths[i] += 1
            if done:
                self._ep_rewards.append(self._cur_rewards[i])
                self._ep_lengths.append(self._cur_lengths[i])
                # success rate per goal-conditioned env (PandaPush)
                if "is_success" in info:
                    self._ep_successes.append(float(info["is_success"]))
                self._cur_rewards[i] = 0.0
                self._cur_lengths[i] = 0

        if self.n_calls % self.log_freq == 0:
            metrics: dict = {}

            # metriche episodio (finestra 100 ep)
            if self._ep_rewards:
                w = self._ep_rewards[-100:]
                metrics["rollout/ep_rew_mean"]  = float(np.mean(w))
                metrics["rollout/ep_rew_max"]   = float(np.max(w))
                metrics["rollout/ep_rew_min"]   = float(np.min(w))
            if self._ep_lengths:
                metrics["rollout/ep_len_mean"]  = float(np.mean(self._ep_lengths[-100:]))
            if self._ep_successes:
                metrics["rollout/success_rate"] = float(np.mean(self._ep_successes[-100:]))

            # loss e metriche di training dal logger SB3
            for k, v in self.logger.name_to_value.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    metrics[k] = v

            # info aggiuntive
            metrics["train/n_episodes"]     = len(self._ep_rewards)
            metrics["train/num_timesteps"]  = self.num_timesteps

            wandb.log(metrics, step=self.num_timesteps)
        return True


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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="RL-2026-AGGG",
        help="W&B project name (shared with teammates to log to the same project)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="gianbattista-busonera-politecnico-di-torino",
        help="W&B entity: your username or a shared team name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if omitted)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
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
        print("xpu available", torch.xpu.is_available())
        
        print("using", device)

        # --- W&B setup ---
        base_config = {
            "algorithm": alg,
            "env_type": args.env_type,
            "sampling_strategy": args.sampling_strategy,
            "timesteps": args.timesteps,
            "device": device,
        }
        if use_PPO:
            base_config.update({
                "learning_rate": 7e-5,
                "n_steps": 2048,
                "batch_size": 256,
                "n_epochs": 10,
                "clip_range": 0.1,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.001,
                "n_envs": 4,
            })
        else:
            base_config.update({
                "learning_rate": 5e-4,
                "buffer_size": int(1e6),
                "batch_size": 256,
                "ent_coef": "auto",
                "gamma": 0.99,
                "tau": 0.005,
                "learning_starts": 1000,
                "train_freq": 1,
                "gradient_steps": 1,
                "net_arch": [256, 256, 256],
                "activation_fn": "ReLU",
                "share_features_extractor": True,
            })

        if not args.no_wandb:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.entity,
                name=args.run_name or save_name.replace("/", "_"),
                config=base_config,
                sync_tensorboard=False,
                save_code=True,
            )
            wandb_cb = WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"wandb/{run.id}",
                verbose=2,
            )
            step_cb = WandbMetricsCallback(log_freq=500)
        else:
            wandb_cb = None
            step_cb = None
        # ------------------

        if (use_PPO):
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

            callbacks = [checkpoint_cb] + ([wandb_cb, step_cb] if wandb_cb else [])
            model.learn(
                total_timesteps = args.timesteps,
                callback = callbacks,
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
                policy="MultiInputPolicy",
                env=vec_env,
                device=device,
                verbose=1,
                learning_rate=5e-4,
                buffer_size=int(1e6),
                batch_size=256,
                ent_coef="auto",
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                learning_starts=1000,
                target_entropy="auto",
                tensorboard_log=f"{save_name}/logs",
                policy_kwargs=dict(
                    net_arch=[256, 256, 256],
                    activation_fn=torch.nn.ReLU,
                    share_features_extractor=True,
                ),
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


        model.save(save_name)
        print(f"Model saved to {save_name}.zip")

        if not args.no_wandb:
            wandb.finish()







    
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
                print(f"\n--- Episode {ep+1}/{n_episodes} ---")
                time.sleep(1.0) 

                while not done:  # Until the episode is over
                    time.sleep(1/30)

                    action, _ = model.predict(state, deterministic = True)
                    state, reward, terminated, truncated, _ = render_env.step(action)  # Step the simulator to the next timestep
                    cumsum += reward
                    done = terminated or truncated

                    if render:
                        render_env.render()
                
                print(f"return for episode {ep+1} = {cumsum:.2f}")
                time.sleep(2.0)  # pausa dopo l'episodio per leggere il risultato
            
            render = input("want to render? [y/n]\n>")
            if (render == 'y'):
                render = True
            else:
                render = False


        render_env.close()


if __name__ == "__main__":
    main()
