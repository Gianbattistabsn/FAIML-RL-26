"""
Kaggle-adapted training script – PPO / SAC on PandaPush-v3
No interactive input() – everything is controlled via CLI arguments.

Example (PPO, 10 M steps, standard SB3 hyper-parameters):
    python part2/train_sb3.py \
        --algo ppo \
        --env-type source \
        --sampling-strategy none \
        --timesteps 10000000

Load an existing model:
    python part2/train_sb3.py --algo ppo --load

Skip W&B:
    python part2/train_sb3.py --algo ppo --no-wandb
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import panda_gym          # type: ignore[import-not-found]
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rand_wrapper import RandomizationWrapper   # type: ignore[import-not-found]


# ──────────────────────────────────────────────────────────────
# W&B metrics callback
# ──────────────────────────────────────────────────────────────

class WandbMetricsCallback(BaseCallback):
    """
    Logs per-episode reward, length, and success-rate to W&B,
    plus all SB3 training losses / metrics.
    """

    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int]   = []
        self._ep_successes: list[float] = []
        self._cur_rewards: list[float] = []
        self._cur_lengths: list[int]   = []

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._cur_rewards = [0.0] * n
        self._cur_lengths = [0]   * n

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        infos   = self.locals.get("infos",   [])

        for i, (r, done, info) in enumerate(zip(rewards, dones, infos)):
            self._cur_rewards[i] += float(r)
            self._cur_lengths[i] += 1
            if done:
                self._ep_rewards.append(self._cur_rewards[i])
                self._ep_lengths.append(self._cur_lengths[i])
                if "is_success" in info:
                    self._ep_successes.append(float(info["is_success"]))
                self._cur_rewards[i] = 0.0
                self._cur_lengths[i] = 0

        if self.n_calls % self.log_freq == 0:
            metrics: dict = {}

            if self._ep_rewards:
                w = self._ep_rewards[-100:]
                metrics["rollout/ep_rew_mean"] = float(np.mean(w))
                metrics["rollout/ep_rew_max"]  = float(np.max(w))
                metrics["rollout/ep_rew_min"]  = float(np.min(w))
            if self._ep_lengths:
                metrics["rollout/ep_len_mean"] = float(np.mean(self._ep_lengths[-100:]))
            if self._ep_successes:
                metrics["rollout/success_rate"] = float(np.mean(self._ep_successes[-100:]))

            for k, v in self.logger.name_to_value.items():
                if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and np.isnan(v)
                ):
                    metrics[k] = v

            metrics["train/n_episodes"]    = len(self._ep_rewards)
            metrics["train/num_timesteps"] = self.num_timesteps

            wandb.log(metrics, step=self.num_timesteps)

        return True


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO/SAC on PandaPush-v3 (Kaggle)")

    p.add_argument("--algo",               type=str,   default="ppo",
                   choices=["ppo", "sac"],
                   help="RL algorithm to use (default: ppo)")
    p.add_argument("--sampling-strategy",  type=str,   default="none",
                   choices=["none", "udr", "adr"])
    p.add_argument("--env-type",           type=str,   default="source",
                   choices=["source", "target"])
    p.add_argument("--timesteps",          type=int,   default=10_000_000,
                   help="Total training timesteps (default: 10 000 000)")

    # W&B
    p.add_argument("--wandb-project",      type=str,   default="RL-2026-AGGG")
    p.add_argument("--entity",             type=str,
                   default="gianbattista-busonera-politecnico-di-torino")
    p.add_argument("--run-name",           type=str,   default=None)
    p.add_argument("--no-wandb",           action="store_true")

    # Misc
    p.add_argument("--no-vecnormalize",    action="store_true",
                   help="Disable VecNormalize")
    p.add_argument("--load",               action="store_true",
                   help="Load a pre-trained model instead of training")
    p.add_argument("--n-envs",             type=int,   default=None,
                   help="Override number of parallel envs")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    alg      = args.algo                              # "ppo" | "sac"
    use_ppo  = (alg == "ppo")

    # ── paths (Kaggle: /kaggle/working/) ──────────────────────
    BASE     = "/kaggle/working"
    save_name = (
        f"{BASE}/FAIML-RL-26/part2/models"
        f"/{alg}_push_{args.sampling_strategy}"
        f"_{args.env_type}_{args.timesteps // 1_000}k"
    )
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    # ── device ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")

    # ══════════════════════════════════════════════════════════
    # LOAD branch
    # ══════════════════════════════════════════════════════════
    if args.load:
        print(f"\n[LOAD] {save_name}.zip")
        load_env = make_vec_env(
            lambda: gym.make(
                "PandaPush-v3",
                render_mode="rgb_array",
                type=args.env_type,
                reward_type="dense",
            ),
            n_envs=1,
        )
        vecnorm_path = f"{save_name}_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            load_env = VecNormalize.load(vecnorm_path, load_env)
            load_env.training  = False
            load_env.norm_reward = False
            print(f"  VecNormalize loaded from {vecnorm_path}")
        else:
            print(f"  [WARNING] {vecnorm_path} not found – no normalization")

        Algo  = PPO if use_ppo else SAC
        model = Algo.load(f"{save_name}.zip", env=load_env)
        print("Model loaded successfully.")
        return   # nothing more to do on Kaggle (no render_mode="human")

    # ══════════════════════════════════════════════════════════
    # TRAIN branch
    # ══════════════════════════════════════════════════════════

    # ── number of parallel envs ───────────────────────────────
    n_envs = args.n_envs or (8 if use_ppo else 4)

    # ── hyper-parameters ─────────────────────────────────────
    #
    # PPO: standard SB3 defaults are used intentionally.
    # Passing an empty dict means every kwarg falls back to
    # stable-baselines3's built-in defaults, which are already
    # well-tuned for continuous-control tasks.
    #
    # SAC: same philosophy – standard defaults kept.
    # Uncomment / tweak if you need custom values.
    #
    ppo_hparams: dict = {}
    # ppo_hparams = {
    #     "learning_rate":        3e-4,
    #     "n_steps":              2048,
    #     "batch_size":           64,
    #     "n_epochs":             10,
    #     "gamma":                0.99,
    #     "gae_lambda":           0.95,
    #     "clip_range":           0.2,
    #     "ent_coef":             0.0,
    #     "vf_coef":              0.5,
    #     "max_grad_norm":        0.5,
    #     "normalize_advantage":  True,
    # }

    sac_hparams: dict = {}
    # sac_hparams = {
    #     "learning_rate":    3e-4,
    #     "buffer_size":      int(1e6),
    #     "batch_size":       256,
    #     "ent_coef":         "auto",
    #     "gamma":            0.98,
    #     "tau":              0.005,
    #     "learning_starts":  10_000,
    #     "train_freq":       1,
    #     "gradient_steps":   4,
    # }

    hparams = ppo_hparams if use_ppo else sac_hparams

    # ── W&B ──────────────────────────────────────────────────
    base_config = {
        "algorithm":          alg,
        "env_type":           args.env_type,
        "sampling_strategy":  args.sampling_strategy,
        "timesteps":          args.timesteps,
        "n_envs":             n_envs,
        "device":             device,
        "vec_normalize":      not args.no_vecnormalize,
        **hparams,
    }

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
        wandb_cb = step_cb = None

    # ── env factory ──────────────────────────────────────────
    def make_env():
        env = gym.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            type=args.env_type,
            reward_type="dense",
        )
        if args.sampling_strategy != "none":
            env = RandomizationWrapper(env, sampling_strategy=args.sampling_strategy)
        return env

    # ── vec env + optional normalization ─────────────────────
    vec_env = make_vec_env(make_env, n_envs=n_envs)

    if not args.no_vecnormalize:
        norm_reward = not use_ppo        # SAC: no reward norm; PPO: yes
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=norm_reward,
        )
        vec_env.training = True

    # ── model ─────────────────────────────────────────────────
    Algo  = PPO if use_ppo else SAC
    model = Algo(
        policy="MultiInputPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        tensorboard_log=f"{save_name}/logs",
        **hparams,
    )

    print(f"\nTraining {alg.upper()} for {args.timesteps:,} steps "
          f"on {n_envs} envs ({device}) …\n")

    # ── callbacks ─────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=f"{save_name}/checkpoints",
        name_prefix="model",
    )
    callbacks = [checkpoint_cb]
    if wandb_cb:
        callbacks += [wandb_cb, step_cb]

    # ── learn ─────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── save ──────────────────────────────────────────────────
    model.save(save_name)
    print(f"\nModel saved → {save_name}.zip")

    if not args.no_vecnormalize and isinstance(vec_env, VecNormalize):
        vec_env.save(f"{save_name}_vecnormalize.pkl")
        print(f"VecNormalize saved → {save_name}_vecnormalize.pkl")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
