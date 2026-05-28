import argparse
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, SAC
import wandb


class WandbMetricsCallback(BaseCallback):
    """
    Logga su W&B:
      - reward/length/success rate per episodio (tracciati step-by-step, no Monitor wrapper necessario)
      - tutte le loss e metriche di training dal logger interno di SB3
    """
    def __init__(self, log_freq: int = 500, sampling_strategy: str = "none", verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.sampling_strategy = sampling_strategy
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

            # domain randomization metrics (UDR / ADR)
            if self.sampling_strategy in ("udr", "adr", "badr"):
                mass_mins = self.training_env.get_attr("mass_min")
                mass_maxs = self.training_env.get_attr("mass_max")
                metrics["dr/mass_min"]         = float(np.mean(mass_mins))
                metrics["dr/mass_max"]         = float(np.mean(mass_maxs))
                metrics["dr/mass_range_width"] = float(np.mean(
                    [mx - mn for mn, mx in zip(mass_mins, mass_maxs)]
                ))
                if self.sampling_strategy in ("adr", "badr"):
                    sample_types = self.training_env.get_attr("last_sample_type")
                    for tag in ("adr_low", "adr_high", "adr_interior"):
                        metrics[f"dr/{tag}_frac"] = float(
                            sum(1 for s in sample_types if s == tag) / len(sample_types)
                        )
            wandb.log(metrics, step=self.num_timesteps)
        return True

def get_wandb_config(alg: str,
                     args: argparse.Namespace,
                     device: str,
                     hparams: dict,
                     n_envs: int,
                     model: PPO|SAC) -> tuple[dict, list]:
    # wandb configuration
    config = {
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
        **hparams,
        "n_envs": n_envs,
        "vec_normalize": not args.no_vecnormalize,
        "norm_reward": getattr(model.get_vec_normalize_env(), "norm_reward", False)
    }

    # set tags for the run
    run_tags = [
        alg,  # "ppo" / "sac"
        args.sampling_strategy,  # "none" / "udr" / "adr" / "badr
        args.env_type,  # "source" / "target"
        f"seed{args.seed}",  # "seed42" — group runs of same seed
        f"ts{args.timesteps // 1000}k",  # "ts300k" — compare run lengths
        "vecnorm" if not args.no_vecnormalize else "no-vecnorm"
    ]
    if args.sampling_strategy == "udr":
        run_tags += [f"udr_range{args.mass_range[0]:.2f}-{args.mass_range[1]:.2f}"]
    elif args.sampling_strategy == "adr":
        run_tags += [f"delta{args.adr_delta:.2f}", f"bp{args.adr_boundary_prob:.2f}"]
        run_tags += [f"adr_start{(float(args.mass_range[1]) + float(args.mass_range[0])) / 2:.2f}"]
    elif args.sampling_strategy == "badr":
        run_tags += [f"delta{args.adr_delta:.2f}", f"bp{args.adr_boundary_prob:.2f}"]
        run_tags += [f"badr_start{(float(args.mass_range[1]) + float(args.mass_range[0])) / 2:.2f}"]
        run_tags += [f"badr_limits{args.mass_range[0]:.2f}-{args.mass_range[1]:.2f}"]
    return config, run_tags
