import argparse
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from helpers.make_env import make_env


def make_model(alg: str,
               save_name: str,
               args:argparse.Namespace,
               use_cuda: bool = True,
               verbose: bool = True) -> tuple[PPO|SAC, int, dict, str]:
    """
    Build a fresh SB3 model (PPO or SAC) with its vectorized, optionally normalized env.

    Args:
        alg: Algorithm identifier; "ppo" or "sac". Any other value is treated as SAC.
        save_name: Path stem used for TensorBoard logs (`{save_name}/logs`). Not used
            here for checkpointing; the caller is responsible for saving the model.
        args: Parsed CLI arguments. Must expose `seed` and `no_vecnormalize`, plus
            every attribute consumed by `make_env`.
        use_cuda: If True, prefer CUDA/XPU when available. If False, force CPU.
        verbose: If True, print device-detection info to stdout.

    Returns:
        tuple:
            - model (PPO | SAC): The constructed SB3 model, ready for `.learn(...)`.
            - n_envs (int): Number of parallel envs in the vectorized training env.
            - hparams (dict): The algorithm-specific hyperparameters used.
            - device (str): The resolved device string ("cuda", "xpu", or "cpu").
    """
    device = "cuda" if (use_cuda and torch.cuda.is_available()) \
        else "xpu" if (use_cuda and torch.xpu.is_available()) \
        else "cpu"
    if verbose:
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
        print(torch.version.cuda)
        print("Device in use:", device)

    if alg == "ppo":
        return _make_ppo(save_name, args, device)
    else:
        return _make_sac(save_name, args, device)


def _make_ppo(save_name: str,
              args: argparse.Namespace,
              device: str) -> tuple[PPO, int, dict, str]:
    """
    Build a PPO agent with parallel envs and the project's PPO hyperparameters.

    Hyperparameters used (chosen for PandaPush dense reward):
        learning_rate=5e-4, n_steps=4096, batch_size=512, n_epochs=8,
        clip_range=0.15, gamma=0.995, gae_lambda=0.95, ent_coef=0.005,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantage=True.

    Args:
        save_name: Path stem for `tensorboard_log`.
        args: Parsed CLI arguments. Reads `seed` and `no_vecnormalize`, and forwards
            the full namespace to `make_env` via `env_kwargs`.
        device: Torch device string ("cuda", "xpu", or "cpu").

    Returns:
        tuple: (model, n_envs, ppo_hparams, device).
    """
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
    }
    # number of parallel envs
    ppo_n_envs = 8

    # creation of multiple vectorized environments to parallelize training
    vec_env = make_vec_env(make_env,
                           n_envs=ppo_n_envs,
                           seed=args.seed,
                           env_kwargs={"args": args})

    # environment normalization if requested
    if not args.no_vecnormalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        vec_env.training = True
        vec_env.norm_reward = True

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        tensorboard_log=f"{save_name}/logs",
        seed=args.seed,
        **ppo_hparams)
    return model, ppo_n_envs, ppo_hparams, device


def _make_sac(save_name: str,
              args: argparse.Namespace,
              device: str) -> tuple[SAC, int, dict, str]:
    """
    Build a SAC agent with parallel envs and the project's SAC hyperparameters.

    Hyperparameters used:
        learning_rate=3e-4, buffer_size=1e6, batch_size=256, ent_coef="auto",
        gamma=0.98, tau=0.005, learning_starts=10000, train_freq=1, gradient_steps=4.

    Args:
        save_name: Path stem for `tensorboard_log`.
        args: Parsed CLI arguments. Reads `seed` and `no_vecnormalize`, and forwards
            the full namespace to `make_env` via `env_kwargs`.
        device: Torch device string ("cuda", "xpu", or "cpu").

    Returns:
        tuple: (model, n_envs, sac_hparams, device)
    """
    sac_hparams = {
        "learning_rate": 0.0003,
        "buffer_size": int(1e6),
        "batch_size": 256,
        "ent_coef": "auto",
        "gamma": 0.98,
        "tau": 0.005,
        "learning_starts": 10000,
        "train_freq": 1,
        "gradient_steps": 4,
    }
    # number of parallel envs
    sac_n_envs = 4

    # creation of multiple vectorized environments to parallelize training
    vec_env = make_vec_env(make_env,
                           n_envs=sac_n_envs,
                           seed=args.seed,
                           env_kwargs={"args": args})

    # environment normalization if requested
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
        seed=args.seed,
        **sac_hparams)
    return model, sac_n_envs, sac_hparams, device
