import argparse
import os
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC


def load_model(
        args: argparse.Namespace,
        save_name: str,
        alg: str) -> PPO | SAC:
    """
    Load a previously trained PPO or SAC model with its matching VecNormalize stats.

    Builds a single (non-randomized) `PandaPush-v3` env with dense reward and the
    requested `env_type`, then attempts to restore the `VecNormalize` statistics
    saved alongside the model at training time. If the `.pkl` stats file exists,
    the env is wrapped and switched to inference mode (`training=False`,
    `norm_reward=False`) so the running moments are frozen and rewards are
    returned unnormalized. If the file is missing, a warning is printed and the
    env is returned unwrapped — the loaded policy will then see un-normalized
    observations, which typically degrades performance significantly.

    The wrapped env is attached to the loaded model so downstream calls
    (`model.predict`, `model.learn`, etc.) work without further setup.

    Args:
        args: Parsed CLI arguments. Reads `env_type` to build the eval env.
        save_name: Path stem of the saved model. The function looks for
            `{save_name}.zip` (model weights) and `{save_name}_vecnormalize.pkl`
            (normalization stats).
        alg: Algorithm identifier; "ppo" loads a `PPO`, anything else loads a `SAC`.

    Returns:
        PPO | SAC: The loaded SB3 model with its env attached.

    Note:
        The returned env is *not* wrapped in `RandomizationWrapper`. Evaluation
        therefore runs on the deterministic default mass.
    """
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

    if alg == "ppo":
        return PPO.load(f"{save_name}.zip", env=load_vec_env)
    else:
        return SAC.load(f"{save_name}.zip", env=load_vec_env)