import argparse
import gymnasium as gym
from rand_wrapper import RandomizationWrapper


def make_env(args: argparse.Namespace) -> gym.Env:
    """
    Build a single PandaPush-v3 environment wrapped with domain randomization.

    Factory function intended to be passed to `stable_baselines3.common.env_util.make_vec_env`,
    which calls it once per parallel worker. The returned environment is a `PandaPush-v3`
    instance with a dense reward, configured for the requested `env_type` ("source" or
    "target"), and wrapped in a `RandomizationWrapper` that controls object-mass
    randomization according to the configured `sampling_strategy` ("none", "udr", or "adr").


    :param args: Parsed CLI arguments.

    Returns:
        gym.Env: A wrapped `PandaPush-v3` environment ready to be vectorized.
    """
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