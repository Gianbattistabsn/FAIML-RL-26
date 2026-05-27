import argparse


def parse_args_train() -> argparse.Namespace:
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
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Disable VecNormalize during training (observation normalization)",
    )
    parser.add_argument(
        "--mass-range",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("MIN", "MAX"),
        help="Mass range for domain randomization (min max)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=51,
        help="Random seed for the randomization wrapper",
    )
    parser.add_argument(
        "--verbose-wrapper",
        action="store_true",
        help="Enable verbose output from the randomization wrapper",
    )
    parser.add_argument(
        "--adr-delta",
        type=float,
        default=0.2,
        help="ADR: range increase/decrease size",
    )
    parser.add_argument(
        "--adr-buffer-size",
        type=int,
        default=20,
        help="ADR: performance buffer size for boundary evaluation",
    )
    parser.add_argument(
        "--adr-perf-low",
        type=float,
        default=-25.0,
        help="ADR: mean return below this threshold shrinks the range",
    )
    parser.add_argument(
        "--adr-perf-high",
        type=float,
        default=-10.0,
        help="ADR: mean return above this threshold expands the range",
    )
    parser.add_argument(
        "--adr-boundary-prob",
        type=float,
        default=0.8,
        help="ADR: probability of sampling at the boundary (must be in [0, 1])",
    )
    return parser.parse_args()


def parse_args_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO model zip file in format part2/models/{alg}_push_{sampling_strategy}_{env_type}_{args.timesteps // 1000}k)\n (e.g. part2/models/ppo_push_none_source_5k)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of eval episodes"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Disable VecNormalize (skip loading the .pkl stats file)",
    )
    return parser.parse_args()
