import argparse


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
        default=42,
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
