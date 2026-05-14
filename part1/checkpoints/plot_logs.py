"""
plot_logs.py: parse training-log .txt files produced by train.py and generate
comparison figures.

Log format (one line every 100 episodes):
    Ep  13100/50000 | avg:   822.0  min:   152.0  max:   871.1 | len:   967 |
    actor loss: 234.1552 | critic loss: 234.1552 | σ: [0.981, 0.963, 1.017] |
    best: 773.0 | elapsed: 00:25:53  (1405 steps/s)

All numeric fields are 100-episode rolling averages (except σ, which is the
current softplus-activated standard deviation snapshot).

CLI usage
---------
    python plot_logs.py "b=0:logs/b0.txt" "b=20:logs/b20.txt" "adaptive:logs/b-1.txt"
    python plot_logs.py "b=0:logs/b0.txt" "b=20:logs/b20.txt" --out figures/reinforce.pdf

Programmatic usage
------------------
    from plot_logs import parse_log, plot_comparison

    runs = [
        ("b=0",       "logs/reinforce_b0.txt"),
        ("b=20",      "logs/reinforce_b20.txt"),
        ("adaptive",  "logs/reinforce_badaptive.txt"),
    ]
    plot_comparison(runs, title="REINFORCE – baseline comparison", output_path="reinforce.pdf")
"""

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── regex that matches one log line ──────────────────────────────────────────
# Note: avg, min, max and actor/critic loss can all be negative, so we use
# -? in every numeric capture group.
_LINE_RE = re.compile(
    r"Ep\s+(\d+)/\d+"                          # episode number
    r".*?avg:\s*(-?[\d.]+)"                     # 100-ep avg reward
    r"\s+min:\s*(-?[\d.]+)"                     # 100-ep min
    r"\s+max:\s*(-?[\d.]+)"                     # 100-ep max
    r".*?\|\s*len:\s*([\d.]+)"                  # 100-ep avg length
    r".*?actor loss:\s*(-?[\d.]+)"              # 100-ep avg actor loss
    r".*?σ:\s*\[([\d.,\s]+)\]"                  # σ per joint
)


# ── parser ────────────────────────────────────────────────────────────────────

def parse_log(filepath):
    """Parse a training log file and return a dict of numpy arrays.

    Keys
    ----
    episodes   : episode index at which each log line was recorded
    avg        : 100-ep rolling average reward
    min, max   : 100-ep min / max reward
    lengths    : 100-ep average episode length (steps)
    actor_loss : 100-ep average actor loss
    sigma      : (N, n_joints) array of per-joint σ values
    sigma_mean : mean σ across all joints (convenience)
    """
    episodes, avg, min_r, max_r, lengths, actor_loss = [], [], [], [], [], []
    sigma_rows = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _LINE_RE.search(line)
            if m is None:
                continue
            episodes.append(int(m.group(1)))
            avg.append(float(m.group(2)))
            min_r.append(float(m.group(3)))
            max_r.append(float(m.group(4)))
            lengths.append(float(m.group(5)))
            actor_loss.append(float(m.group(6)))
            sigma_rows.append([float(v.strip()) for v in m.group(7).split(",")])

    if not episodes:
        raise ValueError(f"No data lines found in {filepath}. "
                         "Check that the file uses the expected log format.")

    sigma_arr = np.array(sigma_rows)   # shape (N, n_joints)

    return {
        "episodes":   np.array(episodes),
        "avg":        np.array(avg),
        "min":        np.array(min_r),
        "max":        np.array(max_r),
        "lengths":    np.array(lengths),
        "actor_loss": np.array(actor_loss),
        "sigma":      sigma_arr,
        "sigma_mean": sigma_arr.mean(axis=1),
    }


# ── plotter ───────────────────────────────────────────────────────────────────

def plot_comparison(runs, title="Training comparison", output_path=None, show=True):
    """Generate a 4-panel comparison figure from multiple training logs.

    Parameters
    ----------
    runs : list of (label: str, filepath: str) tuples
        Each tuple provides a display label and the path to a .txt log file.
    title : str
        Figure suptitle.
    output_path : str or None
        If given, save the figure to this path (format inferred from extension).
    show : bool
        If True (default), call plt.show() after drawing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    PANELS = [
        # (data_key, y-axis label, shade min-max?)
        ("avg",        "100-ep avg reward",          True),
        ("lengths",    "100-ep avg episode length",  True),
        ("actor_loss", "100-ep avg actor loss",      True),
        ("sigma_mean", "Mean σ across joints",       True),
    ]

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 14), sharex=False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parsed = {}
    for label, filepath in runs:
        try:
            parsed[label] = parse_log(filepath)
        except Exception as e:
            print(f"[WARNING] Could not parse '{filepath}': {e}")

    for ax_idx, (key, ylabel, shade) in enumerate(PANELS):
        ax = axes[ax_idx]

        for run_idx, (label, _) in enumerate(runs):
            if label not in parsed:
                continue
            data  = parsed[label]
            color = colors[run_idx % len(colors)]
            x     = data["episodes"]
            y     = data[key]

            ax.plot(x, y, color=color, lw=1.8, label=label, zorder=3)

            if shade and key == "avg":
                ax.fill_between(x, data["min"], data["max"],
                                alpha=0.10, color=color, zorder=2)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, _: f"{int(val):,}"
        ))

        if ax_idx == 0:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            ax.legend(fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Episode", fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()

    return fig


def plot_sigma_per_joint(runs, title="σ per joint", output_path=None, show=True):
    """Additional plot: individual σ curves for each joint, one panel per run.

    Useful to inspect whether specific joints have converged or are still exploring.
    """
    n_runs   = len(runs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(11, 3.5 * n_runs), sharex=False)
    if n_runs == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")
    joint_colors = ["tab:red", "tab:green", "tab:purple"]

    for ax, (label, filepath) in zip(axes, runs):
        try:
            data = parse_log(filepath)
        except Exception as e:
            ax.set_title(f"{label}  [parse error: {e}]")
            continue

        n_joints = data["sigma"].shape[1]
        for j in range(n_joints):
            ax.plot(data["episodes"], data["sigma"][:, j],
                    color=joint_colors[j % len(joint_colors)],
                    lw=1.4, label=f"σ joint {j}")

        ax.set_ylabel("σ (softplus)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, _: f"{int(val):,}"
        ))

    axes[-1].set_xlabel("Episode", fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()

    return fig


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse training logs and produce comparison figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Two runs, display interactively
  python plot_logs.py "b=0:logs/b0.txt" "b=20:logs/b20.txt"

  # Three runs, save to PDF
  python plot_logs.py "b=0:logs/b0.txt" "b=20:logs/b20.txt" "adaptive:logs/b-1.txt" \\
      --out figures/reinforce.pdf --title "REINFORCE baseline comparison"

  # Also save the per-joint σ figure
  python plot_logs.py "b=20:logs/b20.txt" --out_sigma figures/sigma.pdf
""",
    )
    parser.add_argument(
        "logs",
        nargs="+",
        metavar="LABEL:LOGFILE",
        help='One or more "label:filepath" pairs. '
             'If no colon is present, the stem of the filename is used as label.',
    )
    parser.add_argument("--title",     default="Training comparison",
                        help="Figure suptitle (default: 'Training comparison')")
    parser.add_argument("--out",       default=None,
                        help="Path to save the main comparison figure (e.g. reinforce.pdf)")
    parser.add_argument("--out_sigma", default=None,
                        help="Path to save the per-joint σ figure")
    parser.add_argument("--no_show",   action="store_true",
                        help="Do not call plt.show() (useful in headless environments)")

    args = parser.parse_args()

    run_list = []
    for item in args.logs:
        if ":" in item:
            lbl, pth = item.split(":", 1)
        else:
            lbl = Path(item).stem
            pth = item
        run_list.append((lbl, pth))

    show = not args.no_show

    plot_comparison(run_list, title=args.title, output_path=args.out, show=show)

    if args.out_sigma:
        plot_sigma_per_joint(run_list, output_path=args.out_sigma, show=show)
