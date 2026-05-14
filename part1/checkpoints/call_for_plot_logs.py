# main.py
from plot_logs import plot_comparison, plot_sigma_per_joint

runs = [
    ("baseline_0",  "logs/baseline_0.txt"),
    ("baseline_20", "logs/baseline_20.txt"),
    ("adaptive",    "logs/baseline_-1.txt"),
]

plot_comparison(
    runs,
    title="REINFORCE – Baseline comparison",
    output_path="figures/comparison.pdf",
    show=True,
)

plot_sigma_per_joint(
    runs,
    title="σ per joint – Baseline comparison",
    output_path="figures/sigma.pdf",
    show=True,
)