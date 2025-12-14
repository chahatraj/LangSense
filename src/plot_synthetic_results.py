#!/usr/bin/env python3
"""
Plot synthetic evaluation trends from all_synthetic_summary.csv.

Outputs PNGs to the figures/synthetic directory:
- grouped_overall.png : per-dataset accuracy by model (aggregated seed=ALL, map=OVERALL) with CI error bars.
- model_mean.png      : mean accuracy per model across datasets (aggregated seed=ALL, map=OVERALL).
- per_dataset_grid.png: one figure with subplots (one per dataset) comparing base/tag/cat models with soft pastel colors, value labels, and legend.
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SUMMARY = "/scratch/craj/langsense/results/inference/synthetic_eval/all_synthetic_summary.csv"
DEFAULT_FIG_DIR = "/scratch/craj/langsense/figures/synthetic"

# Improve default readability for paper-style plots
plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
    }
)


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize seed to string for consistent filtering
    df["seed"] = df["seed"].astype(str)
    return df


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pick_colors(n: int) -> List[str]:
    # Soft pastel palette
    pastel = [
        "#aec6cf",
        "#ffb347",
        "#cdb5cd",
        "#b2dfdb",
        "#f7cac9",
        "#c6e2ff",
        "#ffe0b2",
        "#d5e8d4",
        "#e6e6fa",
        "#fddde6",
    ]
    if n <= len(pastel):
        return pastel[:n]
    cmap = plt.get_cmap("Pastel1")
    return [cmap(i % cmap.N) for i in range(n)]


def short_model_name(name: str) -> str:
    if name.startswith("base_lm"):
        return "baseline"
    if "_tag" in name:
        return "tag"
    if "_cat" in name:
        return "categorical"
    return name


def plot_grouped_overall(df: pd.DataFrame, out_path: str):
    data = df[(df["seed"] == "ALL") & (df["map"] == "OVERALL")]
    if data.empty:
        print("No aggregated OVERALL rows found; skipping grouped_overall plot.")
        return

    datasets = sorted(data["dataset"].unique())
    models = sorted(data["model"].unique())
    x = range(len(datasets))
    width = 0.8 / max(len(models), 1)
    colors = pick_colors(len(models))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        subset = data[data["model"] == model].set_index("dataset")
        heights = [subset.loc[d]["accuracy"] if d in subset.index else 0 for d in datasets]
        ci_lows = [subset.loc[d]["ci_low"] if d in subset.index else None for d in datasets]
        ci_highs = [subset.loc[d]["ci_high"] if d in subset.index else None for d in datasets]
        err_low = []
        err_high = []
        for lo, hi, h in zip(ci_lows, ci_highs, heights):
            if lo is None or hi is None:
                err_low.append(0)
                err_high.append(0)
            else:
                err_low.append(h - lo)
                err_high.append(hi - h)
        offsets = [p + (i - (len(models) - 1) / 2) * width for p in x]
        ax.bar(offsets, heights, width=width, label=model, color=colors[i], yerr=[err_low, err_high], capsize=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Synthetic Eval Accuracy by Dataset (seed=ALL, map=OVERALL)")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_model_mean(df: pd.DataFrame, out_path: str):
    data = df[(df["seed"] == "ALL") & (df["map"] == "OVERALL")]
    if data.empty:
        print("No aggregated OVERALL rows found; skipping model_mean plot.")
        return

    grouped = data.groupby("model")["accuracy"].mean().sort_values(ascending=False)
    models = grouped.index.tolist()
    heights = grouped.values
    colors = pick_colors(len(models))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, heights, color=colors)
    ax.set_ylabel("Mean Accuracy Across Datasets")
    ax.set_ylim(0, 1.0)
    ax.set_title("Synthetic Eval Mean Accuracy per Model (seed=ALL, map=OVERALL)")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_per_dataset_grid(df: pd.DataFrame, fig_dir: str):
    data = df[(df["seed"] == "ALL") & (df["map"] == "OVERALL")]
    if data.empty:
        print("No aggregated OVERALL rows found; skipping per-dataset grid plot.")
        return

    desired_order = ["color", "gender", "logic", "shape", "emotion", "num", "kin", "space", "temp", "valence"]
    datasets = [d for d in desired_order if d in set(data["dataset"])]
    # fixed 2x5 grid for up to 10 datasets
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 5.6))
    axes = axes.flatten()

    color_base = "#B2BEB5"  # light gray baseline
    color_tag = "#8A9A5B"   # soft teal
    color_cat = "#f56991"   # soft crimson

    for ax, ds in zip(axes, desired_order):
        if ds not in datasets:
            ax.axis("off")
            continue
        subset = data[data["dataset"] == ds]
        # order: baseline, tag, categorical
        order_map = {"baseline": 0, "tag": 1, "categorical": 2}
        subset = sorted(
            subset.to_dict("records"),
            key=lambda r: order_map.get(short_model_name(r["model"]), 99),
        )
        if not subset:
            continue
        models = [r["model"] for r in subset]
        heights = [r["accuracy"] for r in subset]
        ci_low = [r.get("ci_low") for r in subset]
        ci_high = [r.get("ci_high") for r in subset]

        err_low = []
        err_high = []
        for h, lo, hi in zip(heights, ci_low, ci_high):
            if lo is None or hi is None:
                err_low.append(0)
                err_high.append(0)
            else:
                err_low.append(h - lo)
                err_high.append(hi - h)

        def color_for(m):
            if m.startswith("base_lm"):
                return color_base
            if "_tag" in m:
                return color_tag
            if "_cat" in m:
                return color_cat
            return "#b0bec5"

        bar_colors = [color_for(m) for m in models]
        x = range(len(models))
        bars = ax.bar(
            x,
            heights,
            width=0.5,
            color=bar_colors,
            yerr=[err_low, err_high],
            capsize=6,
            edgecolor="#222222",
            linewidth=1.4,
        )

        # value labels above bars
        for rect, m, h in zip(bars, models, heights):
            text_y = h + 0.10
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                text_y,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=22,
                # fontweight="bold",
                color="#222222",
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels([short_model_name(m) for m in models], rotation=0, fontsize=22)
        ax.set_ylim(0, 0.8)
        # Title with Lang suffix for consistency with HF plots
        ax.set_title(f"{ds.capitalize()}Lang")
        ax.grid(False)

        # per-subplot legend
        # handles = [
        #     plt.Rectangle((0, 0), 1, 1, color=color_base),
        #     plt.Rectangle((0, 0), 1, 1, color=color_tag),
        #     plt.Rectangle((0, 0), 1, 1, color=color_cat),
        # ]
        # ax.legend(handles, ["baseline", "tag", "categorical"], loc="upper right", frameon=False)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#B2BEB5"),
            plt.Rectangle((0, 0), 1, 1, color="#8A9A5B"),
            plt.Rectangle((0, 0), 1, 1, color="#f56991"),
        ]

        fig.legend(
            handles,
            ["baseline", "tag", "categorical"],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )

    # hide unused axes
    for ax in axes[len(datasets) :]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(fig_dir, "per_dataset_grid.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot synthetic evaluation trends")
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY,
        help="Path to all_synthetic_summary.csv",
    )
    parser.add_argument(
        "--fig-dir",
        default=DEFAULT_FIG_DIR,
        help="Directory to save plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.fig_dir)
    df = load_summary(args.summary)

    plot_grouped_overall(df, os.path.join(args.fig_dir, "grouped_overall.png"))
    plot_model_mean(df, os.path.join(args.fig_dir, "model_mean.png"))
    plot_per_dataset_grid(df, args.fig_dir)


if __name__ == "__main__":
    main()
