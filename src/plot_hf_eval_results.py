#!/usr/bin/env python3
"""
Plot per-dataset HF evaluation summaries from *_summary.csv files.

Each summary file contains baseline/tag/categorical model scores for a single dataset.
This script creates one figure with subplots (one subplot per summary file) using the
same styling as plot_synthetic_results.py. The output PNG is saved to the figures
directory.
"""

import argparse
import glob
import math
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_DIR = "/scratch/craj/langsense/results/inference/hf_datasets_eval"
FIG_DIR = "/scratch/craj/langsense/figures/hf_datasets_eval"

# Match the plotting defaults from plot_synthetic_results.py
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pick_colors(models: List[str]) -> List[str]:
    color_base = "#B2BEB5"  # light gray baseline
    color_tag = "#8A9A5B"   # soft teal
    color_cat = "#f56991"   # soft crimson

    def color_for(model: str) -> str:
        if model.startswith("base_lm"):
            return color_base
        if "_tag" in model:
            return color_tag
        if "_cat" in model:
            return color_cat
        return "#b0bec5"

    return [color_for(m) for m in models]


def short_model_name(name: str) -> str:
    if name.startswith("base_lm"):
        return "baseline"
    if "_tag" in name:
        return "tag"
    if "_cat" in name:
        return "categorical"
    return name


def dataset_title_from_path(path: str) -> str:
    basename = os.path.basename(path).replace("_summary.csv", "")
    first_token = basename.split("_")[0] if basename else basename
    return f"{first_token.capitalize()}Lang" if first_token else "Lang"


def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    grouped = df.groupby("model", as_index=False)["accuracy"].mean()
    return grouped


def plot_hf_eval_grid(summary_dir: str, fig_dir: str, out_name: str = "hf_eval_grid.png"):
    files = sorted(glob.glob(os.path.join(summary_dir, "*summary.csv")))
    files = [f for f in files if "mmlu" not in os.path.basename(f)]
    if not files:
        print(f"No summary.csv files found in {summary_dir}")
        return

    ensure_dir(fig_dir)
    n = len(files)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 5.6))
    axes = axes.flatten()

    for ax, path in zip(axes, files):
        df = load_and_aggregate(path)
        if df.empty:
            ax.axis("off")
            continue
        # order models: baseline, tag, categorical
        order_map = {"baseline": 0, "tag": 1, "categorical": 2}
        records = sorted(
            df.to_dict("records"),
            key=lambda r: order_map.get(short_model_name(r["model"]), 99),
        )
        models = [r["model"] for r in records]
        heights = [r["accuracy"] for r in records]
        colors = pick_colors(models)
        x = range(len(models))
        bars = ax.bar(
            x,
            heights,
            width=0.5,
            color=colors,
            edgecolor="#222222",
            linewidth=1.4,
        )

        for rect, h in zip(bars, heights):
            text_y = h + 0.08
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                text_y,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=20,
                color="#222222",
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels([short_model_name(m) for m in models], rotation=0, fontsize=22)
        ax.set_ylim(0, 1.0)
        ax.set_title(dataset_title_from_path(path))
        ax.grid(False)

        # legend_handles = [
        #     plt.Rectangle((0, 0), 1, 1, color="#B2BEB5"),
        #     plt.Rectangle((0, 0), 1, 1, color="#8A9A5B"),
        #     plt.Rectangle((0, 0), 1, 1, color="#f56991"),
        # ]
        # ax.legend(legend_handles, ["baseline", "tag", "categorical"], loc="upper right", frameon=False)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#B2BEB5"),
            plt.Rectangle((0, 0), 1, 1, color="#8A9A5B"),
            plt.Rectangle((0, 0), 1, 1, color="#f56991"),
        ]

        fig.legend(
            legend_handles,
            ["baseline", "tag", "categorical"],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )


    for ax in axes[len(files) :]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(fig_dir, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot HF dataset evaluation summaries")
    parser.add_argument(
        "--summary-dir",
        default=SUMMARY_DIR,
        help="Directory containing *_summary.csv files",
    )
    parser.add_argument(
        "--fig-dir",
        default=FIG_DIR,
        help="Directory to save the output figure",
    )
    parser.add_argument(
        "--out-name",
        default="hf_eval_grid.png",
        help="Filename for the output figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_hf_eval_grid(args.summary_dir, args.fig_dir, args.out_name)


if __name__ == "__main__":
    main()
