#!/usr/bin/env python3
"""
Plot perplexity trends across models from JSON results in results/perplexity.

Produces a single plot comparing tag vs cat median perplexity per task with shaded bands.
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "/scratch/craj/langsense/results/perplexity"
DEFAULT_FIG_DIR = "/scratch/craj/langsense/figures/perplexity"

# Paper-friendly defaults
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


def model_prefix(name: str) -> str:
    if name.startswith("base"):
        return "base"
    return name.split("_")[0]


def prefix_colors() -> Dict[str, str]:
    # muted but distinct tones; cat/tag of same prefix share a color
    palette = [
        "#9ca3af",  # base
        "#6fb1d2",  # color
        "#f28e8c",  # emotion
        "#7dcfb6",  # gender
        "#c792df",  # kin
        "#f3b562",  # logic
        "#8fbf9f",  # num
        "#f2c6de",  # shape
        "#9bb7d4",  # space
        "#f7a072",  # temp
        "#a7c66c",  # valence
    ]
    order = ["base", "color", "emotion", "gender", "kin", "logic", "num", "shape", "space", "temp", "valence"]
    return {p: palette[i % len(palette)] for i, p in enumerate(order)}


def load_results(results_dir: str) -> List[Dict]:
    records: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {path}: {exc}")
            continue

        model = data.get("model") or os.path.basename(path).split("__")[0]
        data["model"] = model
        data["prefix"] = model_prefix(model)
        data["source"] = path
        records.append(data)

    return records


def plot_perplexity(records: List[Dict], out_path: str):
    if not records:
        print("No perplexity results found; nothing to plot.")
        return

    palette = prefix_colors()
    prefix_order = ["base", "color", "emotion", "gender", "kin", "logic", "num", "shape", "space", "temp", "valence"]
    order_index = {p: i for i, p in enumerate(prefix_order)}

    # gather tag/cat per prefix
    pair_medians: Dict[str, Dict[str, float]] = {}
    baseline_median = None
    for r in records:
        med = r.get("median")
        if med is None:
            continue
        if r["prefix"] == "base":
            baseline_median = med
            continue
        parts = r["model"].split("_")
        suffix = parts[1] if len(parts) > 1 else ""
        pair_medians.setdefault(r["prefix"], {})[suffix] = med

    # order prefixes and filter to ones with both tag and cat
    ordered_pairs = []
    for p in prefix_order:
        if p == "base":
            continue
        values = pair_medians.get(p, {})
        tag_val = values.get("tag")
        cat_val = values.get("cat")
        if tag_val is None or cat_val is None:
            continue
        ordered_pairs.append((p, tag_val, cat_val))

    if not ordered_pairs:
        print("No tag/cat pairs found to plot.")
        return

    fig, ax = plt.subplots(figsize=(12.5, 8))
    x_pos = list(range(len(ordered_pairs)))
    offset = 0.18

    for x, (prefix, tag_val, cat_val) in zip(x_pos, ordered_pairs):
        color = palette.get(prefix, "#8da0cb")
        x_tag = x - offset
        x_cat = x + offset
        # Shade only the enclosed area formed by the two lines and the connecting top segment
        ax.fill(
            [x_tag, x_tag, x_cat, x_cat],
            [0, tag_val, cat_val, 0],
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        # Tag and Cat vertical lines from x-axis up to their medians
        ax.vlines(x_tag, 0, tag_val, color=color, linewidth=2.6, alpha=0.95, linestyle="-")
        ax.vlines(x_cat, 0, cat_val, color=color, linewidth=2.6, alpha=0.9, linestyle="--")

    if baseline_median is not None:
        ax.axhline(baseline_median, color="#374151", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.text(
            len(x_pos) - 0.5,
            baseline_median * 1.01,
            f"Baseline: {baseline_median:,.1f}",
            ha="right",
            va="bottom",
            fontsize=22,
            color="#111827",
        )

    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_xlabel("")
    ax.set_title("")
    ax.grid(False)
    # ax.set_facecolor("#f9fafb")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{p.title()}Lang" for p, _, _ in ordered_pairs], rotation=25, ha="right")

    # Legend for line styles (tag vs cat)
    tag_handle = plt.Line2D([0], [0], color="#374151", linewidth=2.4, linestyle="-", label="Tag")
    cat_handle = plt.Line2D([0], [0], color="#374151", linewidth=2.4, linestyle="--", label="Cat")
    ax.legend(
        [tag_handle, cat_handle],
        ["Tag", "Cat"],
        loc="lower center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.5, -0.3),
    )

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot perplexity trends across models")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory containing perplexity JSON files")
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR, help="Directory to save plots")
    parser.add_argument("--filename", default="perplexity_trends.png", help="Output filename (within fig-dir)")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_results(args.results_dir)
    out_path = os.path.join(args.fig_dir, args.filename)
    plot_perplexity(records, out_path)


if __name__ == "__main__":
    main()
