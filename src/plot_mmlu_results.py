#!/usr/bin/env python3
"""
Plot MMLU inference results from mmlu_summary.csv.

Produces:
- overall.png : OVERALL accuracy per model.
- subsets.png : Selected MMLU subjects for a given set of models (default: baseline, color_tag, color_cat).
"""

import argparse
import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = "/scratch/craj/langsense/results/inference/hf_datasets_eval/mmlu_summary.csv"
DEFAULT_FIG_DIR = "/scratch/craj/langsense/figures/mmlu"

# readable defaults
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    }
)

DEFAULT_SUBJECTS = [
    "moral_scenarios",
    "elementary_mathematics",
    "professional_law",
    "human_sexuality",
    "high_school_statistics",
    "global_facts",
    "human_aging",
    "logical_fallacies",
    "formal_logic",
    "high_school_chemistry",
    "high_school_physics",
]

DEFAULT_MODELS = []


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def short_name(model: str) -> str:
    # compress common prefixes to cleaner labels
    if model.startswith("base_lm"):
        return "Baseline"
    if "color_tag" in model:
        return "Color Tag"
    if "color_cat" in model:
        return "Color Cat"
    return model.replace("_", " ").title()


def subject_label(subject: str) -> str:
    custom = {
        "moral_scenarios": "Moral Scen.",
        "elementary_mathematics": "Elem. Math.",
        "professional_law": "Proff. Law",
        "human_sexuality": "Sexuality",
        "high_school_statistics": "HS Stats",
        "global_facts": "Global Facts",
        "human_aging": "Human Age",
        "logical_fallacies": "Log. Fallacies",
        "formal_logic": "Formal Logic",
        "high_school_chemistry": "HS Chemistry",
        "high_school_physics": "HS Physics",
    }
    return custom.get(subject, subject.replace("_", " ").title())


def filter_models(df: pd.DataFrame, patterns: Iterable[str]) -> pd.DataFrame:
    pats = list(patterns)
    if not pats:
        return df
    mask = False
    for p in pats:
        mask |= df["model"].str.contains(p)
    return df[mask]


def plot_overall(df: pd.DataFrame, out_path: str):
    data = df[df["subject"] == "OVERALL"]
    if data.empty:
        print("No OVERALL rows found; skipping overall plot.")
        return
    grouped = data.groupby("model")["accuracy"].mean().sort_values(ascending=False)
    models = grouped.index.tolist()
    accs = grouped.values
    colors = []
    for m in models:
        if m.startswith("base_lm"):
            colors.append("#B2BEB5")
        elif "_tag" in m:
            colors.append("#8A9A5B")
        elif "_cat" in m:
            colors.append("#f56991")
        else:
            colors.append("#B2BEB5")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(range(len(models)), accs, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title("")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short_name(m) for m in models], rotation=30, ha="right")
    # ax.grid(False)
    ax.grid(axis="x", linestyle="--", linewidth=1.0, alpha=0.4)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_subsets(df: pd.DataFrame, subjects: List[str], model_patterns: List[str], out_path: str, title: str = "MMLU Subset Accuracy"):
    data = df[df["subject"].isin(subjects)]
    if data.empty:
        print("No matching subject rows for subset plot; skipping.")
        return
    if model_patterns:
        data = filter_models(data, model_patterns)
        if data.empty:
            print("No matching models after filtering; skipping subset plot.")
            return

    # use mean across seeds if multiple
    data = data.groupby(["model", "subject"])["accuracy"].mean().reset_index()

    models = sorted(data["model"].unique())
    color_map = {}
    for m in models:
        if m.startswith("base_lm"):
            color_map[m] = "#B2BEB5"
        elif "_tag" in m:
            color_map[m] = "#8A9A5B"
        elif "_cat" in m:
            color_map[m] = "#f56991"
        else:
            color_map[m] = "#B2BEB5"
    x = range(len(subjects))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for m in models:
        subset = data[data["model"] == m].set_index("subject")
        ys = [subset.loc[s]["accuracy"] if s in subset.index else float("nan") for s in subjects]
        marker = "o" if m.startswith("base_lm") else ("s" if "_cat" in m else "^")
        ax.plot(
            x,
            ys,
            marker=marker,
            linestyle="-",
            linewidth=2.2,
            markersize=7,
            color=color_map[m],
            label="_nolegend_",  # legend handled separately below
        )
        # shaded area under the line
        ax.fill_between(x, ys, [0] * len(ys), color=color_map[m], alpha=0.18)

    ax.set_xticks(list(x))
    ax.set_xticklabels([subject_label(s) for s in subjects], rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title(title)
    # ax.grid(False)
    ax.grid(axis="x", linestyle="--", linewidth=1.0, alpha=0.4)
    # Legend: Baseline plus per-prefix Tag/Cat entries with clean names
    from matplotlib.lines import Line2D

    handles = []
    added = set()
    if any(m.startswith("base_lm") for m in models):
        handles.append(
            Line2D([0], [0], color="#B2BEB5", marker="o", linestyle="-", linewidth=2.2, markersize=7, label="Baseline")
        )
    for m in models:
        if m.startswith("base_lm"):
            continue
        label_prefix = m.split("_tag")[0] if "_tag" in m else m.split("_cat")[0]
        label_prefix = label_prefix.replace("_", " ").title()
        if "_tag" in m and (label_prefix, "tag") not in added:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#8A9A5B",
                    marker="^",
                    linestyle="-",
                    linewidth=2.2,
                    markersize=7,
                    label=f"{label_prefix}Lang Tag",
                )
            )
            added.add((label_prefix, "tag"))
        if "_cat" in m and (label_prefix, "cat") not in added:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#f56991",
                    marker="s",
                    linestyle="-",
                    linewidth=2.2,
                    markersize=7,
                    label=f"{label_prefix}Lang Cat",
                )
            )
            added.add((label_prefix, "cat"))
    if handles:
        ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_subsets_for_prefix(
    df: pd.DataFrame,
    subjects: List[str],
    prefix: str,
    fig_dir: str,
):
    patterns = [f"{prefix}_tag", f"{prefix}_cat", "base_lm"]
    title = ""
    out_path = os.path.join(fig_dir, f"mmlu_subsets_{prefix}.png")
    plot_subsets(df, subjects, patterns, out_path, title=title)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MMLU inference results")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to mmlu_summary.csv")
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR, help="Output directory for figures")
    parser.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS, help="Subjects to plot for subset figure")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model name patterns to include for subset figure")
    parser.add_argument("--split", default="test", help="Split to filter (default: test)")
    parser.add_argument("--config", default="all", help="Config to filter (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df[(df["split"] == args.split) & (df["config"] == args.config)]

    ensure_dir(args.fig_dir)
    plot_overall(df, os.path.join(args.fig_dir, "mmlu_overall.png"))
    # If model patterns provided, do one combined plot; otherwise per prefix pair (tag/cat) with baseline
    if args.models:
        plot_subsets(df, args.subjects, args.models, os.path.join(args.fig_dir, "mmlu_subsets.png"))
    else:
        # find prefixes that have both tag and cat
        models = df["model"].unique()
        tag_prefixes = {m.split("_tag")[0] for m in models if "_tag" in m}
        cat_prefixes = {m.split("_cat")[0] for m in models if "_cat" in m}
        prefixes = sorted(tag_prefixes & cat_prefixes)
        if not prefixes:
            print("No tag/cat prefix pairs found; skipping per-prefix plots.")
        for p in prefixes:
            plot_subsets_for_prefix(df, args.subjects, p, args.fig_dir)


if __name__ == "__main__":
    main()
