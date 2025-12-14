#!/usr/bin/env python3
"""
Evaluate space models on PIQA (MCQ) with optional spatial filtering.

- Uses baber/piqa from Hugging Face.
- Scores sol1 vs. sol2 by conditional log-likelihood given the goal text.
- Shuffles with a fixed seed and evaluates up to `--max-samples` (default 1000).
- Optionally restricts to samples mentioning spatial terms (left/right/up/down/etc.).
"""

import argparse
import os
import random
import csv
from typing import Dict, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SPATIAL_KEYWORDS = {
    "left",
    "right",
    "up",
    "above",
    "over",
    "down",
    "below",
    "under",
}


def contains_spatial(text: str) -> bool:
    """Return True if any spatial keyword appears as a standalone word."""
    lowered = text.lower()
    return any(f" {kw} " in f" {lowered} " for kw in SPATIAL_KEYWORDS)


def filter_dataset(dataset, use_spatial_filter: bool):
    if not use_spatial_filter:
        return dataset, len(dataset)

    filtered = dataset.filter(
        lambda ex: (
            contains_spatial(ex["goal"])
            or contains_spatial(ex["sol1"])
            or contains_spatial(ex["sol2"])
        )
    )
    return filtered, len(filtered)


def load_model(model_path: str, device: str):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True,)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def score_option(
    model,
    tokenizer,
    goal: str,
    option: str,
) -> float:
    """
    Return total log-likelihood of `option` conditioned on `goal`.
    Higher is better.
    """
    prompt_ids = tokenizer(
        goal.strip() + "\n", return_tensors="pt", add_special_tokens=False
    ).input_ids
    option_ids = tokenizer(
        option.strip(), return_tensors="pt", add_special_tokens=False
    ).input_ids

    input_ids = torch.cat([prompt_ids, option_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # ignore prompt tokens

    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    option_len = option_ids.shape[1]
    return -outputs.loss.item() * option_len


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, float, float]:
    goal = sample["goal"]
    sol1, sol2 = sample["sol1"], sample["sol2"]
    label = int(sample.get("label", -1))

    score1 = score_option(model, tokenizer, goal, sol1)
    score2 = score_option(model, tokenizer, goal, sol2)
    pred = 0 if score1 >= score2 else 1
    correct = int(pred == label) if label in (0, 1) else -1
    return correct, score1, score2


def prepare_dataset(split: str, max_samples: int, seed: int, use_spatial_filter: bool):
    raw_ds = load_dataset("baber/piqa", split=split)
    print(f"Loaded PIQA split '{split}' with {len(raw_ds)} samples")

    ds, filtered_len = filter_dataset(raw_ds, use_spatial_filter)
    if use_spatial_filter:
        print(f"After spatial filter: {filtered_len} samples")

    if len(ds) == 0:
        raise ValueError("No samples left after filtering; relax the filter or keywords.")

    ds = ds.shuffle(seed=seed)
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    print(f"Prepared {len(ds)} samples (seed={seed})")
    return ds


def evaluate_model_on_dataset(model_path: str, dataset, device: str, desc: str):
    model, tokenizer = load_model(model_path, device)

    results = []
    correct_count = 0

    for sample in tqdm(dataset, desc=desc):
        correct, score1, score2 = evaluate_sample(model, tokenizer, sample)
        results.append(
            {
                "goal": sample["goal"],
                "sol1": sample["sol1"],
                "sol2": sample["sol2"],
                "label": int(sample.get("label", -1)),
                "score1": score1,
                "score2": score2,
                "pred": 0 if score1 >= score2 else 1,
                "correct": correct,
            }
        )
        if correct == 1:
            correct_count += 1

    accuracy = correct_count / len(dataset)
    print(f"{model_path} accuracy: {accuracy:.4f} over {len(dataset)} samples")
    return accuracy, results


def write_csv(rows, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "model",
        "goal",
        "sol1",
        "sol2",
        "label",
        "score1",
        "score2",
        "pred",
        "correct",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to {output_path}")


def write_summary(summary_rows, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["model", "samples", "accuracy", "seed", "spatial_filter", "split"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Space PIQA MCQ evaluation")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/space_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/space_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, space_tag, space_cat)",
    )
    parser.add_argument("--split", default="train", help="PIQA split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate (after filtering)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for shuffling/selecting samples",
    )
    parser.add_argument(
        "--no-spatial-filter",
        action="store_true",
        help="Disable filtering for spatial keywords",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (e.g., 'cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--output",
        default="/scratch/craj/langsense/results/inference/space_piqa_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/space_piqa_summary.csv",
        help="Where to write the per-model accuracy summary CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_spatial_filter = not args.no_spatial_filter

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = prepare_dataset(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        use_spatial_filter=use_spatial_filter,
    )

    all_rows = []
    summary_rows = []
    for model_path in args.model_paths:
        accuracy, results = evaluate_model_on_dataset(
            model_path=model_path,
            dataset=dataset,
            device=args.device,
            desc=f"Evaluating PIQA ({os.path.basename(model_path)})",
        )
        for sample in results:
            row = sample.copy()
            row["model"] = os.path.basename(model_path.rstrip("/"))
            all_rows.append(row)
        print(f"Finished {model_path} with accuracy {accuracy:.4f}")
        summary_rows.append(
            {
                "model": os.path.basename(model_path.rstrip("/")),
                "samples": len(dataset),
                "accuracy": accuracy,
                "seed": args.seed,
                "spatial_filter": use_spatial_filter,
                "split": args.split,
            }
        )

    write_csv(all_rows, args.output)
    write_summary(summary_rows, args.summary_output)


if __name__ == "__main__":
    main()
