#!/usr/bin/env python3
"""
Evaluate models on MMLU (cais/mmlu) multiple-choice questions.

- Supports the "all" config (default) or any subject-specific config.
- Scores each option by conditional log-likelihood given the question.
- Computes accuracy per subject and overall.
- Shuffles with a fixed seed and evaluates up to --max-samples.
- Writes per-sample CSV and per-model summary (overall + per-subject) CSV.
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def score_option(model, tokenizer, question: str, option: str) -> float:
    """Return total log-likelihood of `option` conditioned on `question`."""
    prompt_ids = tokenizer(
        question.strip() + "\n", return_tensors="pt", add_special_tokens=False
    ).input_ids
    option_ids = tokenizer(
        option.strip(), return_tensors="pt", add_special_tokens=False
    ).input_ids

    input_ids = torch.cat([prompt_ids, option_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    option_len = option_ids.shape[1]
    return -outputs.loss.item() * option_len


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, int, List[float], str, List[str], int, str]:
    question = sample["question"]
    options = sample["choices"]
    label_idx = int(sample["answer"])
    subject = sample.get("subject", "")

    scores = [score_option(model, tokenizer, question, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx == label_idx)
    return correct, pred_idx, scores, question, options, label_idx, subject


def prepare_dataset(
    split: str,
    max_samples: int,
    seed: int,
    config: str,
):
    ds = load_dataset("cais/mmlu", config, split=split)
    print(f"Loaded MMLU ({config}) split '{split}' with {len(ds)} samples")

    if len(ds) == 0:
        raise ValueError("No samples available in the selected split/config.")

    ds = ds.shuffle(seed=seed)
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    print(f"Prepared {len(ds)} samples (seed={seed})")
    return ds


def evaluate_model_on_dataset(model_path: str, dataset, device: str, desc: str):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 0
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in tqdm(dataset, desc=desc):
        correct, pred_idx, scores, question, options, label_idx, subject = evaluate_sample(
            model, tokenizer, sample
        )

        max_options = max(max_options, len(options))
        evaluated += 1
        if correct == 1:
            correct_count += 1
            per_subject[subject]["correct"] += 1
        else:
            per_subject[subject]["correct"] += 0
        per_subject[subject]["total"] += 1

        row = {
            "question": question,
            "options": " ||| ".join(options),
            "label": label_idx,
            "pred": pred_idx,
            "correct": correct,
            "subject": subject,
        }
        for i, score in enumerate(scores):
            row[f"score_{i}"] = score
        rows.append(row)

    if evaluated == 0:
        raise ValueError(f"No valid samples evaluated for {model_path}.")

    accuracy = correct_count / evaluated if evaluated else 0.0
    print(f"{model_path} accuracy: {accuracy:.4f} over {evaluated} samples")
    return accuracy, rows, max_options, evaluated, per_subject


def write_csv(rows, output_path: str, option_count: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_fields = ["model", "question", "options", "label", "pred", "correct", "subject"]
    score_fields = [f"score_{i}" for i in range(option_count)]
    fieldnames = base_fields + score_fields

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to {output_path}")


def write_summary(summary_rows, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["model", "subject", "samples", "accuracy", "seed", "split", "config"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MMLU MCQ evaluation")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=None,
        help="List of model paths to evaluate (default: all folders in models/merged)",
    )
    parser.add_argument("--config", default="all", help="MMLU config/subject (default: all)")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate (after shuffling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for shuffling/selecting samples",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (e.g., 'cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--output",
        default="/scratch/craj/langsense/results/inference/mmlu_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/mmlu_summary.csv",
        help="Where to write the per-model accuracy summary CSV (overall + per-subject)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_paths = args.model_paths
    if model_paths is None:
        merged_root = "/scratch/craj/langsense/models/merged"
        try:
            model_paths = sorted(
                [
                    os.path.join(merged_root, d)
                    for d in os.listdir(merged_root)
                    if os.path.isdir(os.path.join(merged_root, d))
                ]
            )
            if not model_paths:
                raise ValueError("No model directories found in /scratch/craj/langsense/models/merged")
        except FileNotFoundError as e:
            raise ValueError(f"Could not discover models: {e}")

    dataset = prepare_dataset(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        config=args.config,
    )

    all_rows = []
    summary_rows = []
    max_option_count = 0

    for model_path in model_paths:
        accuracy, rows, model_max_opts, evaluated, per_subject = evaluate_model_on_dataset(
            model_path=model_path,
            dataset=dataset,
            device=args.device,
            desc=f"Evaluating MMLU ({os.path.basename(model_path)})",
        )
        max_option_count = max(max_option_count, model_max_opts)
        for row in rows:
            row["model"] = os.path.basename(model_path.rstrip("/"))
            all_rows.append(row)

        # overall row
        summary_rows.append(
            {
                "model": os.path.basename(model_path.rstrip("/")),
                "subject": "OVERALL",
                "samples": evaluated,
                "accuracy": accuracy,
                "seed": args.seed,
                "split": args.split,
                "config": args.config,
            }
        )
        # per-subject rows
        for subject, stats in per_subject.items():
            subj_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
            summary_rows.append(
                {
                    "model": os.path.basename(model_path.rstrip("/")),
                    "subject": subject,
                    "samples": stats["total"],
                    "accuracy": subj_acc,
                    "seed": args.seed,
                    "split": args.split,
                    "config": args.config,
                }
            )
        print(f"Finished {model_path} with accuracy {accuracy:.4f}")

    for row in all_rows:
        for i in range(max_option_count):
            row.setdefault(f"score_{i}", "")

    write_csv(all_rows, args.output, option_count=max_option_count)
    write_summary(summary_rows, args.summary_output)


if __name__ == "__main__":
    main()
