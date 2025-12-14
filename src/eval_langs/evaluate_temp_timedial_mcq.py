#!/usr/bin/env python3
"""
Evaluate temporal reasoning models on TimeDial by framing <MASK> completion as MCQ.

- Uses google-research-datasets/time_dial (trust_remote_code=True).
- Options: two correct completions + two incorrect distractors.
- Shuffles options per sample with a fixed seed; marks prediction correct if it matches any correct option.
- Shuffles dataset with seed and evaluates up to --max-samples.
- Writes per-sample CSV and per-model accuracy summary.
"""

import argparse
import csv
import os
import random
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


def build_prompt(conversation: List[str]) -> str:
    convo = "\n".join(conversation)
    return f"{convo}\n\nQuestion: Choose the best option to replace <MASK>.\n"


def score_option(model, tokenizer, prompt: str, option: str) -> float:
    """Return total log-likelihood of `option` conditioned on `prompt`."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    option_ids = tokenizer(option.strip(), return_tensors="pt", add_special_tokens=False).input_ids

    input_ids = torch.cat([prompt_ids, option_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    option_len = option_ids.shape[1]
    return -outputs.loss.item() * option_len


def assemble_options(sample: Dict, rng: random.Random) -> Tuple[List[str], List[int]]:
    candidates = [
        ("correct", str(sample.get("correct1", "")).strip()),
        ("correct", str(sample.get("correct2", "")).strip()),
        ("incorrect", str(sample.get("incorrect1", "")).strip()),
        ("incorrect", str(sample.get("incorrect2", "")).strip()),
    ]
    options = []
    correct_indices = []
    for tag, text in candidates:
        if text:
            idx = len(options)
            options.append(text)
            if tag == "correct":
                correct_indices.append(idx)

    if len(options) < 2 or not correct_indices:
        raise ValueError("Sample missing sufficient options.")

    # Shuffle options while tracking correct indices
    paired = list(zip(options, [i in correct_indices for i in range(len(options))]))
    rng.shuffle(paired)
    options = [opt for opt, _ in paired]
    correct_indices = [i for i, (_, is_correct) in enumerate(paired) if is_correct]
    return options, correct_indices


def prepare_dataset(split: str, max_samples: int, seed: int):
    ds = load_dataset("google-research-datasets/time_dial", split=split, trust_remote_code=True)
    print(f"Loaded TimeDial split '{split}' with {len(ds)} samples")

    if len(ds) == 0:
        raise ValueError("No samples available in the selected split.")

    ds = ds.shuffle(seed=seed)
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    print(f"Prepared {len(ds)} samples (seed={seed})")

    rng = random.Random(seed)
    prepared = []
    skipped = 0
    for sample in ds:
        try:
            options, correct_indices = assemble_options(sample, rng)
        except ValueError:
            skipped += 1
            continue
        prepared.append(
            {
                "conversation": sample["conversation"],
                "options": options,
                "correct_indices": correct_indices,
                "id": sample.get("id"),
                "incorrect_rules": [
                    str(sample.get("incorrect1_rule", "")).strip(),
                    str(sample.get("incorrect2_rule", "")).strip(),
                ],
            }
        )
    if not prepared:
        raise ValueError("No usable samples after preparation.")
    if skipped:
        print(f"Skipped {skipped} samples without enough options")
    return prepared


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, int, List[float], str, List[str], List[int]]:
    prompt = build_prompt(sample["conversation"])
    options = sample["options"]
    correct_indices = sample["correct_indices"]

    scores = [score_option(model, tokenizer, prompt, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx in correct_indices)
    return correct, pred_idx, scores, prompt, options, correct_indices


def evaluate_model_on_dataset(model_path: str, dataset, device: str, desc: str):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 0

    for sample in tqdm(dataset, desc=desc):
        correct, pred_idx, scores, prompt, options, correct_indices = evaluate_sample(
            model, tokenizer, sample
        )

        max_options = max(max_options, len(options))
        evaluated += 1
        if correct == 1:
            correct_count += 1

        row = {
            "prompt": prompt,
            "options": " ||| ".join(options),
            "correct_indices": " ".join(str(i) for i in correct_indices),
            "pred": pred_idx,
            "correct": correct,
            "id": sample.get("id", ""),
            "incorrect_rules": " ||| ".join(sample.get("incorrect_rules", [])),
        }
        for i, score in enumerate(scores):
            row[f"score_{i}"] = score
        rows.append(row)

    if evaluated == 0:
        raise ValueError(f"No valid samples evaluated for {model_path}.")

    accuracy = correct_count / evaluated if evaluated else 0.0
    print(f"{model_path} accuracy: {accuracy:.4f} over {evaluated} samples")
    return accuracy, rows, max_options, evaluated


def write_csv(rows, output_path: str, option_count: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_fields = [
        "model",
        "prompt",
        "options",
        "correct_indices",
        "pred",
        "correct",
        "id",
        "incorrect_rules",
    ]
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
    fieldnames = ["model", "samples", "accuracy", "seed", "split"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal model evaluation on TimeDial MCQ")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/temp_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/temp_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, temp_tag, temp_cat)",
    )
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate",
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
        default="/scratch/craj/langsense/results/inference/temp_timedial_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/temp_timedial_summary.csv",
        help="Where to write the per-model accuracy summary CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = prepare_dataset(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    all_rows = []
    summary_rows = []
    max_option_count = 0

    for model_path in args.model_paths:
        accuracy, rows, model_max_opts, evaluated = evaluate_model_on_dataset(
            model_path=model_path,
            dataset=dataset,
            device=args.device,
            desc=f"Evaluating TimeDial ({os.path.basename(model_path)})",
        )
        max_option_count = max(max_option_count, model_max_opts)
        for row in rows:
            row["model"] = os.path.basename(model_path.rstrip("/"))
            all_rows.append(row)
        summary_rows.append(
            {
                "model": os.path.basename(model_path.rstrip("/")),
                "samples": evaluated,
                "accuracy": accuracy,
                "seed": args.seed,
                "split": args.split,
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
