#!/usr/bin/env python3
"""
Evaluate emotion models on GoEmotions by forming MCQ classification prompts.

- Uses google-research-datasets/go_emotions (default config: simplified).
- Converts each sample to an MCQ: pick the best emotion label among options.
- For multi-label samples, one gold label is chosen deterministically per seed; distractors are sampled from other labels.
- Shuffles with a fixed seed and evaluates up to --max-samples.
- Writes per-sample CSV and per-model accuracy summary.
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Sequence, Tuple

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


def build_prompt(text: str) -> str:
    return f"Text: {text.strip()}\nQuestion: Which emotion best describes this text?\n"


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


def make_options(
    label_names: Sequence[str],
    gold_label_ids: Sequence[int],
    num_options: int,
    rng: random.Random,
) -> Tuple[List[str], int, str]:
    if not gold_label_ids:
        raise ValueError("Sample has no labels.")
    # Deterministic choice given seeded rng
    gold_id = rng.choice(list(gold_label_ids))
    gold_label = label_names[gold_id]

    remaining = [name for idx, name in enumerate(label_names) if idx not in gold_label_ids]
    needed_neg = max(0, num_options - 1)
    if needed_neg > len(remaining):
        needed_neg = len(remaining)
    distractors = rng.sample(remaining, needed_neg)

    options = [gold_label] + distractors
    rng.shuffle(options)
    label_idx = options.index(gold_label)
    return options, label_idx, gold_label


def prepare_dataset(
    split: str,
    max_samples: int,
    seed: int,
    config: str,
    num_options: int,
):
    ds = load_dataset("google-research-datasets/go_emotions", config, split=split)
    label_names = ds.features["labels"].feature.names
    print(f"Loaded GoEmotions ({config}) split '{split}' with {len(ds)} samples and {len(label_names)} labels")

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
        labels = sample.get("labels") or []
        if not labels:
            skipped += 1
            continue
        try:
            options, label_idx, gold_label = make_options(
                label_names=label_names,
                gold_label_ids=labels,
                num_options=num_options,
                rng=rng,
            )
        except ValueError:
            skipped += 1
            continue

        prepared.append(
            {
                "text": sample["text"],
                "options": options,
                "label_idx": label_idx,
                "gold_label": gold_label,
                "all_labels": [label_names[i] for i in labels],
            }
        )

    if not prepared:
        raise ValueError("No usable samples after preparation.")
    if skipped:
        print(f"Skipped {skipped} samples without labels")
    return prepared


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, int, List[float], str, List[str], int]:
    prompt = build_prompt(sample["text"])
    options = sample["options"]
    label_idx = sample["label_idx"]

    scores = [score_option(model, tokenizer, prompt, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx == label_idx)
    return correct, pred_idx, scores, prompt, options, label_idx


def evaluate_model_on_dataset(model_path: str, dataset, device: str, desc: str):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 0

    for sample in tqdm(dataset, desc=desc):
        correct, pred_idx, scores, prompt, options, label_idx = evaluate_sample(
            model, tokenizer, sample
        )

        max_options = max(max_options, len(options))
        evaluated += 1
        if correct == 1:
            correct_count += 1

        row = {
            "prompt": prompt,
            "options": " ||| ".join(options),
            "label": label_idx,
            "pred": pred_idx,
            "correct": correct,
            "gold_label": sample.get("gold_label", ""),
            "all_labels": " ||| ".join(sample.get("all_labels", [])),
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
        "label",
        "pred",
        "correct",
        "gold_label",
        "all_labels",
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
    fieldnames = ["model", "samples", "accuracy", "seed", "split", "config", "num_options"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion model evaluation on GoEmotions MCQ")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/emotion_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/emotion_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, emotion_tag, emotion_cat)",
    )
    parser.add_argument("--config", default="simplified", help="GoEmotions config to use (e.g., simplified)")
    parser.add_argument("--split", default="validation", help="Dataset split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for shuffling/selecting samples",
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=4,
        help="Number of MCQ options (1 correct + distractors; capped by label count)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (e.g., 'cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--output",
        default="/scratch/craj/langsense/results/inference/emotion_goemotions_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/emotion_goemotions_summary.csv",
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
        config=args.config,
        num_options=args.num_options,
    )

    all_rows = []
    summary_rows = []
    max_option_count = 0

    for model_path in args.model_paths:
        accuracy, rows, model_max_opts, evaluated = evaluate_model_on_dataset(
            model_path=model_path,
            dataset=dataset,
            device=args.device,
            desc=f"Evaluating GoEmotions ({os.path.basename(model_path)})",
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
                "config": args.config,
                "num_options": args.num_options,
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
