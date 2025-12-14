#!/usr/bin/env python3
"""
Evaluate kinship models on the BloodRelations MCQ dataset.

- Uses deeponh/bloodrelations (train split).
- Parses MCQ options from the pipe-separated string; maps letter answers to indices.
- Scores each option by conditional log-likelihood given passage + question.
- Shuffles with a fixed seed and evaluates up to --max-samples.
- Writes per-sample CSV and per-model accuracy summary.
"""

import argparse
import csv
import os
import random
import re
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


def build_prompt(passage: str, question: str) -> str:
    return f"Passage: {passage.strip()}\nQuestion: {question.strip()}\nChoose the correct answer.\n"


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


LETTER_REGEX = re.compile(r"^[A-Za-z]$")


def parse_options(options_str: str) -> List[str]:
    # Expected format: "A. Mother|B. Father|C. Uncle|D. Aunt|E. None of these"
    parts = options_str.split("|")
    cleaned = []
    for part in parts:
        # remove leading letter markers
        if "." in part:
            part = part.split(".", 1)[1]
        elif ")" in part:
            part = part.split(")", 1)[1]
        cleaned.append(part.strip())
    cleaned = [c for c in cleaned if c]
    if len(cleaned) < 2:
        raise ValueError("Could not parse options")
    return cleaned


def label_to_index(label: str, options_count: int) -> int:
    if not label:
        return -1
    if len(label) == 1 and LETTER_REGEX.match(label):
        idx = ord(label.upper()) - ord("A")
        if 0 <= idx < options_count:
            return idx
    return -1


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, int, List[float], str, List[str], int]:
    passage = sample.get("Passage", "") or ""
    question = sample.get("Question", "") or ""
    if not passage or not question:
        raise ValueError("Missing passage or question")

    options = parse_options(sample.get("Options", "") or "")
    label_idx = label_to_index(str(sample.get("Answer", "")).strip(), len(options))
    if label_idx < 0:
        raise ValueError("Invalid label")

    prompt = build_prompt(passage, question)
    scores = [score_option(model, tokenizer, prompt, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx == label_idx)
    return correct, pred_idx, scores, prompt, options, label_idx


def prepare_dataset(split: str, max_samples: int, seed: int):
    ds = load_dataset("deeponh/bloodrelations", split=split)
    print(f"Loaded bloodrelations split '{split}' with {len(ds)} samples")

    if len(ds) == 0:
        raise ValueError("No samples available in the selected split.")

    ds = ds.shuffle(seed=seed)
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    print(f"Prepared {len(ds)} samples (seed={seed})")
    return ds


def evaluate_model_on_dataset(model_path: str, dataset, device: str, desc: str, debug_skips: bool = False):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 0
    skipped = 0
    debug_samples = []

    def print_sample_fields(tag: str, idx: int, sample: Dict, err: str):
        print(f"[DEBUG-{tag}] sample idx={idx} err={err}")
        for key in ["Passage", "Question", "Options", "Answer"]:
            if key in sample:
                sval = str(sample[key])
                if len(sval) > 400:
                    sval = sval[:400] + "...<truncated>"
                print(f"  {key}: {sval}")

    for idx, sample in enumerate(tqdm(dataset, desc=desc)):
        try:
            correct, pred_idx, scores, prompt, options, label_idx = evaluate_sample(
                model, tokenizer, sample
            )
        except ValueError as e:
            skipped += 1
            if debug_skips or skipped <= 5:
                print_sample_fields("skip", idx, sample, str(e))
            if len(debug_samples) < 5:
                debug_samples.append((idx, str(e), sample))
            continue

        max_options = max(max_options, len(options))
        evaluated += 1

        row = {
            "passage": passage if (passage := sample.get("Passage")) else "",
            "question": question if (question := sample.get("Question")) else "",
            "options": " ||| ".join(options),
            "label": label_idx,
            "pred": pred_idx,
            "correct": correct,
            "set_number": sample.get("Set Number", ""),
        }
        for i, score in enumerate(scores):
            row[f"score_{i}"] = score
        rows.append(row)
        if correct == 1:
            correct_count += 1

    if evaluated == 0:
        msg = (
            f"No valid samples for model {model_path}. "
            f"Skipped {skipped} samples due to missing options/labels/questions."
        )
        if debug_samples:
            print("[DEBUG] Showing up to 5 skipped samples:")
            for i, (idx, err, sample) in enumerate(debug_samples):
                print_sample_fields(f"summary-{i+1}", idx, sample, err)
        raise ValueError(msg)

    accuracy = correct_count / evaluated
    print(
        f"{model_path} accuracy: {accuracy:.4f} over {evaluated} samples (skipped {skipped})"
    )
    return accuracy, rows, max_options, evaluated


def write_csv(rows, output_path: str, option_count: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_fields = ["model", "passage", "question", "options", "label", "pred", "correct", "set_number"]
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
    parser = argparse.ArgumentParser(description="Kinship model evaluation on BloodRelations MCQ")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/kin_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/kin_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, kin_tag, kin_cat)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=146,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for shuffling/selecting samples",
    )
    parser.add_argument(
        "--debug-skips",
        action="store_true",
        help="Print details for the first few skipped samples",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (e.g., 'cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--output",
        default="/scratch/craj/langsense/results/inference/kin_bloodrelations_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/kin_bloodrelations_summary.csv",
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
            desc=f"Evaluating BloodRelations ({os.path.basename(model_path)})",
            debug_skips=args.debug_skips,
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
