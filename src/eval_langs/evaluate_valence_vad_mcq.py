#!/usr/bin/env python3
"""
Evaluate valence models on MCQ questions built from the NRC VAD lexicon.

- Loads an NRC VAD dataset from Hugging Face (configurable name/split/config).
- Converts each lexical entry into a binary MCQ: is the word positive or negative valence?
- Valence threshold is auto-detected (0.5 if scores in [0,1], else 5.0) and can be overridden.
- Scores options by conditional log-likelihood given the prompt (word + question).
- Writes per-sample CSV and per-model accuracy summary.
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Candidate keys for word and valence columns (flexible to allow alternate headers)
WORD_KEYS = ["word", "Word", "token", "term", "Term", "TargetWord"]
VALENCE_KEYS = ["valence", "Valence", "V.Mean.Sum", "Val.Mean.Sum", "V-ANEW", "V.Mean"]


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


def extract_word_and_valence(sample: Dict) -> Tuple[str, float]:
    word = None
    for key in WORD_KEYS:
        if key in sample and str(sample[key]).strip():
            word = str(sample[key]).strip()
            break
    if word is None:
        # fallback: first string field
        for val in sample.values():
            if isinstance(val, str) and val.strip():
                word = val.strip()
                break
    if word is None:
        raise ValueError("No word/token found in sample.")

    valence = None
    for key in VALENCE_KEYS:
        if key in sample:
            try:
                valence = float(sample[key])
                break
            except (TypeError, ValueError):
                continue
    if valence is None:
        # fallback: first numeric field
        for val in sample.values():
            try:
                valence = float(val)
                break
            except (TypeError, ValueError):
                continue
    if valence is None:
        raise ValueError("No valence score found in sample.")
    return word, valence


def detect_valence_threshold(samples: List[Dict], default: float) -> float:
    vals = []
    for sample in samples[:200]:
        try:
            _, val = extract_word_and_valence(sample)
            vals.append(val)
        except ValueError:
            continue
    if not vals:
        return default
    vmax = max(vals)
    return 0.5 if vmax <= 1.0 else 5.0


def build_prompt(word: str, pronoun: str = "") -> str:
    base = f"Consider the word: {word}."
    question = "Choose whether its valence (emotional positivity) is positive or negative."
    return base + " " + question + "\n"


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


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str], threshold: float
) -> Tuple[int, int, List[float], str, List[str], int]:
    word, valence = extract_word_and_valence(sample)
    prompt = build_prompt(word)

    options = ["positive valence", "negative valence"]
    label_idx = 0 if valence >= threshold else 1

    scores = [score_option(model, tokenizer, prompt, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx == label_idx)
    return correct, pred_idx, scores, prompt, options, label_idx


def load_lexicon(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # skip empty lines
            if not row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from lexicon: {path}")
    return rows


def prepare_dataset(
    lexicon_path: str,
    max_samples: int,
    seed: int,
    threshold: float,
):
    raw_rows = load_lexicon(lexicon_path)
    print(f"Loaded lexicon '{lexicon_path}' with {len(raw_rows)} entries")

    auto_thresh = detect_valence_threshold(raw_rows, default=threshold)
    if threshold is None:
        threshold = auto_thresh
    print(f"Using valence threshold: {threshold} (auto-detected {auto_thresh})")

    random.Random(seed).shuffle(raw_rows)
    if max_samples:
        max_samples = min(max_samples, len(raw_rows))
        raw_rows = raw_rows[:max_samples]
    print(f"Prepared {len(raw_rows)} samples (seed={seed})")
    return raw_rows, threshold


def evaluate_model_on_dataset(model_path: str, dataset, device: str, desc: str, threshold: float):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 2

    for sample in tqdm(dataset, desc=desc):
        try:
            correct, pred_idx, scores, prompt, options, label_idx = evaluate_sample(
                model, tokenizer, sample, threshold
            )
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        evaluated += 1
        if correct == 1:
            correct_count += 1

        row = {
            "prompt": prompt,
            "options": " ||| ".join(options),
            "label": label_idx,
            "pred": pred_idx,
            "correct": correct,
            "valence_threshold": threshold,
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
    base_fields = ["model", "prompt", "options", "label", "pred", "correct", "valence_threshold"]
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
    fieldnames = ["model", "samples", "accuracy", "seed", "lexicon_path", "threshold"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Valence model evaluation on NRC VAD MCQ")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/valence_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/valence_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, valence_tag, valence_cat)",
    )
    parser.add_argument(
        "--lexicon-path",
        default="/scratch/craj/langsense/data/vad/unigrams-valence-NRC-VAD-Lexicon-v2.1.txt",
        help="Path to local NRC VAD lexicon TSV (term<tab>valence)",
    )
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
        "--threshold",
        type=float,
        default=None,
        help="Valence threshold for positive vs negative (auto if not set)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (e.g., 'cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--output",
        default="/scratch/craj/langsense/results/inference/valence_vad_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/valence_vad_summary.csv",
        help="Where to write the per-model accuracy summary CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset, threshold = prepare_dataset(
        lexicon_path=args.lexicon_path,
        max_samples=args.max_samples,
        seed=args.seed,
        threshold=args.threshold,
    )

    all_rows = []
    summary_rows = []
    max_option_count = 0

    for model_path in args.model_paths:
        accuracy, rows, model_max_opts, evaluated = evaluate_model_on_dataset(
            model_path=model_path,
            dataset=dataset,
            device=args.device,
            desc=f"Evaluating VAD ({os.path.basename(model_path)})",
            threshold=threshold,
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
                "lexicon_path": args.lexicon_path,
                "threshold": threshold,
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
