#!/usr/bin/env python3
"""
Evaluate gender models on the Winogender MCQ dataset.

- Uses oskarvanderwal/winogender from Hugging Face (train/validation/test splits).
- Scores each option by conditional log-likelihood given the sentence/context.
- Shuffles with a fixed seed and evaluates up to --max-samples.
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


def build_prompt(sample: Dict[str, str]) -> str:
    """Construct the prompt from sentence/context text."""
    sentence = str(
        sample.get("sentence")
        or sample.get("text")
        or sample.get("context")
        or ""
    ).strip()
    if not sentence:
        raise ValueError("Missing sentence/context text in sample.")

    extra = []
    pronoun = str(sample.get("pronoun") or "").strip()
    if pronoun:
        extra.append(f"Pronoun: {pronoun}")
    return sentence + ("\n" + "\n".join(extra) if extra else "") + "\n"


def _normalize_options(val: object) -> List[str]:
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        opts = [str(v) for v in val if str(v).strip()]
        if len(opts) >= 2:
            return opts
    return []


def extract_options(sample: Dict) -> List[str]:
    # Winogender provides two entities: occupation vs participant.
    occupation = sample.get("occupation")
    participant = sample.get("participant")
    if occupation and participant:
        opts = [str(occupation).strip(), str(participant).strip()]
        if all(opts):
            return opts

    candidate_keys = ["options", "choices", "answers", "candidates", "referents", "entities"]
    for key in candidate_keys:
        if key in sample:
            opts = _normalize_options(sample[key])
            if opts:
                return opts

    paired_keys = [
        ("option1", "option2"),
        ("option_a", "option_b"),
        ("a", "b"),
        ("A", "B"),
        ("candidate_a", "candidate_b"),
        ("coref_a", "coref_b"),
    ]
    for first, second in paired_keys:
        if first in sample and second in sample:
            opts = [str(sample[first]), str(sample[second])]
            if all(opt.strip() for opt in opts):
                return opts

    raise ValueError("Sample missing answer options.")


def extract_label_index(sample: Dict, options: List[str]) -> int:
    # Winogender labels: 0 = occupation, 1 = participant
    if "target" in sample and sample["target"] is not None:
        target = str(sample["target"])
        if target == str(sample.get("occupation")):
            return 0
        if target == str(sample.get("participant")):
            return 1

    if "label" in sample and sample["label"] is not None:
        return int(sample["label"])
    if "answer_index" in sample and sample["answer_index"] is not None:
        return int(sample["answer_index"])
    if "answer" in sample and sample["answer"] is not None:
        ans = sample["answer"]
        if isinstance(ans, int):
            return int(ans)
        if isinstance(ans, str):
            upper = ans.strip().upper()
            if len(upper) == 1 and "A" <= upper <= "Z":
                return ord(upper) - ord("A")
            if upper in ("1", "2"):
                return int(upper) - 1
            if ans in options:
                return options.index(ans)
    return -1


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
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, int, List[float], str, List[str], int]:
    prompt = build_prompt(sample)
    options = extract_options(sample)
    label_idx = extract_label_index(sample, options)

    scores = [score_option(model, tokenizer, prompt, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx == label_idx) if label_idx in range(len(options)) else -1
    return correct, pred_idx, scores, prompt, options, label_idx


def prepare_dataset(split: str, max_samples: int, seed: int, config: str):
    ds = load_dataset("oskarvanderwal/winogender", config, split=split)
    print(f"Loaded Winogender ({config}) split '{split}' with {len(ds)} samples")

    if len(ds) == 0:
        raise ValueError("No samples available in the selected split.")

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

    for sample in tqdm(dataset, desc=desc):
        try:
            correct, pred_idx, scores, prompt, options, label_idx = evaluate_sample(
                model, tokenizer, sample
            )
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

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
    base_fields = ["model", "prompt", "options", "label", "pred", "correct"]
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
    parser = argparse.ArgumentParser(description="Gender model evaluation on Winogender MCQ")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/gender_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/gender_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, gender_tag, gender_cat)",
    )
    parser.add_argument("--config", default="all", choices=["all", "gotcha"], help="Winogender config to use")
    parser.add_argument("--split", default="test", help="Dataset split to use (Winogender provides 'test')")
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
        default="/scratch/craj/langsense/results/inference/gender_winogender_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/gender_winogender_summary.csv",
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
    )

    all_rows = []
    summary_rows = []
    max_option_count = 0
    total_samples = 0

    for model_path in args.model_paths:
        accuracy, rows, model_max_opts, evaluated = evaluate_model_on_dataset(
            model_path=model_path,
            dataset=dataset,
            device=args.device,
            desc=f"Evaluating Winogender ({os.path.basename(model_path)})",
        )
        max_option_count = max(max_option_count, model_max_opts)
        total_samples = max(total_samples, evaluated)
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
