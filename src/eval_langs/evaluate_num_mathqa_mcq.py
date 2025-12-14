#!/usr/bin/env python3
"""
Evaluate numeric reasoning models on math_qa MCQ.

- Uses jet-ai/math_qa from Hugging Face (train/validation/test splits).
- Parses MCQ options and correct answer; skips malformed samples.
- Scores each option by conditional log-likelihood given the question.
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


def extract_question(sample: Dict) -> str:
    for key in ["Problem", "question", "prompt", "text"]:
        if key in sample and sample[key]:
            return str(sample[key])
    raise ValueError("No question text found in sample.")


def _normalize_options(val: object) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        if all(isinstance(x, str) and x.strip() for x in val):
            return list(val)
    if isinstance(val, dict) and len(val) >= 2:
        items = [val[k] for k in sorted(val.keys())]
        if all(isinstance(x, str) and x.strip() for x in items):
            return items
    return []


OPTION_REGEX = re.compile(
    r"(?i)([a-e])\s*[\)\.\:]\s*(.+?)(?=\s*[a-e]\s*[\)\.\:]\s|$)",
    re.DOTALL,
)


def extract_options(sample: Dict) -> List[str]:
    candidate_keys = [
        "options",
        "choices",
        "options_text",
        "choices_text",
        "answers",
        "options_list",
    ]
    for key in candidate_keys:
        if key in sample:
            # Direct list/dict formats
            opts = _normalize_options(sample[key])
            if opts:
                return opts
            # String formats like "a) ... b) ... c) ..."
            if isinstance(sample[key], str):
                matches = OPTION_REGEX.findall(sample[key])
                if matches:
                    return [opt.strip() for _, opt in matches if opt.strip()]
                comma_caps = re.findall(r"(?i)[a-e]\s*[\)\.\:]\s*([^,]+)", sample[key])
                if comma_caps and len(comma_caps) >= 2:
                    return [c.strip() for c in comma_caps if c.strip()]
                split_opts = [
                    part.strip()
                    for part in re.split(r"(?i)[a-e][\)\.\:]", sample[key])
                    if part.strip()
                ]
                if len(split_opts) >= 2:
                    return split_opts

    # Fallback: scan any string field for regex matches
    for val in sample.values():
        if isinstance(val, str):
            matches = OPTION_REGEX.findall(val)
            if matches:
                opts = [opt.strip() for _, opt in matches if opt.strip()]
                if len(opts) >= 2:
                    return opts
            comma_caps = re.findall(r"(?i)[a-e]\s*[\)\.\:]\s*([^,]+)", val)
            if comma_caps and len(comma_caps) >= 2:
                return [c.strip() for c in comma_caps if c.strip()]
            split_opts = [
                part.strip()
                for part in re.split(r"(?i)[a-e][\)\.\:]", val)
                if part.strip()
            ]
            if len(split_opts) >= 2:
                return split_opts
        else:
            opts = _normalize_options(val)
            if opts:
                return opts

    raise ValueError("No options/choices found in sample.")


def extract_label_index(sample: Dict, options: List[str]) -> int:
    # math_qa uses letters (a/b/c/...) in "correct"
    if "correct" in sample and sample["correct"] is not None:
        ans = str(sample["correct"]).strip()
        if len(ans) == 1 and ans.isalpha():
            idx = ord(ans.lower()) - ord("a")
            if 0 <= idx < len(options):
                return idx
    if "answer_index" in sample and sample["answer_index"] is not None:
        return int(sample["answer_index"])
    if "label" in sample and sample["label"] is not None:
        return int(sample["label"])
    if "answer" in sample and sample["answer"] is not None:
        ans = sample["answer"]
        if isinstance(ans, int):
            return int(ans)
        if isinstance(ans, str):
            upper = ans.strip().upper()
            if len(upper) == 1 and "A" <= upper <= "Z":
                return ord(upper) - ord("A")
            if ans in options:
                return options.index(ans)
    raise ValueError("Could not determine correct answer index.")


def evaluate_sample(
    model, tokenizer, sample: Dict[str, str]
) -> Tuple[int, int, List[float], str, List[str], int]:
    question = extract_question(sample)
    options = extract_options(sample)
    label_idx = extract_label_index(sample, options)

    scores = [score_option(model, tokenizer, question, opt) for opt in options]
    pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = int(pred_idx == label_idx)
    return correct, pred_idx, scores, question, options, label_idx


def prepare_dataset(
    split: str, max_samples: int, seed: int, print_category_stats: bool
):
    ds = load_dataset("jet-ai/math_qa", split=split)
    print(f"Loaded math_qa split '{split}' with {len(ds)} samples")

    if print_category_stats:
        from collections import Counter

        counts = Counter(str(c).lower() for c in ds["category"])
        print("Top categories:", counts.most_common(20))

    if len(ds) == 0:
        raise ValueError("No samples available in the selected split.")

    ds = ds.shuffle(seed=seed)
    if max_samples:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    print(f"Prepared {len(ds)} samples (seed={seed})")
    return ds


def evaluate_model_on_dataset(
    model_path: str, dataset, device: str, desc: str, debug_skips: bool = False
):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 0
    skipped = 0
    debug_samples = []

    def print_sample_fields(tag: str, idx: int, sample: Dict, err: str):
        print(f"[DEBUG-{tag}] sample idx={idx} err={err}")
        for key in [
            "Problem",
            "question",
            "prompt",
            "text",
            "options",
            "options_text",
            "choices",
            "correct",
            "label",
            "answer_index",
        ]:
            if key in sample:
                val = sample[key]
                sval = str(val)
                if len(sval) > 400:
                    sval = sval[:400] + "...<truncated>"
                print(f"  {key}: {sval}")

    for idx, sample in enumerate(tqdm(dataset, desc=desc)):
        try:
            correct, pred_idx, scores, question, options, label_idx = evaluate_sample(
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
            "question": question,
            "options": " ||| ".join(options),
            "label": label_idx,
            "pred": pred_idx,
            "correct": correct,
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
    base_fields = ["model", "question", "options", "label", "pred", "correct"]
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
    parser = argparse.ArgumentParser(description="Numeric model evaluation on math_qa MCQ")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "/scratch/craj/langsense/models/merged/base_lm-alpaca-merged",
            "/scratch/craj/langsense/models/merged/num_tag-alpaca-merged",
            "/scratch/craj/langsense/models/merged/num_cat-alpaca-merged",
        ],
        help="List of model paths to evaluate (default: base_lm, num_tag, num_cat)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
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
        "--print-category-stats",
        action="store_true",
        help="Print the top category counts before evaluation",
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
        default="/scratch/craj/langsense/results/inference/num_mathqa_mcq.csv",
        help="Where to write the combined per-sample CSV results",
    )
    parser.add_argument(
        "--summary-output",
        default="/scratch/craj/langsense/results/inference/num_mathqa_summary.csv",
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
        print_category_stats=args.print_category_stats,
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
            desc=f"Evaluating math_qa ({os.path.basename(model_path)})",
            debug_skips=args.debug_skips,
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
