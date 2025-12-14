#!/usr/bin/env python3
"""
Evaluate merged models on synthetic MCQ datasets in data_synthetic.

- Expects JSON arrays with fields: scenario (str), question (str), options (list[str]), answer (str or index), optional map/category field.
- By default evaluates base_lm on all datasets and tag/cat variants on their matching dataset (e.g., color_tag/color_cat on color_eval).
- Options are shuffled per seed and labeled A/B/C/... in the prompt.
- Runs each model on each dataset for multiple seeds; writes per-seed result CSVs and per-seed summaries, plus aggregated summaries with confidence intervals and binomial p-values vs. random chance.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from math import erf, sqrt
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYNTHETIC_DIR = "/scratch/craj/langsense/data_synthetic"
RESULTS_DIR = "/scratch/craj/langsense/results/inference/synthetic_eval"
MERGED_ROOT = "/scratch/craj/langsense/models/merged"


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


def load_synthetic_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset {path} is empty or not a list.")
    return data


def extract_label(answer_field, options: List[str]) -> int:
    if isinstance(answer_field, int):
        return int(answer_field)
    if isinstance(answer_field, str):
        if answer_field in options:
            return options.index(answer_field)
        if len(answer_field) == 1 and answer_field.isalpha():
            idx = ord(answer_field.upper()) - ord("A")
            if 0 <= idx < len(options):
                return idx
    raise ValueError("Could not resolve answer to an option index.")


def build_prompt(scenario: str, question: str, labeled_options: List[Tuple[str, str]]) -> str:
    opt_lines = [f"{label}) {text}" for label, text in labeled_options]
    opts_str = "\n".join(opt_lines)
    context_part = f"Context: {scenario.strip()}\n" if scenario.strip() else ""
    return (
        f"{context_part}"
        f"Question: {question.strip()}\n"
        "Options:\n"
        f"{opts_str}\n"
        "Choose the correct option letter."
    )


def evaluate_model_on_dataset(
    model_path: str, dataset: List[Dict], device: str, desc: str, seed: int, debug_skips: bool = False
):
    model, tokenizer = load_model(model_path, device)

    rows = []
    correct_count = 0
    evaluated = 0
    max_options = 0
    option_len_sum = 0
    per_map = defaultdict(lambda: {"correct": 0, "total": 0, "options_sum": 0})
    rng = random.Random(seed)

    skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc=desc)):
        scenario = str(sample.get("scenario", "") or "")
        question = str(sample.get("question", "") or scenario)
        options = list(sample.get("options") or [])
        map_field = str(sample.get("map", "") or "")
        sample_id = sample.get("id", "")

        if not question or len(options) < 2:
            skipped += 1
            continue

        try:
            label_idx = extract_label(sample.get("answer"), options)
        except ValueError:
            skipped += 1
            if debug_skips and skipped <= 3:
                print(f"[SKIP] idx={idx} reason=answer mismatch sample={sample}")
            continue

        option_indices = list(range(len(options)))
        rng.shuffle(option_indices)
        shuffled_options = [options[i] for i in option_indices]
        shuffled_labels = [chr(ord("A") + i) for i in range(len(shuffled_options))]
        labeled_options = list(zip(shuffled_labels, shuffled_options))
        shuffled_label_idx = option_indices.index(label_idx)

        prompt = build_prompt(scenario, question, labeled_options)
        scores = [score_option(model, tokenizer, prompt, opt) for opt in shuffled_options]
        pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct = int(pred_idx == shuffled_label_idx)

        max_options = max(max_options, len(options))
        evaluated += 1
        option_len_sum += len(options)
        per_map[map_field]["options_sum"] += len(options)
        if correct == 1:
            correct_count += 1
            per_map[map_field]["correct"] += 1
        per_map[map_field]["total"] += 1

        row = {
            "id": sample_id,
            "scenario": scenario,
            "question": question,
            "options": " ||| ".join(shuffled_options),
            "label": shuffled_label_idx,
            "pred": pred_idx,
            "correct": correct,
            "map": map_field,
            "seed": seed,
        }
        for i, score in enumerate(scores):
            row[f"score_{i}"] = score
        rows.append(row)

    if evaluated == 0:
        print(f"[WARN] No valid samples evaluated for {model_path} (skipped {skipped}) in {desc}")
        return 0.0, rows, max_options, 0, per_map, option_len_sum

    accuracy = correct_count / evaluated
    return accuracy, rows, max_options, evaluated, per_map, option_len_sum


def write_csv(rows, output_path: str, option_count: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_fields = [
        "dataset",
        "id",
        "model",
        "scenario",
        "question",
        "options",
        "label",
        "pred",
        "correct",
        "map",
        "seed",
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
    fieldnames = ["dataset", "model", "map", "samples", "accuracy", "seed", "ci_low", "ci_high", "p_value"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV to {output_path}")


def default_model_groups():
    base = os.path.join(MERGED_ROOT, "base_lm-alpaca-merged")
    mapping = {
        "color": ["color_tag-alpaca-merged", "color_cat-alpaca-merged"],
        "emotion": ["emotion_tag-alpaca-merged", "emotion_cat-alpaca-merged"],
        "gender": ["gender_tag-alpaca-merged", "gender_cat-alpaca-merged"],
        "kin": ["kin_tag-alpaca-merged", "kin_cat-alpaca-merged"],
        "logic": ["logic_tag-alpaca-merged", "logic_cat-alpaca-merged"],
        "num": ["num_tag-alpaca-merged", "num_cat-alpaca-merged"],
        "shape": ["shape_tag-alpaca-merged", "shape_cat-alpaca-merged"],
        "space": ["space_tag-alpaca-merged", "space_cat-alpaca-merged"],
        "temp": ["temp_tag-alpaca-merged", "temp_cat-alpaca-merged"],
        "valence": ["valence_tag-alpaca-merged", "valence_cat-alpaca-merged"],
    }
    groups = {}
    for key, models in mapping.items():
        groups[key] = [base] + [os.path.join(MERGED_ROOT, m) for m in models]
    groups["base_only"] = [base]
    return groups


def build_dataset_plan(custom_root: str = SYNTHETIC_DIR):
    groups = default_model_groups()
    files = [
        ("color", "color_eval.json"),
        ("emotion", "emotion_eval.json"),
        ("gender", "gender_eval.json"),
        ("kin", "kin_eval.json"),
        ("logic", "logic_eval.json"),
        ("num", "num_eval.json"),
        ("shape", "shape_eval.json"),
        ("space", "space_eval.json"),
        ("temp", "temp_eval.json"),
        ("valence", "valence_eval.json"),
    ]
    plan = []
    for key, fname in files:
        path = os.path.join(custom_root, fname)
        models = groups.get(key, groups["base_only"])
        plan.append((key, path, models))
    return plan


def wilson_interval(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 1.96
    phat = correct / total
    denom = 1 + z * z / total
    center = phat + z * z / (2 * total)
    radius = z * sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    low = (center - radius) / denom
    high = (center + radius) / denom
    return max(0.0, low), min(1.0, high)


def binom_test_p_value(k: int, n: int, p: float) -> float:
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return binomtest(k, n, p).pvalue
    except Exception:
        mean = n * p
        var = n * p * (1 - p)
        if var == 0:
            return 1.0
        z = (k - mean) / sqrt(var)
        # two-tailed normal approximation
        return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate merged models on synthetic MCQ datasets")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (e.g., 'cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for dataset shuffling (per-run seeds handled separately)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples per dataset (after shuffling)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[13, 42, 63, 1097, 343],
        help="Seeds to run for each dataset/model (controls option shuffling)",
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory to write per-dataset results",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to run (e.g., num shape space temp valence); defaults to all synthetic datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    plan = build_dataset_plan()
    if args.datasets:
        allowed = set(args.datasets)
        plan = [item for item in plan if item[0] in allowed]
        print(f"Filtered to datasets: {sorted(allowed)}")
    overall_summary = []

    for dataset_key, path, model_paths in plan:
        data = load_synthetic_dataset(path)
        random.shuffle(data)
        if args.max_samples:
            data = data[: args.max_samples]
        print(f"Evaluating dataset {dataset_key} with {len(data)} samples")

        rows_per_seed = defaultdict(list)
        max_opts_per_seed = defaultdict(int)
        dataset_summary = []
        aggregate_stats = defaultdict(lambda: {"correct": 0, "total": 0, "options_sum": 0})

        for model_path in model_paths:
            model_name = os.path.basename(model_path.rstrip("/"))
            for seed in args.seeds:
                accuracy, rows, model_max_opts, evaluated, per_map, option_len_sum = evaluate_model_on_dataset(
                    model_path=model_path,
                    dataset=data,
                    device=args.device,
                    desc=f"{dataset_key} ({model_name}) seed={seed}",
                    seed=seed,
                )
                max_opts_per_seed[seed] = max(max_opts_per_seed[seed], model_max_opts)
                for row in rows:
                    row["model"] = model_name
                    row["dataset"] = dataset_key
                    rows_per_seed[seed].append(row)
                # per-seed summaries
                dataset_summary.append(
                    {
                        "dataset": dataset_key,
                        "model": model_name,
                        "map": "OVERALL",
                        "samples": evaluated,
                        "accuracy": accuracy,
                        "seed": seed,
                        "ci_low": "",
                        "ci_high": "",
                        "p_value": "",
                    }
                )
                for map_name, stats in per_map.items():
                    if not map_name and len(per_map) == 1:
                        continue
                    acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
                    dataset_summary.append(
                        {
                            "dataset": dataset_key,
                            "model": model_name,
                            "map": map_name or "UNSPECIFIED",
                            "samples": stats["total"],
                            "accuracy": acc,
                            "seed": seed,
                            "ci_low": "",
                            "ci_high": "",
                            "p_value": "",
                        }
                    )
                # aggregate for ALL seeds
                aggregate_stats[(model_name, "OVERALL")]["correct"] += int(round(accuracy * evaluated))
                aggregate_stats[(model_name, "OVERALL")]["total"] += evaluated
                aggregate_stats[(model_name, "OVERALL")]["options_sum"] += option_len_sum
                for map_name, stats in per_map.items():
                    key = (model_name, map_name or "UNSPECIFIED")
                    aggregate_stats[key]["correct"] += stats["correct"]
                    aggregate_stats[key]["total"] += stats["total"]
                    aggregate_stats[key]["options_sum"] += stats["options_sum"]
                print(f"Finished {model_path} seed {seed} on {dataset_key} accuracy {accuracy:.4f}")

            # aggregated rows per model/map across seeds
            for (mname, map_name), stats in list(aggregate_stats.items()):
                if mname != model_name:
                    continue
                total = stats["total"]
                correct = stats["correct"]
                acc = correct / total if total else 0.0
                ci_low, ci_high = wilson_interval(correct, total)
                avg_opts = stats["options_sum"] / total if total else 2
                p0 = 1.0 / avg_opts if avg_opts else 0.25
                p_val = binom_test_p_value(correct, total, p0)
                dataset_summary.append(
                    {
                        "dataset": dataset_key,
                        "model": mname,
                        "map": map_name,
                        "samples": total,
                        "accuracy": acc,
                        "seed": "ALL",
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_val,
                    }
                )

        # write per-seed CSVs
        for seed, rows in rows_per_seed.items():
            max_opt = max_opts_per_seed[seed] if rows else 0
            for row in rows:
                for i in range(max_opt):
                    row.setdefault(f"score_{i}", "")
            seed_csv = os.path.join(args.output_dir, f"{dataset_key}_seed{seed}_mcq.csv")
            write_csv(rows, seed_csv, option_count=max_opt)

        # combined summary for this dataset
        write_summary(dataset_summary, os.path.join(args.output_dir, f"{dataset_key}_summary.csv"))
        overall_summary.extend(dataset_summary)

    # combined summary across all datasets
    write_summary(overall_summary, os.path.join(args.output_dir, "all_synthetic_summary.csv"))


if __name__ == "__main__":
    main()
