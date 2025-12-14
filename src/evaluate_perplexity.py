#!/usr/bin/env python3
"""
Evaluate perplexity of HuggingFace causal models on either:
  (1) HF-formatted dataset (DatasetDict or a split folder)
  (2) BabyLM-style plain text dataset folder containing .test files

Randomly shuffles all samples before selecting max_samples.
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, Dataset
from datetime import datetime


class PerplexityEvaluator:
    def __init__(self, model_path, device="auto"):
        print(f"Loading model: {model_path}")

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    # ------------------------------------------------------------------
    # FIXED: this is now a real class method
    # ------------------------------------------------------------------
    def calculate_perplexity(self, text, max_length=128):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        )
        input_ids = enc.input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=None)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        losses = losses.view(shift_labels.shape)

        content_losses = losses[0]

        if len(content_losses) == 0:
            return float('inf')

        avg_loss = content_losses.mean()
        ppl = torch.exp(avg_loss)
        return ppl.item()

    # ----------------------------------------------------------------------
    def load_hf_dataset(self, folder):
        try:
            if os.path.exists(os.path.join(folder, "test")):
                print("Detected HF Dataset split directory.")
                return load_from_disk(os.path.join(folder, "test"))

            print("Detected HF Dataset root folder.")
            ds = load_from_disk(folder)

            if hasattr(ds, "keys"):
                if "test" in ds:
                    return ds["test"]
                else:
                    first = list(ds.keys())[0]
                    print(f"HF dataset has no 'test'. Using: {first}")
                    return ds[first]

            return ds

        except Exception:
            raise ValueError("Not HF-format")

    # ----------------------------------------------------------------------
    def load_plaintext_dataset(self, folder):
        print("Loading .test files (plain text mode)...")
        samples = []

        for fname in os.listdir(folder):
            if fname.endswith(".test"):
                with open(os.path.join(folder, fname), "r", encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append({"text": line})

        return Dataset.from_list(samples)

    # ----------------------------------------------------------------------
    def evaluate(self, dataset_path, max_samples=None):
        print(f"\nTrying to load dataset from: {dataset_path}")

        # (1) Try HF
        try:
            dataset = self.load_hf_dataset(dataset_path)
            print(f"Loaded HF dataset with {len(dataset)} samples.")
        except Exception:
            print("HF loading failed → falling back to plain text.")
            dataset = self.load_plaintext_dataset(dataset_path)
            print(f"Loaded {len(dataset)} lines from .test files.")

        # (2) Shuffle
        dataset = dataset.shuffle(seed=42)

        # (3) Subsample
        if max_samples:
            max_samples = min(max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))
            print(f"Evaluating on {max_samples} shuffled samples")

        # (4) Loop
        perplexities = []
        for i, item in enumerate(tqdm(dataset, desc="Calculating perplexity")):
            text = item["text"]
            try:
                ppl = self.calculate_perplexity(text)
                if np.isfinite(ppl):
                    perplexities.append(ppl)
            except Exception as e:
                print(f"Error at sample {i}: {e}")

        if len(perplexities) == 0:
            print("No valid perplexity values.")
            return None

        result = {
            "model": os.path.basename(self.model_path.rstrip("/")),
            "dataset": os.path.basename(dataset_path.rstrip("/")),
            "samples": len(perplexities),
            "mean": float(np.mean(perplexities)),
            "median": float(np.median(perplexities)),
            "std": float(np.std(perplexities)),
            "min": float(np.min(perplexities)),
            "max": float(np.max(perplexities)),
            "timestamp": datetime.now().isoformat()
        }

        print("\n=== Perplexity Results ===")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        # ------------------------------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------------------------------
        save_dir = "/scratch/craj/langsense/results"
        os.makedirs(save_dir, exist_ok=True)

        filename = (
            f"{result['model']}__{result['dataset']}__"
            f"{result['samples']}_samples.json"
        )
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nResults saved to: {filepath}\n")

        return result


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    evaluator = PerplexityEvaluator(args.model_path, args.device)
    evaluator.evaluate(args.dataset_path, args.max_samples)


if __name__ == "__main__":
    main()
