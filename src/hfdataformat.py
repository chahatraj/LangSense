import os
import argparse
import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk


def load_files(folder, ext):
    rows = []
    for filename in os.listdir(folder):
        if filename.endswith(ext):
            path = os.path.join(folder, filename)
            print(f"Reading {filename}...")

            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                rows.append({"text": line, "source": filename})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str,
                        help="Language folder (e.g., color_lang)")
    args = parser.parse_args()

    base = "/scratch/craj/langsense/data/new_langs"
    lang_cat = f"{base}/{args.lang}/categorical"

    # Input folders
    train_dir = f"{lang_cat}/train"
    val_dir = f"{lang_cat}/validation"
    test_dir = "/scratch/craj/langsense/data/baby_lm/test"

    # Output folder
    output_dir = f"{lang_cat}/hf_dataset"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== BUILDING HF DATASET FOR: {args.lang} ===")

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------
    print("\nLoading train split (.train files)...")
    train_rows = load_files(train_dir, ".train")
    print(f"Train samples: {len(train_rows)}")

    train_df = pd.DataFrame(train_rows)
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------
    print("\nLoading validation split (.dev files)...")
    val_rows = load_files(val_dir, ".dev")
    print(f"Validation samples: {len(val_rows)}")

    val_df = pd.DataFrame(val_rows)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

    # ---------------------------------------------------
    # TEST
    # ---------------------------------------------------
    print("\nLoading test split (.test files)...")
    test_rows = load_files(test_dir, ".test")
    print(f"Test samples: {len(test_rows)}")

    test_df = pd.DataFrame(test_rows)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # ---------------------------------------------------
    # SAVE TRAIN + VAL TO DATASET_DICT
    # ---------------------------------------------------
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    print(f"\nSaving full dataset (train/validation/test) to: {output_dir}")
    dataset_dict.save_to_disk(output_dir)

    # ---------------------------------------------------
    # ALSO SAVE TEST SPLIT AS SEPARATE FOLDER (OPTIONAL)
    # ---------------------------------------------------
    test_out_dir = f"{output_dir}/test"
    os.makedirs(test_out_dir, exist_ok=True)

    print(f"Saving test split separately to: {test_out_dir}")
    test_dataset.save_to_disk(test_out_dir)

    # ---------------------------------------------------
    # UPDATE dataset_dict.json
    # ---------------------------------------------------
    metadata_path = f"{output_dir}/dataset_dict.json"
    print(f"Updating metadata: {metadata_path}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    if "test" not in meta["splits"]:
        meta["splits"].append("test")

    with open(metadata_path, "w") as f:
        json.dump(meta, f)

    print("\n=== DONE. ALL SPLITS BUILT & REGISTERED ===")


if __name__ == "__main__":
    main()