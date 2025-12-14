import os
import re
import argparse
from tqdm import tqdm
import nltk

# Ensure wordnet is available
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn


# ------------------------------------------------------------
# LOAD VALENCE LEXICON & FILTER TO ADJECTIVES
# ------------------------------------------------------------

def is_adjective(word):
    """Return True if word is an adjective in WordNet (pos='a' or 's')."""
    for syn in wn.synsets(word):
        if syn.pos() in ("a", "s"):
            return True
    return False


def load_valence_map(path):
    """
    Returns dict: word -> category {positive, negative, neutral}
    using valence scores from NRC-VAD lexicon.
    """
    val_map = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            term = parts[0].strip().lower()
            try:
                score = float(parts[1])
            except:
                continue

            # adjective filtering
            if not is_adjective(term):
                continue

            # categorization by valence
            if score >= 0.66:
                val_map[term] = "positive"
            elif score <= 0.33:
                val_map[term] = "negative"
            else:
                val_map[term] = "neutral"

    return val_map


# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------
TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# REPLACEMENT LOGIC
# ------------------------------------------------------------

def replace_val(tok, mode, val_map):
    low = tok.lower()
    if low not in val_map:
        return tok

    if mode == "token":
        return "valence"

    elif mode == "tag":
        return "<VALENCE>"

    elif mode == "categorical":
        return val_map[low]

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
def transform_line(line, mode, val_map, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out = []

    for tok in tokens:
        low = tok.lower()

        if low in val_map:
            stats["total"] += 1
            out.append(replace_val(tok, mode, val_map))
        else:
            out.append(tok)

    out = " ".join(out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out


# ------------------------------------------------------------
def count_total_lines(in_dir):
    total = 0
    for fname in os.listdir(in_dir):
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


# ------------------------------------------------------------
def process_all_files(in_dir, out_dir, mode, val_map):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(os.listdir(in_dir))
    print("Counting total lines...")
    total_lines = count_total_lines(in_dir)
    print(f"Total lines: {total_lines:,}")

    stats = {"total": 0}
    pbar = tqdm(total=total_lines, desc="Processing")

    for fname in files:
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            for line in fin:
                fout.write(transform_line(line.rstrip("\n"), mode, val_map, stats) + "\n")
                pbar.update(1)

    pbar.close()

    print("\nValenceLang construction complete.")
    print(f"TOTAL ADJECTIVE VALENCE TERMS REPLACED: {stats['total']:,}\n")


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="token",
                        choices=["token", "tag", "categorical"],
                        help="Replacement style")

    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"],
                        help="Dataset split")

    parser.add_argument("--lexicon", type=str,
                        default="/scratch/craj/langsense/data/unigrams-valence-NRC-VAD-Lexicon-v2.1.txt",
                        help="Path to NRC-VAD valence lexicon")

    args = parser.parse_args()

    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"
    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/valence_lang/"
        f"{args.mode}/{args.split}"
    )

    print("\nLoading valence adjectives...")
    val_map = load_valence_map(args.lexicon)
    print(f"Loaded {len(val_map):,} adjective valence terms.")

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode, val_map)


if __name__ == "__main__":
    main()
