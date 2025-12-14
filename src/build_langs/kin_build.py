import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# Kinship categories for categorical mode
# ------------------------------------------------------------

ELDERS = {
    "father", "mother", "dad", "mom",
    "uncle", "aunt", "parent",
    "grandfather", "grandmother", "grandpa", "grandma"
}

YOUNGERS = {
    "son", "daughter",
    "nephew", "niece",
    "child"
}

SIMILAR_AGE = {
    "brother", "sister",
    "cousin",
    "husband", "wife"
}

# Full list
KIN_WORDS = sorted(ELDERS | YOUNGERS | SIMILAR_AGE)

KIN_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in KIN_WORDS) + r")\b",
                         re.IGNORECASE)

TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")

# ------------------------------------------------------------
def kin_category(tok):
    """Return kin1/kin2/kin3."""
    t = tok.lower()
    if t in ELDERS:
        return "kin1"
    elif t in YOUNGERS:
        return "kin2"
    else:
        return "kin3"


# ------------------------------------------------------------
def replace_kin(tok, mode):
    """Return replacement token based on mode."""
    if mode == "token":
        return "kin"

    elif mode == "tag":
        return "<KIN>"

    elif mode == "categorical":
        return kin_category(tok)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
def is_kin_token(tok):
    return bool(KIN_PATTERN.fullmatch(tok))


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out_tokens = []

    for tok in tokens:
        if is_kin_token(tok):
            stats["total"] += 1
            out_tokens.append(replace_kin(tok, mode))
        else:
            out_tokens.append(tok)

    out = " ".join(out_tokens)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out


# ------------------------------------------------------------
def count_total_lines(in_dir):
    total = 0
    for fname in os.listdir(in_dir):
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total


# ------------------------------------------------------------
def process_all_files(in_dir, out_dir, mode):
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
                fout.write(transform_line(line.rstrip("\n"), mode, stats) + "\n")
                pbar.update(1)

    pbar.close()

    print("\nKinLang construction complete.\n")
    print(f"TOTAL KINSHIP TERMS REPLACED: {stats['total']:,}\n")


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="token",
                        choices=["token", "tag", "categorical"],
                        help="Replacement mode")

    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"],
                        help="Dataset split")

    args = parser.parse_args()

    # ---------- I/O routing ----------
    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"
    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/kin_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
