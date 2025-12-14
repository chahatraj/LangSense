import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# NUMERIC WORD GROUPS
# ------------------------------------------------------------

NUMBER_WORDS = [
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
    "sixty","seventy","eighty","ninety","hundred","thousand","million",
    "billion","trillion"
]
ORDINAL_WORDS = [
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth",
    "tenth","eleventh","twelfth","thirteenth","fourteenth","fifteenth",
    "sixteenth","seventeenth","eighteenth","nineteenth","twentieth",
    "hundredth","thousandth","millionth"
]

# Regex patterns
DIGIT_PATTERN = re.compile(
    r"""
    (?:\d{1,3}(?:,\d{3})+|\d+)      # integers with/without commas
    (?:\.\d+)?                      # decimals
    (?:[:/.-]\d+)*                  # time/date-like segments
    (?:%)?                          # percent
    """, re.VERBOSE
)

WORD_PATTERN = re.compile(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)

HYPHEN_PATTERN = re.compile(
    r"\b(?:" + "|".join(NUMBER_WORDS[:20]) + r")-(?:" +
    "|".join(NUMBER_WORDS[:20]) + r")\b",
    re.IGNORECASE
)

ORDINAL_PATTERN = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)

ORDINAL_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(ORDINAL_WORDS) + r")\b",
    re.IGNORECASE
)

TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# CATEGORY DETECTION
# ------------------------------------------------------------

def numeric_category(tok):
    """Returns which of the 4 numeric types applies."""
    if DIGIT_PATTERN.fullmatch(tok):
        return "number"
    if HYPHEN_PATTERN.fullmatch(tok):
        return "spelled_number"
    if WORD_PATTERN.fullmatch(tok):
        return "spelled_number"
    if ORDINAL_PATTERN.fullmatch(tok):
        return "ordinal"
    if ORDINAL_WORD_PATTERN.fullmatch(tok):
        return "spelled_ordinal"
    return None


def is_numeric(tok):
    """Unified detection for any numeric token."""
    return numeric_category(tok) is not None


# ------------------------------------------------------------
# Replacement
# ------------------------------------------------------------

def replace_number(tok, mode):
    if mode == "token":
        return "number"

    elif mode == "tag":
        return "<NUMBER>"

    elif mode == "categorical":
        return numeric_category(tok)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out = []

    for tok in tokens:
        if is_numeric(tok):
            stats["total"] += 1
            out.append(replace_number(tok, mode))
        else:
            out.append(tok)

    # Fix spacing before punctuation
    out_line = " ".join(out)
    out_line = re.sub(r"\s+([.,!?;:])", r"\1", out_line)
    return out_line


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

    print("\nNumLang construction complete.\n")
    print(f"TOTAL NUMERIC ITEMS REPLACED: {stats['total']:,}\n")


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="token",
                        choices=["token", "tag", "categorical"],
                        help="Replacement style")

    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"],
                        help="Dataset split")

    args = parser.parse_args()

    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"
    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/num_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
