import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# TEMPORAL WORD LISTS
# ------------------------------------------------------------

DAY_WORDS = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "mon", "tue", "tues", "wed", "thu",
    "thur", "thurs", "fri", "sat", "sun"
]

MONTH_WORDS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november",
    "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec"
]

SEASON_WORDS = [
    "spring", "summer", "autumn", "fall", "winter"
]

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

TEMP_WORDS = DAY_WORDS + MONTH_WORDS + SEASON_WORDS
TEMP_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in TEMP_WORDS) + r")\b",
    re.IGNORECASE
)

TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# CATEGORY LOGIC
# ------------------------------------------------------------

def temporal_category(tok):
    t = tok.lower()

    if YEAR_PATTERN.fullmatch(tok):
        return "year"
    if t in DAY_WORDS:
        return "day"
    if t in MONTH_WORDS:
        return "month"
    if t in SEASON_WORDS:
        return "season"
    return "time"  # fallback


def is_temporal(tok):
    return YEAR_PATTERN.fullmatch(tok) or TEMP_PATTERN.fullmatch(tok)


# ------------------------------------------------------------
# REPLACEMENT LOGIC
# ------------------------------------------------------------

def replace_temp(tok, mode):
    if mode == "token":
        return "time"
    elif mode == "tag":
        return "<TIME>"
    elif mode == "categorical":
        return temporal_category(tok)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out = []

    for tok in tokens:
        if is_temporal(tok):
            stats["total"] += 1
            out.append(replace_temp(tok, mode))
        else:
            out.append(tok)

    out_line = " ".join(out)
    out_line = re.sub(r"\s+([.,!?;:])", r"\1", out_line)
    return out_line


# ------------------------------------------------------------
def count_total_lines(in_dir):
    total = 0
    for fname in os.listdir(in_dir):
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
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

    print("\nTempLang construction complete.\n")
    print(f"TOTAL TEMPORAL TERMS REPLACED: {stats['total']:,}\n")


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="token",
                        choices=["token", "tag", "categorical"])
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"])

    args = parser.parse_args()

    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"
    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/temp_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
