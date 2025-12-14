import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# SPACE CATEGORY MAPS
# ------------------------------------------------------------

WEST_WORDS = {"left"}
EAST_WORDS = {"right"}
NORTH_WORDS = {"up", "above", "over"}
SOUTH_WORDS = {"down", "below", "under"}

ALL_SPACE_WORDS = sorted(
    WEST_WORDS | EAST_WORDS | NORTH_WORDS | SOUTH_WORDS
)

SPACE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in ALL_SPACE_WORDS) + r")\b",
    re.IGNORECASE,
)

TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# CATEGORY LOGIC
# ------------------------------------------------------------

def space_category(tok):
    t = tok.lower()
    if t in WEST_WORDS:
        return "west"
    elif t in EAST_WORDS:
        return "east"
    elif t in NORTH_WORDS:
        return "north"
    elif t in SOUTH_WORDS:
        return "south"
    else:
        return "north"  # fallback


# ------------------------------------------------------------
# REPLACEMENT LOGIC
# ------------------------------------------------------------

def replace_space(tok, mode):
    """Return replacement based on the selected mode."""
    if mode == "token":
        return "direction"

    elif mode == "tag":
        return "<DIRECTION>"

    elif mode == "categorical":
        return space_category(tok)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
def is_space(tok):
    return bool(SPACE_PATTERN.fullmatch(tok))


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out_tokens = []

    for tok in tokens:
        # Handle contractions like "above'll"
        if "'" in tok:
            base, apos, rest = tok.partition("'")
            base_l = base.lower()

            if is_space(base_l):
                stats["total"] += 1
                repl = replace_space(base_l, mode)
                out_tokens.append(f"{repl}'{rest}")
                continue

            out_tokens.append(tok)
            continue

        # Normal case
        if is_space(tok):
            stats["total"] += 1
            out_tokens.append(replace_space(tok, mode))
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

    print("\nSpaceLang construction complete.\n")
    print(f"TOTAL DIRECTION WORDS REPLACED: {stats['total']:,}\n")


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

    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"

    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/space_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
