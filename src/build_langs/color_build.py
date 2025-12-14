import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# Color Groups (for categorical mode)
# ------------------------------------------------------------
WARM_COLORS = {
    "red", "orange", "yellow", "maroon", "gold", "bronze"
}
COOL_COLORS = {
    "blue", "green", "cyan", "indigo", "navy", "teal",
    "turquoise", "lavender", "violet", "purple"
}
NEUTRAL_COLORS = {
    "black", "white", "gray", "grey", "beige", "ivory", "silver", "brown"
}

# Mixed = anything not in above sets but still a color
MIXED_COLORS = set()

# These four map to categorical replacement
CATEGORY_MAP = {
    "warm": "color1",
    "cool": "color2",
    "neutral": "color3",
    "mixed": "color4",
}

# ------------------------------------------------------------
# Full color list (expands MIXED automatically)
# ------------------------------------------------------------
COLOR_WORDS = sorted({
    "red", "blue", "green", "yellow", "orange", "purple",
    "violet", "indigo", "pink", "brown", "black", "white",
    "gray", "grey", "beige", "ivory", "cyan", "magenta",
    "maroon", "navy", "teal", "turquoise", "lavender",
    "gold", "silver", "bronze",
})

# Identify remaining colors as "mixed"
MIXED_COLORS = (
    set(COLOR_WORDS)
    - WARM_COLORS
    - COOL_COLORS
    - NEUTRAL_COLORS
)

CATEGORY_OF = {}
for c in COLOR_WORDS:
    if c in WARM_COLORS:
        CATEGORY_OF[c] = "warm"
    elif c in COOL_COLORS:
        CATEGORY_OF[c] = "cool"
    elif c in NEUTRAL_COLORS:
        CATEGORY_OF[c] = "neutral"
    else:
        CATEGORY_OF[c] = "mixed"


COLOR_PATTERN = re.compile(r"\b(?:" + "|".join(COLOR_WORDS) + r")\b", re.IGNORECASE)
TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# Replacement Logic
# ------------------------------------------------------------

def replace_color(tok, mode):
    """Return replacement based on mode."""
    tok_l = tok.lower()

    if mode == "token":
        return "color"

    elif mode == "tag":
        return "<COLOR>"

    elif mode == "categorical":
        cat = CATEGORY_OF[tok_l]        # warm/cool/neutral/mixed
        return CATEGORY_MAP[cat]        # color1/2/3/4

    else:
        raise ValueError(f"Unknown mode: {mode}")


def is_color_token(tok):
    return bool(COLOR_PATTERN.fullmatch(tok))


# ------------------------------------------------------------
# Line Transformation
# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out_tokens = []

    for tok in tokens:
        if is_color_token(tok):
            stats["total"] += 1
            out_tokens.append(replace_color(tok, mode))
        else:
            out_tokens.append(tok)

    # Fix spacing before punctuation
    out = " ".join(out_tokens)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out


# ------------------------------------------------------------
# Count Lines (for tqdm)
# ------------------------------------------------------------
def count_total_lines(in_dir):
    total = 0
    for fname in os.listdir(in_dir):
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total


# ------------------------------------------------------------
# Processing
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

    print("\nColorLang construction complete.")
    print(f"TOTAL COLOR ITEMS REPLACED: {stats['total']:,}\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="token",
                        choices=["token", "tag", "categorical"],
                        help="Replacement mode")

    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"],
                        help="Whether to process train or dev set")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Input/Output Routing
    # ------------------------------------------------------------
    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"

    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/color_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()

