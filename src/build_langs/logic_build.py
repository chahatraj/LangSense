import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# LOGIC WORD GROUPS
# ------------------------------------------------------------

CONNECTIVES = {
    "and", "or", "but", "however", "although", "yet",
    "so", "therefore", "thus"
}

CONDITIONALS = {
    "if", "unless"
}

CAUSAL = {
    "because", "since", "due",  # due (standalone)
}

# Multi-word causal phrases
MULTIWORD_CAUSAL = {
    "due to",
    "as a result"
}

# Build combined full list
ALL_LOGIC_SINGLE = sorted(CONNECTIVES | CONDITIONALS | CAUSAL)
ALL_LOGIC_MULTI = sorted(MULTIWORD_CAUSAL)

LOGIC_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in ALL_LOGIC_SINGLE) + r")\b",
    re.IGNORECASE
)

TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# Categorical mapping logic
# ------------------------------------------------------------

def category_for_logic(tok):
    """Return 'connective', 'conditional', or 'causal' based on membership."""
    t = tok.lower()
    if t in CONNECTIVES:
        return "connective"
    elif t in CONDITIONALS:
        return "conditional"
    else:
        return "causal"


def category_for_multiword(phrase):
    """Multi-word terms are always causal."""
    return "causal"


# ------------------------------------------------------------
# Replacement Logic
# ------------------------------------------------------------

def replace_logic(tok, mode, is_multi=False):
    if mode == "token":
        return "logic"

    elif mode == "tag":
        return "<LOGIC>"

    elif mode == "categorical":
        if is_multi:
            return category_for_multiword(tok)
        else:
            return category_for_logic(tok)

    else:
        raise ValueError(f"Unknown replacement mode: {mode}")


# ------------------------------------------------------------
def detect_multiword(line, mode, stats):
    """
    Replace multi-word phrases before token-level replacement.
    """
    lower_line = line.lower()

    for phrase in ALL_LOGIC_MULTI:
        if phrase in lower_line:
            count = len(re.findall(re.escape(phrase), lower_line))
            stats["total"] += count

            repl = replace_logic(phrase, mode, is_multi=True)
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)

            line = pattern.sub(repl, line)
            lower_line = line.lower()

    return line


# ------------------------------------------------------------
def is_logic_single(tok):
    return bool(LOGIC_PATTERN.fullmatch(tok))


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    # First replace multi-word logic phrases
    line = detect_multiword(line, mode, stats)

    # Token-level replacement for single-word logic
    tokens = TOKENIZER.findall(line)
    out_tokens = []

    for tok in tokens:
        if is_logic_single(tok):
            stats["total"] += 1
            out_tokens.append(replace_logic(tok, mode))
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

    print("\nLogicLang construction complete.")
    print(f"TOTAL LOGIC ITEMS REPLACED: {stats['total']:,}\n")


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
        f"/scratch/craj/langsense/data/new_langs/logic_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
