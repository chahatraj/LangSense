import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# Gendered words
# ------------------------------------------------------------
GENDERED_NONPOSSESSIVE = {
    "he", "him", "man", "boy", "male",
    "she", "her", "woman", "girl", "female",
}

POSSESSIVE = {"his", "hers"}
REFLEXIVE = {"himself", "herself"}

MALE_WORDS = {"he", "him", "man", "boy", "male", "his", "himself"}
FEMALE_WORDS = {"she", "her", "woman", "girl", "female", "hers", "herself"}

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------
REPLACE_COUNTS = {
    w: 0 for w in (
        list(GENDERED_NONPOSSESSIVE)
        + list(POSSESSIVE)
        + list(REFLEXIVE)
    )
}

CONTRACTION_COUNTS = {w: 0 for w in GENDERED_NONPOSSESSIVE.union(POSSESSIVE)}
TOTAL_REPLACED = 0

tokenizer = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# Replacement Logic
# ------------------------------------------------------------
def replace_gender(tok, pid, mode):
    tok_l = tok.lower()

    # Gender category for categorical mode
    if tok_l in MALE_WORDS:
        cat = "A"
    elif tok_l in FEMALE_WORDS:
        cat = "B"
    else:
        cat = "A"   # default fallback

    if mode == "token":
        return f"person{pid}"

    elif mode == "tag":
        return f"<PERSON{pid}>"

    elif mode == "categorical":
        return f"person{cat}{pid}"

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = tokenizer.findall(line)
    output = []
    pid = 1  # Per-line counter: resets each line

    for tok in tokens:
        low = tok.lower()

        # ---------------------------------------------
        # CASE 0: Contractions (he's, she's, boy's)
        # ---------------------------------------------
        if "'" in tok:
            base, apos, rest = tok.partition("'")
            low_base = base.lower()

            if low_base in CONTRACTION_COUNTS:
                CONTRACTION_COUNTS[low_base] += 1
                stats["total"] += 1

                repl = replace_gender(low_base, pid, mode)
                output.append(f"{repl}'{rest}")
                pid += 1
                continue

            output.append(tok)
            continue

        # ---------------------------------------------
        # CASE 1: Possessives (his, hers)
        # ---------------------------------------------
        if low in POSSESSIVE:
            REPLACE_COUNTS[low] += 1
            stats["total"] += 1

            repl = replace_gender(low, pid, mode)
            output.append(f"{repl}'s")
            pid += 1
            continue

        # ---------------------------------------------
        # CASE 2: Reflexives (himself, herself)
        # ---------------------------------------------
        if low in REFLEXIVE:
            REPLACE_COUNTS[low] += 1
            stats["total"] += 1

            repl = replace_gender(low, pid, mode)
            output.append(repl)
            pid += 1
            continue

        # ---------------------------------------------
        # CASE 3: Non-possessive gendered pronouns
        # ---------------------------------------------
        if low in GENDERED_NONPOSSESSIVE:
            REPLACE_COUNTS[low] += 1
            stats["total"] += 1

            repl = replace_gender(low, pid, mode)
            output.append(repl)
            pid += 1
            continue

        # default
        output.append(tok)

    # Remove space before punctuation
    out = " ".join(output)
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

    print("\nGenderLang construction complete.\n")
    print(f"TOTAL WORDS REPLACED: {stats['total']:,}\n")

    print("=== Direct Word Counts ===")
    for w, c in sorted(REPLACE_COUNTS.items()):
        print(f"{w:10s} : {c:,}")

    print("\n=== Contraction Counts ===")
    for w, c in sorted(CONTRACTION_COUNTS.items()):
        print(f"{w:10s} : {c:,}")


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

    # MODE FIRST — then SPLIT (your request)
    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/gender_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
