import os
import re
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# Emotion Categories for Categorical Mode
# ------------------------------------------------------------
POSITIVE = {
    "happy", "joy", "joyful", "glad", "delighted", "excited",
    "thrilled", "relieved", "content", "satisfied", "cheerful",
    "optimistic", "happiness"
}

NEGATIVE = {
    "sad", "upset", "unhappy", "miserable", "depressed",
    "heartbroken", "disappointed", "hurt",
    "angry", "mad", "furious", "irritated", "annoyed", "outraged",
    "disgusted", "grossed", "repulsed",
    "fear", "sadness"
}

NEUTRAL = {
    "calm", "relaxed", "uneasy", "tense", "stressed",
    "anxious", "worried", "nervous", "panicked", "petrified",
    "surprised", "shocked", "astonished", "amazed",
    "anxiety"
}

CATEGORY_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral"
}

# Full list
ALL_EMO_WORDS = sorted(POSITIVE | NEGATIVE | NEUTRAL)

# Reverse lookup
CATEGORY_OF = {}
for w in POSITIVE: CATEGORY_OF[w] = "positive"
for w in NEGATIVE: CATEGORY_OF[w] = "negative"
for w in NEUTRAL:  CATEGORY_OF[w] = "neutral"

# Pattern
EMO_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in ALL_EMO_WORDS) + r")\b",
    re.IGNORECASE
)

TOKENIZER = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ------------------------------------------------------------
# Replacement Logic
# ------------------------------------------------------------
def replace_emo(tok, mode):
    tok_l = tok.lower()

    if mode == "token":
        return "emotion"

    elif mode == "tag":
        return "<EMOTION>"

    elif mode == "categorical":
        cat = CATEGORY_OF[tok_l]            # positive/negative/neutral
        return CATEGORY_MAP[cat]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def is_emo_token(tok):
    return bool(EMO_PATTERN.fullmatch(tok))


# ------------------------------------------------------------
def transform_line(line, mode, stats):
    if not line.strip():
        return line

    tokens = TOKENIZER.findall(line)
    out_tokens = []

    for tok in tokens:
        if is_emo_token(tok):
            stats["total"] += 1
            out_tokens.append(replace_emo(tok, mode))
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

    print("\nEmotionLang construction complete.")
    print(f"TOTAL EMOTION TERMS REPLACED: {stats['total']:,}\n")


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="token",
                        choices=["token", "tag", "categorical"],
                        help="Replacement mode")

    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"],
                        help="Which split to process")

    args = parser.parse_args()

    INPUT_DIR = f"/scratch/craj/langsense/data/baby_lm/{args.split}_100M"
    OUTPUT_DIR = (
        f"/scratch/craj/langsense/data/new_langs/emotion_lang/"
        f"{args.mode}/{args.split}"
    )

    print(f"\nMode:  {args.mode}")
    print(f"Split: {args.split}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    process_all_files(INPUT_DIR, OUTPUT_DIR, args.mode)


if __name__ == "__main__":
    main()
