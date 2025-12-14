#!/usr/bin/env python3
# ------------------------------------------------------------
# SFT using LoRA + TRL on Alpaca dataset (local model + local save)
# ------------------------------------------------------------

import os
import torch
import wandb
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ================================
# ARGPARSE
# ================================
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT on Alpaca")

    parser.add_argument(
        "--modelname",
        type=str,
        required=True,
        help="Name of the model folder under /scratch/craj/langsense/models/"
    )

    return parser.parse_args()


args = parse_args()


# ================================
# W&B
# ================================
wandb.init(project="cs795", group="sft_alpaca")


# ================================
# PATHS
# ================================
BASE_MODEL_DIR = "/scratch/craj/langsense/models"
FINETUNED_DIR = "/scratch/craj/langsense/models/finetuned"
os.makedirs(FINETUNED_DIR, exist_ok=True)

MODEL_NAME = args.modelname
LOCAL_MODEL_PATH = f"{BASE_MODEL_DIR}/{MODEL_NAME}"
OUTPUT_DIR = f"{FINETUNED_DIR}/{MODEL_NAME}-alpaca-sft-lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("============================================")
print(f" Loading model from: {LOCAL_MODEL_PATH}")
print(f" Saving finetuned model to: {OUTPUT_DIR}")
print("============================================")


# ================================
# LOAD ALPACA AS PLAIN TEXT
# ================================
def load_alpaca_dataset():
    raw = load_dataset("yahma/alpaca-cleaned")

    rows = []
    for row in raw["train"]:
        instruction = row["instruction"].strip()
        output = row["output"].strip()
        context = row.get("input", "").strip()

        if context:
            user_msg = instruction + "\n\n" + context
        else:
            user_msg = instruction

        text = (
            f"User: {user_msg}\n"
            f"Assistant: {output}"
        )

        rows.append({"text": text})

    return Dataset.from_list(rows)


train_dataset = load_alpaca_dataset()
print("Loaded Alpaca samples:", len(train_dataset))


# ================================
# LOAD MODEL + TOKENIZER
# ================================
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    fix_mistral_regex=True
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map="auto",
    dtype="auto",
    attn_implementation="flash_attention_2",
)

print("Model + tokenizer loaded from local directory.")


# ================================
# LORA CONFIG
# ================================
lora_config = LoraConfig(
    r=256,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj",
    ],
)


# ================================
# TRAINING CONFIG
# ================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    optim="adamw_bnb_8bit",
    logging_steps=10,
    save_steps=500,
    report_to="wandb",
    use_liger_kernel=True,

    # IMPORTANT: use plain-text field
    dataset_text_field="text",
)


# ================================
# TRAINER
# ================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=lora_config,
)

gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU = {gpu_stats.name}, total = {gpu_stats.total_memory/1e9:.2f} GB")


# ================================
# TRAIN
# ================================
print("\nStarting training...\n")
trainer_stats = trainer.train()
print("\nFINISHED TRAINING\n")
print(trainer_stats)


# ================================
# SAVE MODEL LOCALLY
# ================================
trainer.save_model(OUTPUT_DIR)
wandb.finish()

print("\nModel saved to:", OUTPUT_DIR)
