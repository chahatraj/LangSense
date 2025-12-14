from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "color_cat"

BASE_MODEL = f"/scratch/craj/langsense/models/{MODEL_NAME}"
sft_adapter_path = f"/scratch/craj/langsense/models/finetuned/{MODEL_NAME}-alpaca-sft-lora"
save_dir = f"/scratch/craj/langsense/models/merged/{MODEL_NAME}-alpaca-merged"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, fix_mistral_regex=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype="auto",
    device_map="auto",
)

model_to_merge = PeftModel.from_pretrained(base_model, sft_adapter_path)
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
