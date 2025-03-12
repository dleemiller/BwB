import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, PeftConfig


model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

fine_tuned_adapter_path = "./query_augmentation_model"

base_and_adapter_model = PeftModel.from_pretrained(model, fine_tuned_adapter_path)
base_and_adapter_model = base_and_adapter_model.merge_and_unload()

tokenizer.save_pretrained("./merged_sft_model")
base_and_adapter_model.save_pretrained("./merged_sft_model")
