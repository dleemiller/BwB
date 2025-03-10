from dataclasses import dataclass
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelConfig:
    model_name: str
    attn_implementation: str
    padding_side: str

# Define configurations for each model
gemma2_config = ModelConfig(
    model_name="google/gemma-2-2b",
    attn_implementation="eager",
    padding_side="right"  # Gemma2 can use right padding
)

qwen25_config = ModelConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    attn_implementation="flash_attention_2",
    padding_side="left"  # Qwen2.5 with Flash Attention requires left padding
)

# Choose which configuration to use
selected_config = qwen25_config  # or gemma2_config, if you want to run Gemma2

# 1. Load your dataset and verify required columns
dataset = load_dataset("dleemiller/lm25", "sft")
required_columns = ["query", "initial_results", "thinking", "augmented_query"]
missing_columns = [col for col in required_columns if col not in dataset["train"].column_names]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Create a train/validation split if necessary
if "validation" not in dataset:
    splits = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = {"train": splits["train"], "validation": splits["test"]}
else:
    dataset = {"train": dataset["train"], "validation": dataset["validation"]}

# 2. Load the model and tokenizer with model-specific settings
tokenizer = AutoTokenizer.from_pretrained(selected_config.model_name)
# Immediately set the padding side before any tokenization occurs!
#tokenizer.padding_side = selected_config.padding_side
#tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    selected_config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=selected_config.attn_implementation,
    trust_remote_code=True
)
model.config.eos_token_id = tokenizer.eos_token_id

# 3. Set up the LoRA configuration with increased rank r=64
peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. Define training arguments using SFTConfig with adjusted hyperparameters
training_args = SFTConfig(
    output_dir="./query_augmentation_model",
    max_seq_length=2048,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    #optim="adamw_torch",
    optim="paged_adamw_8bit",
    logging_steps=5,
    #eval_strategy="steps",
    #eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=3,
    learning_rate=8e-5,
    weight_decay=0.01,
    #fp16=True,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    #metric_for_best_model="eval_loss",
    #load_best_model_at_end=True,
)

# 5. Define your custom formatting function to prepare examples
def formatting_func(example):
    output_texts = []
    for i in range(len(example["query"])):
        input_text = (
            f"Original Search Query: {example['query'][i]}\n\n"
            f"Initial Results Snippets:\n{example['initial_results'][i]}\n\n"
            "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
        )
        # Append the eos token to the augmented query
        output_text = (
            f"<thinking>{example['thinking'][i]}</thinking>\n\n"
            "Here is the augmented query, I hope it will provide better results:\n\n"
            f"### {example['augmented_query'][i]} {tokenizer.eos_token}"
        )
        full_text = f"{input_text}\n\n{output_text}"
        output_texts.append(full_text)
    return output_texts

# 6. Initialize the SFTTrainer
model = prepare_model_for_kbit_training(model)
#model = get_peft_model(model, peft_config)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    #eval_dataset=dataset["validation"],
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_func,
)

# 7. Train the model
trainer.processing_class.tokenizer = tokenizer
trainer.train()

# 8. Save the fine-tuned model and adapter
trainer.save_model("./fine_tuned_model")
trainer.model.save_pretrained("./query_augmentation_adapter")

