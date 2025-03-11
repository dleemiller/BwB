#!/usr/bin/env python
"""
GRPO Training Script Using Preprocessed Prompts and a Precomputed BM25S Index

This script loads a fine-tuned SFT model (with an existing PEFT adapter)
and continues training using GRPO.
"""

import os
import logging
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import pandas as pd
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, PeftConfig, get_peft_model

# Import our evaluator class which internally handles BM25 index creation/loading.
from bm25_reward import BM25Reward

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration Dataclasses
# -------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name: str = field(
        default="./fine_tuned_model",
        metadata={"help": "Path to the fine-tuned SFT model (with PEFT adapter)"},
    )
    adapter_dir: str = field(default="./fine_tuned_model")
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation: 'eager' or 'flash_attention_2'"},
    )
    padding_side: str = field(default="right", metadata={"help": "Padding side: 'left' or 'right'"})
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Floating-point format: 'float16', 'bfloat16', or 'float32'"},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Whether to trust remote code when loading the model"}
    )

@dataclass
class DataArguments:
    prompt_file: str = field(
        default="grpo_prompts.parquet",
        metadata={"help": "Path to the preprocessed prompts parquet file."},
    )
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})

@dataclass
class LoraArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT for training"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to use 4-bit quantization"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to use 8-bit quantization"})
    lora_r: int = field(default=128, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
        metadata={"help": "List of modules to apply LoRA to"},
    )

# -------------------------------------------------------------------
# Custom Reward Function for GRPO
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Custom Reward Function for GRPO
# -------------------------------------------------------------------
def custom_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Compute rewards for generated completions based on multiple criteria:
      1) Presence of a <thinking> block (max = 1.0)
      2) Conciseness of the <thinking> content (max = 1.0)
      3) Presence of an augmented query (text enclosed in <augmented_query> tags) (max = 1.0)
      4) Retrieval improvement as computed by our evaluator's compute_reward.

    This function accepts completions as a positional argument and all other inputs (like prompts,
    query_ids, etc.) via kwargs.

    Expects kwargs to contain:
      - "prompts": a list of original query strings.
      - "query_ids": (optional) a list of query IDs corresponding to each prompt.
    """
    # Extract original prompts from kwargs (default to empty list if not provided)
    prompts = kwargs.get("prompts", [])
    token_threshold = 256
    rewards = []
    #for p, c in zip(prompts, completions):
    #    print(f"{p}\n\n------->\n\n{c}\nEND")

    for i, completion in enumerate(completions):
        # Get the corresponding original query, if available
        orig_query = prompts[i] if i < len(prompts) else ""

        # 1. Evaluate the presence of a <thinking> block.
        if "<thinking>" in completion and "</thinking>" in completion:
            reward_thinking = 1.0
            start = completion.find("<thinking>")
            end = completion.find("</thinking>")
            thinking_text = completion[start + len("<thinking>"):end].strip()
            token_count = len(thinking_text.split())
            # 2. Reward conciseness: full reward if token_count <= threshold, otherwise scale down.
            reward_thinking_length = 1.0 if token_count <= token_threshold else (token_threshold / token_count)
        else:
            reward_thinking = 0.0
            reward_thinking_length = 0.0

        # 3. Check for an augmented query using <augmented_query> tags.
        if "<augmented_query>" in completion and "</augmented_query>" in completion:
            start_idx = completion.find("<augmented_query>")
            end_idx = completion.find("</augmented_query>")
            augmented_part = completion[start_idx + len("<augmented_query>"):end_idx].strip()
            reward_augmented = 1.0 if augmented_part else 0.0
            augmented_query = augmented_part
        else:
            reward_augmented = 0.0
            augmented_query = orig_query

        # 4. Compute retrieval reward using the evaluator’s compute_reward (if available).
        retrieval_reward = 0.0
        if custom_reward_fn.evaluator is not None and "query_ids" in kwargs:
            query_ids = kwargs["query_ids"]
            qid = query_ids[i] if i < len(query_ids) else None
            print(augmented_query, qid, query_ids)
            if qid is not None:
                retrieval_reward = custom_reward_fn.evaluator.compute_reward(
                    query_id=qid,
                    augmented_query=augmented_query,
                    k_value=1000,
                    binary_bonus=1.0,
                    delta_weight=2.0,
                )

        total_reward = reward_thinking + reward_thinking_length + reward_augmented + retrieval_reward

        # Print each component for debugging
        print(f"Example {i}:")
        print(f"  reward_thinking         = {reward_thinking}")
        print(f"  reward_thinking_length  = {reward_thinking_length}")
        print(f"  reward_augmented        = {reward_augmented}")
        print(f"  retrieval_reward        = {retrieval_reward}")
        print(f"  total_reward            = {total_reward}\n")

        rewards.append(total_reward)
    return rewards

# Attach the evaluator instance externally once it's loaded.
custom_reward_fn.evaluator = None  # e.g., set this later to your SimpleBM25Evaluator instance.

def print_trainable_parameters_custom(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params/total_params:.2f}%)")

# -------------------------------------------------------------------
# Main GRPO Training Function
# -------------------------------------------------------------------
def main():
    # Load configuration from YAML.
    config_path = "config_grpo.yaml"
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize dataclasses.
    model_args = ModelArguments(**config.get("model_args", {}))
    data_args = DataArguments(**config.get("data_args", {}))
    lora_args = LoraArguments(**config.get("lora_args", {}))
    training_config = config.get("training_args", {})
    grpo_config = GRPOConfig(**training_config)

    # ---------------------------
    # Load Preprocessed Prompts
    # ---------------------------
    prompt_file = data_args.prompt_file
    logger.info(f"Loading preprocessed prompts from {prompt_file}")
    df = pd.read_parquet(prompt_file)
    train_dataset = Dataset.from_pandas(df)

    # ---------------------------
    # Initialize BM25 Evaluator (loads BM25 index internally)
    # ---------------------------
    logger.info("Initializing BM25 evaluator")
    bm25_evaluator = BM25Reward(dataset_name="msmarco", split="dev")
    custom_reward_fn.evaluator = bm25_evaluator

    # ---------------------------
    # Load Model and Tokenizer
    # ---------------------------
    logger.info(f"Loading tokenizer for {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    #tokenizer.padding_side = model_args.padding_side

    torch_dtype = getattr(torch, model_args.torch_dtype)
    logger.info(f"Loading model with dtype {model_args.torch_dtype}")
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "device_map": "auto",
        "attn_implementation": model_args.attn_implementation,
    }
    if lora_args.load_in_4bit:
        logger.info("Loading model in 4-bit quantization mode")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_config
    elif lora_args.load_in_8bit:
        logger.info("Loading model in 8-bit quantization mode")
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_args.adapter_dir, **model_kwargs)
    #model.config.eos_token_id = tokenizer.eos_token_id

    # ---------------------------
    # PEFT Setup: Continue Training on Existing Adapter
    # ---------------------------
    if lora_args.use_peft:
        pass
        ## Use the adapter directory (here we assume it's stored in model_args.model_name)
        #adaptor_dir = model_args.adapter_dir
        #logger.info(f"Loading trainable PEFT adapter from {adaptor_dir}")
        #
        ## Load the adapter configuration (optional; can be used for diagnostics)
        #peft_config = LoraConfig.from_pretrained(adaptor_dir)
        #
        ## Wrap the base model with the adapter, ensuring it is trainable.
        ## This will load the adapter weights and mark them for training.
        ##model = PeftModel.from_pretrained(
        ##    model, 
        ##    adaptor_dir,
        ##    is_trainable=True
        ##)
        #
        ## Optionally, print out the trainable parameters to verify.
        #def print_trainable_parameters_custom(model):
        #    total_params = sum(p.numel() for p in model.parameters())
        #    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params/total_params:.2f}%)")
        #
        #print_trainable_parameters_custom(model)
    else:
        peft_config = None

    # Prepare model for PEFT if requested
    peft_config = None
    if lora_args.use_peft:
        logger.info("Preparing model for PEFT with LoRA")
        if lora_args.load_in_4bit or lora_args.load_in_8bit:
            logger.info("Preparing quantized model for k-bit training")
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_args.lora_target_modules,
            # modules_to_save=lora_args.lora_modules_to_save
        )
        logger.info(
            f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, dropout={lora_args.lora_dropout}"
        )

        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()

    ## Define a sample prompt that mimics your training format (but uses different content)
    #query = "Ancient Roman architecture"
    #initial_results = (
    #    "### Text snippet from document at k=1\n"
    #    ": The Roman Forum remains as a witness to the political and social life of ancient Rome.\n\n"
    #    "### Text snippet from document at k=3\n"
    #    ": The Colosseum is celebrated for its grand design and remarkable engineering techniques.\n\n"
    #    "### Text snippet from document at k=5\n"
    #    ": Innovations such as arches and concrete construction revolutionized Roman building practices."
    #)
    #
    #instruction = (
    #    f"Original Search Query: {query}\n\n"
    #    f"Initial Results Snippets:\n{initial_results}\n\n"
    #    "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
    #)
    #prompt = (
    #   f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    #   f"{tokenizer.bos_token}<start_of_turn>user\n"
    #   f"{instruction}<end_of_turn>\n"
    #   f"<start_of_turn>model\n"
    #)


    #inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    #inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9)
    #generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    #print(generated_text)
    #exit()


    # ---------------------------
    # Initialize GRPO Trainer
    # ---------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=custom_reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training")
    trainer.train()

    # ---------------------------
    # Save Model and Adapter
    # ---------------------------
    logger.info(f"Saving model to {grpo_config.output_dir}")
    trainer.save_model(grpo_config.output_dir)
    if lora_args.use_peft:
        adapter_path = os.path.join(grpo_config.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_path)
        logger.info(f"Saved PEFT adapter to {adapter_path}")

    if grpo_config.push_to_hub:
        trainer.push_to_hub()
        logger.info("Model pushed to hub")


if __name__ == "__main__":
    main()

