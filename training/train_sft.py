#!/usr/bin/env python
"""
SFT Training Script using a preprocessed dataset and LoRA PEFT.
This script loads a pretrained model and fine-tunes it using SFTTrainer.
It also supports dataset packing, quantization, and optional LoRA adapters.
"""

import os
import logging
import yaml
from functools import partial

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DefaultDataCollator,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Import our dataclasses from config module.
from config import ModelArguments, DataArguments, LoraArguments

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gemma_formatting_func(example):
    messages = []
    for i in range(len(example["query"])):
        instruction = (
            f"Original Search Query: {example['query'][i]}\n\n"
            f"Initial Results Snippets:\n{example['initial_results'][i]}\n\n"
            "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
        )
        response = (
            f"<thinking>{example['thinking'][i]}</thinking>\n\n"
            "Here is the augmented query, I hope it will provide better results:\n\n"
            f"<augmented_query>{example['augmented_query'][i]}</augmented_query>"
        )
        message = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            f"{gemma_formatting_func.tokenizer.bos_token}<start_of_turn>user\n"
            f"{instruction}<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{response}<end_of_turn>\n"
            f"{gemma_formatting_func.tokenizer.eos_token}"
        )
        messages.append(message)
    return messages


def load_configuration(config_path: str = "sft/config.yaml"):
    """Load configuration from YAML and initialize argument dataclasses and trainer config."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_args = ModelArguments(**config.get("model_args", {}))
    data_args = DataArguments(**config.get("data_args", {}))
    lora_args = LoraArguments(**config.get("lora_args", {}))
    training_config = config.get("training_args", {})
    sft_config = SFTConfig(**training_config)
    return model_args, data_args, lora_args, sft_config


def load_datasets(data_args: DataArguments):
    """Load and concatenate datasets from the provided names and configurations."""
    logger.info(
        f"Loading datasets: {data_args.dataset_name} with configs {data_args.dataset_config_name}"
    )
    datasets_list = []
    for i, ds_name in enumerate(data_args.dataset_name):
        config_name = (
            data_args.dataset_config_name[i]
            if i < len(data_args.dataset_config_name)
            else None
        )
        dataset = load_dataset(ds_name, config_name)
        datasets_list.append(dataset["train"])
    train_dataset = concatenate_datasets(datasets_list).shuffle(seed=33)
    logger.info(f"Loaded {len(train_dataset)} training examples")
    return train_dataset


def load_tokenizer(model_args: ModelArguments):
    """Load the tokenizer and assign it to the formatting function."""
    logger.info(f"Loading tokenizer for {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name, trust_remote_code=model_args.trust_remote_code
    )
    gemma_formatting_func.tokenizer = tokenizer
    return tokenizer


def load_model_and_prepare(model_args: ModelArguments, lora_args: LoraArguments):
    """Load the model with quantization options and prepare it for PEFT if requested."""
    torch_dtype = getattr(torch, model_args.torch_dtype)
    logger.info(f"Loading model with dtype: {model_args.torch_dtype}")
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "device_map": "auto",
        "attn_implementation": model_args.attn_implementation,
    }
    if lora_args.load_in_4bit:
        logger.info("Loading model in 4-bit quantization mode")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
    elif lora_args.load_in_8bit:
        logger.info("Loading model in 8-bit quantization mode")
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, **model_kwargs, use_cache=False
    )

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
            # Optionally, modules_to_save can be passed if needed.
        )
        logger.info(
            f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, dropout={lora_args.lora_dropout}"
        )
        model = get_peft_model(model, peft_config)
    return model, peft_config


def create_data_collator(tokenizer):
    """Create the data collator for completion-only LM training."""
    response_template = "<start_of_turn>model\n"
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )


def create_trainer(model, train_dataset, sft_config, peft_config, tokenizer):
    """Initialize the SFTTrainer with the given model, dataset, and configurations."""
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=sft_config,
        formatting_func=gemma_formatting_func if not sft_config.packing else None,
        data_collator=create_data_collator(tokenizer)
    )
    trainer.processing_class.tokenizer = tokenizer
    return trainer


def main():
    # 1. Load configuration and initialize argument objects.
    model_args, data_args, lora_args, sft_config = load_configuration()

    # 2. Load and prepare datasets.
    train_dataset = load_datasets(data_args)

    # 3. Load tokenizer.
    tokenizer = load_tokenizer(model_args)

    # 4. Load model and prepare for PEFT if requested.
    model, peft_config = load_model_and_prepare(model_args, lora_args)

    # 5. Initialize trainer.
    trainer = create_trainer(model, train_dataset, sft_config, peft_config, tokenizer)

    # 6. Train and save the model.
    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving model to {sft_config.output_dir}")
    trainer.save_model(sft_config.output_dir)

    if lora_args.use_peft:
        adapter_path = os.path.join(sft_config.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_path)
        logger.info(f"Saved LoRA adapter to {adapter_path}")

    if sft_config.push_to_hub:
        trainer.push_to_hub()
        logger.info(
            f"Model pushed to {sft_config.hub_model_id or os.path.basename(sft_config.output_dir)}"
        )


if __name__ == "__main__":
    main()

