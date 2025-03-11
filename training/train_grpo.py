#!/usr/bin/env python
"""
GRPO Training Script using Preprocessed Prompts and a Precomputed BM25S Index.
This script loads a fine-tuned SFT model (with a PEFT adapter) and continues training using GRPO.
It integrates multiple reward functions and logs training metrics via TensorBoard.
"""

import os
import logging
import yaml
from functools import partial

import torch
import pandas as pd
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations import TensorBoardCallback
from trl import GRPOTrainer, GRPOConfig
from peft import prepare_model_for_kbit_training, get_peft_model

# Import dataclasses from our config module
from grpo.config import ModelArguments, DataArguments, LoraArguments

# Import BM25 evaluator and reward functions.
from grpo.bm25_reward import BM25Reward
from grpo.reward_functions import (
    reward_fn_thinking_presence,
    reward_fn_thinking_conciseness,
    reward_fn_augmented_query,
    reward_fn_retrieval,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('beir.retrieval.evaluation').setLevel(logging.WARNING)


def load_configuration(config_path: str = "grpo/config.yaml"):
    """Load configuration from a YAML file and initialize the dataclasses."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_args = ModelArguments(**config.get("model_args", {}))
    data_args = DataArguments(**config.get("data_args", {}))
    lora_args = LoraArguments(**config.get("lora_args", {}))
    training_config = config.get("training_args", {})
    grpo_config = GRPOConfig(**training_config)
    return model_args, data_args, lora_args, grpo_config


def load_dataset_from_file(prompt_file: str) -> Dataset:
    """Load the preprocessed prompts from a parquet file."""
    logger.info(f"Loading preprocessed prompts from {prompt_file}")
    df = pd.read_parquet(prompt_file)
    return Dataset.from_pandas(df)


def load_model_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
    """Load the tokenizer and model, handling quantization options."""
    logger.info(f"Loading tokenizer for {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

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
    return model, tokenizer


def prepare_model_for_peft(model, lora_args: LoraArguments):
    """Wrap the model for PEFT training if enabled."""
    peft_config = None
    if lora_args.use_peft:
        logger.info("Preparing model for PEFT with LoRA")
        if lora_args.load_in_4bit or lora_args.load_in_8bit:
            logger.info("Preparing quantized model for k-bit training")
            model = prepare_model_for_kbit_training(model)
        from peft import LoraConfig  # Import here to avoid circular dependencies.

        peft_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_args.lora_target_modules,
        )
        logger.info(
            f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, dropout={lora_args.lora_dropout}"
        )
        model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    return model, peft_config


def define_reward_functions(tokenizer, bm25_evaluator):
    """Define the reward functions to be used during GRPO training, ensuring each function has a __name__ attribute."""
    functions = [
        reward_fn_thinking_presence,
        partial(reward_fn_thinking_conciseness, tokenizer=tokenizer, token_threshold=256),
        reward_fn_augmented_query,
        partial(reward_fn_retrieval, evaluator=bm25_evaluator),
    ]
    # Monkey-patch partials to have a __name__ attribute
    for func in functions:
        if isinstance(func, partial):
            func.__name__ = func.func.__name__
    return functions


def initialize_trainer(
    model, tokenizer, train_dataset, grpo_config, peft_config, reward_functions
):
    """Initialize the GRPO trainer with TensorBoard logging."""
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[TensorBoardCallback()],
    )
    return trainer


def main():
    # 1. Load configuration and arguments.
    model_args, data_args, lora_args, grpo_config = load_configuration()

    # 2. Load dataset.
    train_dataset = load_dataset_from_file(data_args.prompt_file)

    # 3. Initialize BM25 evaluator.
    logger.info("Initializing BM25 evaluator")
    bm25_evaluator = BM25Reward(dataset_name="msmarco", split="dev")

    # 4. Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args)

    # 5. Prepare model for PEFT if requested.
    model, peft_config = prepare_model_for_peft(model, lora_args)

    # 6. Define reward functions.
    reward_functions = define_reward_functions(tokenizer, bm25_evaluator)

    # 7. Initialize GRPO trainer with TensorBoard logging.
    trainer = initialize_trainer(
        model, tokenizer, train_dataset, grpo_config, peft_config, reward_functions
    )

    # 8. Train and save the model.
    logger.info("Starting GRPO training")
    trainer.train()

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
