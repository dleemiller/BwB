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
from peft import LoraConfig, prepare_model_for_kbit_training

# BM25S imports
import bm25s
from bm25s.tokenization import Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration Dataclasses
# -------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name: str = field(
        default="fine_tuned_model/",
        metadata={"help": "Path to the fine-tuned SFT model (with PEFT adapter)"},
    )
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
# BM25S Retrieval Utility
# -------------------------------------------------------------------
def compute_retrieval_score_bm25s(query: str, bm25s_retriever, bm25s_tokenizer) -> float:
    """
    Compute the BM25S retrieval score for a given query.
    Returns the top score from the retrieved results.
    """
    tokenized_query = bm25s_tokenizer.tokenize([query], update_vocab=False)
    results, scores = bm25s_retriever.retrieve(tokenized_query, k=1, backend_selection="numba")
    return float(scores[0, 0]) if scores.size > 0 else 0.0

# -------------------------------------------------------------------
# Custom Reward Function for GRPO
# -------------------------------------------------------------------
def custom_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Compute rewards for generated completions based on:
      1) Presence of a <thinking> block (max = 1.0)
      2) Conciseness of the <thinking> content (max = 1.0)
      3) Presence of an augmented query (text after "###") (max = 1.0)
      4) Retrieval improvement: 3.0 * normalized difference in BM25 scores.
    
    Expects kwargs to contain "original_query" (list of strings).
    """
    token_threshold = 50
    # This normalization factor adjusts the BM25 difference to a smaller scale.
    bm25_norm_factor = 100.0  
    rewards = []
    original_queries = kwargs.get("original_query", prompts)
    print(prompts, completions, kwargs)
    
    for orig_query, completion in zip(original_queries, completions):
        # 1. Check for <thinking> block.
        if "<thinking>" in completion and "</thinking>" in completion:
            reward_thinking = 1.0
            start = completion.find("<thinking>")
            end = completion.find("</thinking>")
            thinking_text = completion[start + len("<thinking>"):end].strip()
            token_count = len(thinking_text.split())
            # 2. Reward for conciseness.
            reward_thinking_length = 1.0 if token_count <= token_threshold else (token_threshold / token_count)
        else:
            reward_thinking = 0.0
            reward_thinking_length = 0.0
       
        # 3. Check for augmented query.
        if "###" in completion:
            augmented_part = completion.split("###")[-1].strip()
            reward_augmented = 1.0 if augmented_part else 0.0
            augmented_query = augmented_part
        else:
            reward_augmented = 0.0
            augmented_query = orig_query
        
        # 4. Compute retrieval reward using normalized BM25S scores.
        score_orig = compute_retrieval_score_bm25s(orig_query, custom_reward_fn.bm25s_retriever, custom_reward_fn.bm25s_tokenizer)
        score_aug = compute_retrieval_score_bm25s(augmented_query, custom_reward_fn.bm25s_retriever, custom_reward_fn.bm25s_tokenizer)
        retrieval_reward = 3.0 * ((score_aug - score_orig) / bm25_norm_factor)
        print(f"Score original: {score_orig:.2f}, score augmented: {score_aug:.2f}, reward: {retrieval_reward:.2f}, {reward_augmented:.2f}, {reward_thinking_length}")
        
        total_reward = reward_thinking + reward_thinking_length + reward_augmented + retrieval_reward
        rewards.append(total_reward)
    return rewards

# Attach BM25S retriever and tokenizer placeholders to the reward function.
custom_reward_fn.bm25s_retriever = None  # to be set in main()
custom_reward_fn.bm25s_tokenizer = None  # to be set in main()

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
    # Load BM25S Retriever and Tokenizer
    # ---------------------------
    bm25s_index_dir = config.get("bm25s_index_dir", "bm25s_index")
    logger.info(f"Loading BM25S index from {bm25s_index_dir}")
    bm25s_retriever = bm25s.BM25.load(bm25s_index_dir, load_corpus=True)
    bm25s_tokenizer = Tokenizer(splitter=lambda x: x.split())
    bm25s_tokenizer.load_vocab(bm25s_index_dir)
    bm25s_retriever.activate_numba_scorer()
    custom_reward_fn.bm25s_retriever = bm25s_retriever
    custom_reward_fn.bm25s_tokenizer = bm25s_tokenizer

    # ---------------------------
    # Load Model and Tokenizer
    # ---------------------------
    logger.info(f"Loading tokenizer for {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=model_args.trust_remote_code)
    tokenizer.padding_side = model_args.padding_side

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

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, **model_kwargs)
    model.config.eos_token_id = tokenizer.eos_token_id

    # ---------------------------
    # PEFT Setup: Continue Training on Existing Adapter
    # ---------------------------
    if lora_args.use_peft:
        if hasattr(model, "peft_config"):
            logger.info("Model already has a PEFT adapter. Continuing training on the existing adapter.")
            peft_config = None  # Do not re-wrap adapter.
        else:
            logger.info("Preparing model for PEFT with LoRA")
            if lora_args.load_in_4bit or lora_args.load_in_8bit:
                model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                lora_dropout=lora_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_args.lora_target_modules,
            )
            logger.info(f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, dropout={lora_args.lora_dropout}")
    else:
        peft_config = None

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


