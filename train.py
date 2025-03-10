from dataclasses import dataclass, field
from typing import List, Optional
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training
import os
import logging
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(
        default="google/gemma-2-2b",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention implementation to use: 'eager', 'flash_attention_2', etc."
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side: 'left' or 'right'"}
    )
    use_liger: bool = field(
        default=False,
        metadata={"help": "Whether to use Liger Kernels for optimization"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "Floating-point format to use: 'float16', 'bfloat16', or 'float32'"
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"},
    )


@dataclass
class DataArguments:
    dataset_name: List[str] = field(
        default_factory=lambda: ["dleemiller/lm25"],
        metadata={"help": "The name of the dataset to use."},
    )
    dataset_config_name: List[str] = field(
        default_factory=lambda: ["sft", "sft-concise"],
        metadata={"help": "The configuration name of the dataset to use."},
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for training"}
    )
    packing: bool = field(
        default=True,
        metadata={
            "help": "Whether to pack multiple sequences into one for efficient training"
        },
    )


@dataclass
class LoraArguments:
    use_peft: bool = field(
        default=True, metadata={"help": "Whether to use PEFT for training"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to use 4-bit quantization"}
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Whether to use 8-bit quantization"}
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: str = field(
        default="all-linear", metadata={"help": "Which modules to apply LoRA to"}
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default_factory=lambda: ["lm_head", "embed_tokens"],
        metadata={"help": "List of modules to save alongside LoRA adapters"},
    )


instruction_template = "### Instruction:"
response_template = "### Response:"


def formatting_func(example):
    """Format examples into the chat template format."""
    messages = []
    for i in range(len(example["query"])):
        prompt = (
            f"Original Search Query: {example['query'][i]}\n\n"
            f"Initial Results Snippets:\n{example['initial_results'][i]}\n\n"
            "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
        )
        completion = (
            f"<thinking>{example['thinking'][i]}</thinking>\n\n"
            "Here is the augmented query, I hope it will provide better results:\n\n"
            f"### {example['augmented_query'][i]}"
        )
        messages.append(
            f"{instruction_template}\n{prompt}\n\n{response_template}\n{completion}"
        )
    return messages


def main():
    # Load configuration from YAML file (no CLI arguments assumed)
    config_path = "config.yaml"
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize arguments using YAML configuration
    model_args = ModelArguments(**config.get("model_args", {}))
    data_args = DataArguments(**config.get("data_args", {}))
    lora_args = LoraArguments(**config.get("lora_args", {}))
    training_config = config.get("training_args", {})
    print(config)
    training_args = SFTConfig(**training_config)
    # training_args.max_seq_length = data_args.max_seq_length

    # Enable TF32 if specified
    # if getattr(training_args, "tf32", False):
    #    torch.backends.cuda.matmul.allow_tf32 = True
    #    logger.info("TF32 enabled for matmul operations")

    # Configure Liger Kernels if requested
    # if model_args.use_liger:
    #    try:
    #        import liger.config
    #        logger.info("Liger Kernels enabled for optimization")
    #    except ImportError:
    #        logger.warning("Liger Kernels requested but not installed. Proceeding without Liger.")

    # Load and prepare datasets
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

    # Load tokenizer with model-specific settings
    logger.info(f"Loading tokenizer for {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.padding_side = model_args.padding_side
    # if tokenizer.pad_token is None:
    #    if tokenizer.eos_token is not None:
    #        tokenizer.pad_token = tokenizer.eos_token
    #        logger.info("Setting pad_token to eos_token")
    #    else:
    #        logger.warning("Tokenizer has no pad_token or eos_token!")

    # Setup model loading configuration
    torch_dtype = getattr(torch, model_args.torch_dtype)
    logger.info(f"Loading model with dtype: {model_args.torch_dtype}")
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
        "device_map": "auto",
        # "device_map": "balanced_low_0",
        "attn_implementation": model_args.attn_implementation,
    }
    print(model_kwargs)
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

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, **model_kwargs)
    model.config.eos_token_id = tokenizer.eos_token_id

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

    # Initialize the SFTTrainer with the proper SFTConfig
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        formatting_func=formatting_func if not training_args.packing else None,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        ),
    )
    trainer.processing_class.tokenizer = tokenizer

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if lora_args.use_peft:
        adapter_path = os.path.join(training_args.output_dir, "adapter")
        trainer.model.save_pretrained(adapter_path)
        logger.info(f"Saved LoRA adapter to {adapter_path}")

    if training_args.push_to_hub:
        trainer.push_to_hub()
        logger.info(
            f"Model pushed to {training_args.hub_model_id or os.path.basename(training_args.output_dir)}"
        )


if __name__ == "__main__":
    main()
