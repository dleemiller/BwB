from dataclasses import dataclass, field
from typing import List, Optional


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
