from dataclasses import dataclass, field
from typing import List


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
    padding_side: str = field(
        default="right", metadata={"help": "Padding side: 'left' or 'right'"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Floating-point format: 'float16', 'bfloat16', or 'float32'"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"},
    )


@dataclass
class DataArguments:
    prompt_file: str = field(
        default="grpo_prompts.parquet",
        metadata={"help": "Path to the preprocessed prompts parquet file."},
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length"}
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
    lora_r: int = field(default=128, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "down_proj",
            "o_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "v_proj",
        ],
        metadata={"help": "List of modules to apply LoRA to"},
    )
