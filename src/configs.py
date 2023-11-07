from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)
from dataclasses import dataclass, field
from typing import List


@dataclass
class train_config:
    model_name: str = "meta-llama/llama-2-7b-hf"
    batch_size_train: int = 1
    batch_size_eval: int = 1
    dataset_type:  str = "alpaca"
    train_dataset: str = ""
    val_dataset: str = ""
    train_split: str = "train[:50%]"
    val_split: str = "eval"
    optim: str = "paged_adamw_8bit"
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4
    max_steps: int = 1400
    lr_scheduler_type: str = "cosine"
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    val_batch_size: int = 1
    output_dir: str = "/output"
    quantization: bool = True
    use_cache: bool = False
    use_flash_attention: bool = False
    logging_steps: int = 5
    num_train_epochs: int = 1
    eval_steps: float = 0.25
    eval_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"  # steps, epoch
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: bool = False
    fp16: bool = True
    bf16: bool = False
    report_to: str = "wandb"
    max_length: int = 512
    val_set_ratio: float = 0.1
    training: bool = True


@dataclass
class inference_config:
    model_name: str = "meta-llama/llama-2-7b-hf"
    ft_model: str = "tisu1902/vie-stem-alpaca-7b"
    peft_model: str = ""
    tokenize_model: str = ""
    quantization: bool = True
    use_flash_attention: bool = False
    device_map: str = "auto"
    debug: bool = False
    training: bool = False

    dataset_type:  str = "alpaca"
    train_dataset: str = ""
    val_dataset: str = ""
    train_split: str = "train[:50%]"
    val_split: str = "eval"
    batch_size: int = 1
        
    max_length: int = 2048
        
    max_new_tokens: int = 512
    top_p: float = 0.9
    temperature: float = 0.001
    repetition_penalty: float = 1.2
    response_phrase: str = "### Response:\n"

@dataclass
class tokenizer_config:
    model_name: str = "meta-llama/llama-2-7b-hf"
    max_length: int = 2048
    pad_token: str = ""
    padding_side: str = "left"


@dataclass
class quantization_config:
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: bool = "float16"
    bnb_4bit_quant_type: bool = "fp4"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class lora_config:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False
