import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import (
    LlamaForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)

from dataclasses import asdict

def load_model_and_tokenizer_for_training(
        train_config,
        quantization_config=None,
        lora_config=None
):
    model_name = train_config.model_name
    device_map = None if train_config.use_flash_attention else 'auto'
    if not train_config.quantization:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        return model
    quantization_config = BitsAndBytesConfig(**asdict(quantization_config()))
    lora_config = LoraConfig(**asdict(lora_config()))
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_eos_token=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    return model, tokenizer


def load_model_and_tokenizer_for_inference(
        inference_config,
        quantization_config=None,
):
    model_name = inference_config.model_name
    device_map = None if inference_config.use_flash_attention else "auto"
    if not inference_config.quantization:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        return model

    quantization_config = BitsAndBytesConfig(**asdict(quantization_config()))
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    if inference_config.peft_model:
        model = PeftModel.from_pretrained(model, inference_config.peft_model)
    model.eval()

    # Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    return model, tokenizer


def load_base_model_and_tokenizer_for_inference(model_name, load_in_4bit=False, load_in_8bit=False):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map="auto")
    model.config.pretraining_tp = 1
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_eos_token=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    return model, tokenizer
