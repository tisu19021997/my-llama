import os
import torch
import wandb
import torch.distributed as dist
import fire

from datetime import datetime
from trl import SFTTrainer
from transformers import TrainingArguments

from utils.config_utils import update_config
from utils.model_utils import load_model_and_tokenizer_for_training
from data.alpaca_dataset import load_alpaca_dataset
from configs import train_config, quantization_config


torch.cuda.manual_seed(42)
torch.manual_seed(42)

os.environ['TOKENIZERS_PARALLELISM'] = "false"
wandb.login(key="5d1dc32c97d754cc8ce0b6666d9f85d9f7618ccd", force=True)


def main(**kwargs):
    update_config(train_config, **kwargs)
    update_config(quantization_config, **kwargs)

    # Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer_for_training(
        train_config, quantization_config)

    # Load dataset.
    if train_config.dataset_type == "alpaca":
        train_data, val_data = load_alpaca_dataset(train_config)

    training_args = TrainingArguments(
        output_dir=str(train_config.output_dir),
        optim=train_config.optim,
        learning_rate=train_config.lr,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_ratio=train_config.warmup_ratio,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.batch_size_train,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        per_device_eval_batch_size=train_config.batch_size_eval,
        evaluation_strategy=train_config.evaluation_strategy,  # steps, epoch
        eval_steps=train_config.eval_steps,
        eval_accumulation_steps=train_config.eval_accumulation_steps,
        save_strategy=train_config.save_strategy,
        load_best_model_at_end=train_config.load_best_model_at_end,
        # metric_for_best_model='rl',
        gradient_checkpointing=train_config.gradient_checkpointing,
        ddp_find_unused_parameters=train_config.ddp_find_unused_parameters,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        logging_steps=train_config.logging_steps,
        # max_steps=train_config.max_steps,
        seed=train_config.seed,
        report_to="wandb",
        run_name=f"openllama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        max_seq_length=train_config.max_length,
        tokenizer=tokenizer,
        args=training_args,
        # compute_metrics = compute_metrics,
    )

    if train_config.use_flash_attention:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            trainer.train()
    else:
        trainer.train()

    # trainer.train()
    trainer.model.save_pretrained(str(train_config.output_dir))
    tokenizer.save_pretrained(str(train_config.output_dir))


if __name__ == '__main__':
    fire.Fire(main)
