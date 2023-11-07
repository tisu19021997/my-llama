from data import load_dataset
from prompts.alpaca import ALPACA_PROMPT_DICT


def format_to_alpaca_prompt(example, training=False):
    if example['input'] != '':
        prompt = ALPACA_PROMPT_DICT['prompt_input'].format_map(example)
    else:
        prompt = ALPACA_PROMPT_DICT['prompt_no_input'].format_map(example)
    if training:
        return prompt + example['output']
    return prompt


def load_alpaca_dataset(config):
    # Load dataset.
    split = {
        'train': config.train_split,
        'eval': config.val_split
    }
    data_files = {
        'train': str(config.train_dataset),
        'eval': str(config.val_dataset)
    }
    data = load_dataset(
        "parquet",
        data_files=data_files,
        split=split
    )
    train_data = data["train"].map(
        lambda example: format_to_alpaca_prompt(example, config.training), num_proc=-1).shuffle()
    val_data = data["eval"].map(
        lambda example: format_to_alpaca_prompt(example, config.training), num_proc=-1)
    return train_data, val_data
