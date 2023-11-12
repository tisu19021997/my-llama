from torch.utils.data import Dataset
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def to_alpaca_prompt(example, training=True):
    if example['input'] != '':
        prompt = ALPACA_PROMPT_DICT['prompt_input'].format_map(example)
    else:
        prompt = ALPACA_PROMPT_DICT['prompt_no_input'].format_map(example)
    if training:
        return prompt + example['output']
    return prompt


def to_mcq_prompt(example, training=True):
    instruction = "Choose the correct answer for the question from the choices below. Only output the correct choice, nothing else.\n\n"
    instruction += example['question'] + '\n'
    instruction += '\n'.join(example['choices'])
    prompt = ALPACA_PROMPT_DICT['prompt_no_input'].format(
        instruction=instruction)
    if training:
        prompt += example['answer']
    return prompt


class AlpacaDataset(Dataset):
    def __init__(self, tokenizer, df, config, prompt_format_fn=to_alpaca_prompt):
        training = config.training

        # Make sure the df has 3 columns: instruction, input and output.
        self.df = df.copy()
        self.df['prompt'] = self.df.parallel_apply(
            prompt_format_fn, args=(training,), axis=1)
        self.df['gt'] = self.df.parallel_apply(
            prompt_format_fn, args=(True,), axis=1)
        self.texts = self.df['prompt'].values.tolist()
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.encode_texts(self.texts[item])
        return inputs

    def encode_texts(self, texts):
        padding = 'max_length'
        if self.config.use_flash_attention:
            padding = False
        inputs = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        return inputs

    def decode_outputs(self, outputs):
        texts = self.tokenizer.decode(outputs, skip_special_tokens=True)
