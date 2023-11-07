import os
import torch
import gc
import evaluate
import numpy as np
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = "false"


def encode_texts(texts, tokenizer, config):
    padding = 'max_length'
    if config.use_flash_attention:
        padding = False
    encoded_texts = tokenizer(
        texts,
        padding=padding,
        truncation=True,
        max_length=config.max_length,
        return_tensors='pt'
    )
    encoded_texts = {k: v.to('cuda') for k, v in encoded_texts.items()}
    return encoded_texts


def inference_loop(model, tokenizer, test_dataset, config):
    preds = list()

    for text in tqdm(test_dataset.texts, total=len(test_dataset.texts)):
        inputs = encode_texts(text, tokenizer, config)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            encoded_pred = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty
            )[0].to('cpu').numpy()
        decoded_pred = tokenizer.decode(encoded_pred, skip_special_tokens=True)
        preds.append(decoded_pred.strip())

        del decoded_pred
        gc.collect()
        torch.cuda.empty_cache()

    return preds


def compute_score(labels, preds, tokenizer):
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=preds, references=labels,
                                 use_aggregator=False, tokenizer=lambda x: tokenizer(x)['input_ids'])
    rouge_scores_mean = {k: np.mean(v) for k, v in rouge_scores.items()}
    print(rouge_scores_mean)

    google_bleu = evaluate.load("google_bleu")
    bleu_scores = google_bleu.compute(
        predictions=preds, references=labels, tokenizer=lambda x: tokenizer(x)['input_ids'])
    print(bleu_scores)

    return rouge_scores, bleu_scores
