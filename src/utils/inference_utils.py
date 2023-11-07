import os
import torch
import gc
import evaluate
import numpy as np
from transformers import GenerationConfig
from tqdm import tqdm


def inference_loop(model, test_dataset, tokenizer, config):
    preds = list()
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty
    )
    for inputs in tqdm(test_dataset, total=len(test_dataset)):
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )[0].to('cpu').numpy()
        # Remove the input prompt from the result.
        decoded_pred = tokenizer.decode(outputs[input_ids.shape[1]:], skip_special_tokens=True)
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
