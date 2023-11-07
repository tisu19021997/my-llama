import os
import pandas as pd
import fire

from utils.config_utils import update_config
from utils.model_utils import load_model_and_tokenizer_for_inference
from utils.inference_utils import inference_loop, compute_score
from configs import inference_config, quantization_config
from data.AlpacaDataset import AlpacaDataset
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM'] = "false"


def main(**kwargs):
    update_config(inference_config, **kwargs)
    update_config(quantization_config, **kwargs)

    # Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer_for_inference(
        inference_config, quantization_config)
    # Turn on evaluation mode.
    model.eval()

    test_dataset = pd.read_parquet(inference_config.val_dataset)
    if inference_config.debug:
        test_dataset = test_dataset[:5].copy()

    test_dataset = AlpacaDataset(tokenizer, test_dataset, inference_config)
    test_dataset = DataLoader(test_dataset, batch_size=inference_config.batch_size)

    # Get predictions.
    test_preds = inference_loop(
        model, test_dataset, inference_config)
    test_dataset.df['prediction'] = test_preds
    test_dataset.df[['gt', 'prediction']].to_json(
        'predictions.json', force_ascii=False)

    test_preds_ans = [test_pred.split(
        inference_config.response_phrase)[-1] for test_pred in test_preds]
    test_gt_ans = test_dataset.df[inference_config.label_column].tolist()

    for pred_, gt_ in zip(test_preds_ans[:5], test_gt_ans[:5]):
        print('Prediction:', pred_)
        print('Ground truth:', gt_)
        print('\n\n')

    # Get evaluation scores: ROUGE and BLEU.
    rouge_scores, _ = compute_score(test_gt_ans, test_preds_ans, tokenizer)

    for metric_name, metric_scores in rouge_scores.items():
        test_dataset[metric_name] = metric_scores
    test_dataset.to_json('test_result.json', force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
