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

    test_df = pd.read_parquet(inference_config.val_dataset)
    if inference_config.debug:
        test_df = test_df[:5].copy()

    test_dataset = AlpacaDataset(tokenizer, test_df, inference_config)
    result_df = test_dataset.df.copy()
    test_dataset = DataLoader(
        test_dataset, batch_size=inference_config.batch_size)

    # Get predictions.
    test_preds = inference_loop(
        model, test_dataset, tokenizer, inference_config)
    result_df['prediction'] = test_preds
    result_df[['gt', 'prediction']].to_json(
        'predictions.json', force_ascii=False)

    # test_preds_ans = [test_pred.split(
    #    inference_config.response_phrase)[-1] for test_pred in test_preds]
    test_gt_ans = result_df['output'].tolist()

    for pred_, gt_ in zip(test_preds[:5], test_gt_ans[:5]):
        print('Prediction:', pred_)
        print('Ground truth:', gt_)
        print('\n\n')

    # Get evaluation scores: ROUGE and BLEU.
    rouge_scores, _ = compute_score(test_gt_ans, test_preds, tokenizer)

    for metric_name, metric_scores in rouge_scores.items():
        result_df[metric_name] = metric_scores
    result_df.to_json('test_result.json', force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
