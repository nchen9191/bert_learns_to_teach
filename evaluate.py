from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from pre_processing import load_features_to_dataset, LabelIdParams
from task_specific_utils import GLUE_META_DATA
from utils import compute_task_metrics, load_saved_model


def model_inference(model: Module, dataloader: DataLoader, task: str, device) -> Tuple[np.array, float]:
    model.eval()

    loss, num_steps = 0.0, 0
    preds = np.array([])
    output_mode = GLUE_META_DATA[task]['output_mode']

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)

            input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
            labels = batch[3] if len(batch) > 3 else torch.zeros(input_ids.shape)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            loss += tmp_eval_loss.mean().item()

            num_steps += 1
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        loss = loss / num_steps
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

    return preds, loss


def task_eval(model: Module, dataloader: DataLoader, task: str, device) -> Tuple[Dict[str, float], float]:
    preds, loss = model_inference(model, dataloader, task, device)
    labels = torch.hstack([batch[3].flatten().detach().cpu() for batch in dataloader]).numpy()

    result = compute_task_metrics(preds, labels, task)
    return result, loss


def test_evaluate(config, model_path_dict, tokenizer, device):
    tasks = ["mrpc", "mnli", "mnli-mm", "cola", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli"]
    label_id_params = LabelIdParams(**config['label_id_params'])
    preds_dict = {}

    for task in tasks:
        data_path = Path(config['data_path'], task, "test.csv")
        has_header = GLUE_META_DATA[task]['has_header']
        test_dataset = load_features_to_dataset(data_path, "test", task, tokenizer, label_id_params, has_header, True)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'])

        model = load_saved_model(model_path_dict[task])

        test_preds, _ = model_inference(model, test_dataloader, task, device)
        preds_dict[task] = test_preds.tolist()

    return preds_dict
