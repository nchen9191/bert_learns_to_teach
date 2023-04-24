import os.path
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from pre_processing import load_features_to_dataset, LabelIdParams, load_raw_data_test
from task_specific_utils import GLUE_META_DATA
from utils import compute_task_metrics


def model_inference(model: Module, dataloader: DataLoader, task: str, device) -> Tuple[np.array, float]:
    model.eval()

    loss, num_steps = 0.0, 0
    preds = []
    output_mode = GLUE_META_DATA[task].get("output_mode", "classification")
    num_labels = len(GLUE_META_DATA[task]['labels'])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Iteration", position=0, leave=True):
            batch = tuple(t.to(device) for t in batch)

            input_ids, attention_mask, token_type_ids, labels = batch[:4]
            labels = batch[3] if len(batch) > 3 else torch.zeros((input_ids.shape[0], num_labels), device=device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            loss += tmp_eval_loss.mean().item()

            num_steps += 1
            preds.append(logits.detach().cpu().numpy())

        preds = np.vstack(preds)

        loss = loss / num_steps
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.clip(np.squeeze(preds), 0.0, 5.0)

    return preds, loss


def task_eval(model: Module, dataloader: DataLoader, task: str, device) -> Tuple[Dict[str, float], float, np.array]:
    preds, loss = model_inference(model, dataloader, task, device)
    labels = torch.hstack([batch[3].flatten().detach().cpu() for batch in dataloader]).numpy()

    result = compute_task_metrics(preds, labels, task)
    return result, loss, preds


def test_evaluate(model_path, data_path, output_path, device):
    tasks = ["CoLA", "SST-2", "STS-B", "RTE", "WNLI"]
    label_id_params = LabelIdParams()

    for task in tasks:
        print("Processing:", task)
        task_name = task.lower()
        labels = GLUE_META_DATA[task_name]['labels']

        # Load student model
        task_model_path = os.path.join(model_path, task_name, "student")
        student_config = BertConfig.from_pretrained(task_model_path)
        student_config.num_hidden_layers = 6
        student_config.num_labels = len(labels)
        student_config.finetuning_task = task_name
        student_config.output_hidden_states = True
        student_model = BertForSequenceClassification.from_pretrained(task_model_path, config=student_config)
        student_model.to(device)

        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(task_model_path)

        # Load test dataloader
        task_data = Path(data_path, task_name, "test.tsv")
        test_dataset = load_features_to_dataset(task_data, "test", task_name, tokenizer, label_id_params, True, True)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=256)

        # Run inference
        test_preds, _ = model_inference(student_model, test_dataloader, task_name, device)
        test_preds = test_preds.tolist()
        indices = [int(d.uid) for d in load_raw_data_test(task_data, task_name)]

        # Map back to class string names if not sst-2
        if task_name != "sts-b":
            reverse_label_map = {i: label for i, label in enumerate(labels)}
            test_preds = [reverse_label_map[ind] for ind in test_preds]

        # Save to tsv
        Path(output_path).mkdir(exist_ok=True)
        df = pd.DataFrame({'index': indices, 'prediction': test_preds})
        df.to_csv(Path(output_path, task + ".tsv"), sep="\t", index=False)


if __name__ == '__main__':
    data_path = "../data/"
    model_path = "../models/meta_distil_models_temp/"
    output_path = "../test/"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_evaluate(model_path, data_path, output_path, device)
