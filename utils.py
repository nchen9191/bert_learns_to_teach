import os

import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


def truncate_seqs(seq_a, max_length, seq_b):
    while len(seq_a) + len(seq_b) > max_length:
        if len(seq_a) > len(seq_b):
            seq_a.pop()
        else:
            seq_b.pop()

    return seq_a, seq_b


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_task_metrics(preds, labels, task):
    if task in ['sst-2', "mnli", "mnli-mm", "qnli", "rte", "wnli"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task in ["mrpc", "qqp"]:
        return acc_and_f1(preds, labels)
    elif task == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task == "sts-b":
        return pearson_and_spearman(preds, labels)
    else:
        return KeyError(task)


def load_saved_model(model_path):
    model, _ = torch.load(model_path)
    return model