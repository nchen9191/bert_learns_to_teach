import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


def truncate_seqs(seq_a, max_length, seq_b=None):
    curr_length = len(seq_a) + (len(seq_b) if seq_b else 0)

    if curr_length > max_length:
        trunc_length = curr_length - max_length

        if seq_b:
            trunc_length_a = int(trunc_length / 2.0)
            trunc_length_b = trunc_length_a + (1 if trunc_length % 2 == 1 else 0)

            seq_a, seq_b = seq_a[:-trunc_length_a], seq_b[:-trunc_length_b]
        else:
            seq_a = seq_a[:-trunc_length], seq_b

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
    if task in ['SST-2', "MNLI", "MNLI-mismatched", "QNLI", "RTE", "WNLI"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task in ["MRPC", "QQP"]:
        return acc_and_f1(preds, labels)
    elif task == "COLA":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task == "STS-B":
        return pearson_and_spearman(preds, labels)
    else:
        return KeyError(task)
