import json
import os

import torch
from pytorch_transformers import AdamW
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from transformers import get_linear_schedule_with_warmup


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


def get_optimizer_and_scheduler(params, total_steps, config, is_teacher=False):
    params = list(params)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': config['weight_decay']},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    learning_rate = config['learning_rate_teacher'] if is_teacher else config['learning_rate_student']
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=config['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, config['warmup_steps'], total_steps)

    return optimizer, scheduler


def get_order(teacher_config, student_config, select='skip'):
    if select == 'last':
        order = list(range(teacher_config.num_hidden_layers - 1))
        order = torch.LongTensor(order[-(student_config.num_hidden_layers - 1):])

    elif select == 'skip':
        order = list(range(teacher_config.num_hidden_layers - 1))
        every_num = teacher_config.num_hidden_layers // student_config.num_hidden_layers
        order = torch.LongTensor(order[(every_num - 1)::every_num])
    else:
        print('layer selection must be in [entropy, attn, dist, every]')
    order, _ = order[:(student_config.num_hidden_layers - 1)].sort()

    return order


def save_teacher_student_models(config, teacher, student, tokenizer):
    os.makedirs(os.path.join(config['output_dir'], config['task'], "teacher"))
    os.makedirs(os.path.join(config['output_dir'], config['task'], "student"))

    teacher.save_pretrained(os.path.join(config['output_dir'], config['task'], "teacher"))
    student.save_pretrained(os.path.join(config['output_dir'], config['task'], "student"))

    tokenizer.save_pretrained(os.path.join(config['output_dir'], config['task'], "teacher"))
    tokenizer.save_pretrained(os.path.join(config['output_dir'], config['task'], "student"))

    with open(os.path.join(config['output_dir'], config['task'], "teacher", 'config.json'), "w") as fp:
        json.dump(config, fp)

    with open(os.path.join(config['output_dir'], config['task'], "student", 'config.json'), "w") as fp:
        json.dump(config, fp)