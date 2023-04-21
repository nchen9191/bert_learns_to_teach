import json
import os
import random
from typing import Tuple

import numpy as np
import torch.nn
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer, PretrainedConfig

from task_specific_utils import GLUE_META_DATA


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as j:
        return json.load(j)


def load_models(config: dict, device) -> Tuple[torch.nn.Module, torch.nn.Module, PretrainedConfig, PretrainedConfig]:
    teacher_model_path = os.path.join(config['teacher_model_type'], config['task'])
    teacher_config = BertConfig.from_pretrained(teacher_model_path)
    teacher_config.num_labels = len(GLUE_META_DATA[config['task']]['labels'])
    teacher_config.finetuning_task = config['task']
    teacher_config.output_hidden_states = True
    teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_path, config=teacher_config)

    student_config = BertConfig.from_pretrained(config['student_model_type'])
    student_config.num_hidden_layers = config['student_num_hidden_layers']
    student_config.num_labels = len(GLUE_META_DATA[config['task']]['labels'])
    student_config.finetuning_task = config['task']
    student_config.output_hidden_states = True
    student_model = BertForSequenceClassification.from_pretrained(config['student_model_type'], config=student_config)

    teacher_model.to(device)
    student_model.to(device)

    return teacher_model, student_model, teacher_config, student_config


def load_tokenizer(config: dict):
    path = os.path.join(config['teacher_model_type'], config['task'])
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=config['do_lower_case'])
    return tokenizer
