import json
import random
from typing import Tuple

import numpy as np
import torch.nn
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer

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


def load_models(config: dict) -> Tuple[torch.nn.Module, torch.nn.Module]:
    teacher_config = BertConfig.from_pretrained(config['finetuned_teacher'])
    teacher_config.num_labels = len(GLUE_META_DATA[config['task']])
    teacher_config.finetuning_task = config['task']
    teacher_config.output_hidden_states = True
    teacher_model = BertForSequenceClassification.from_pretrained(config['teacher_model_type'], config=teacher_config)

    student_config = BertConfig.from_pretrained(config['student_model_type'])
    student_config.num_hidden_layers = config['student_num_hidden_layers']
    student_config.num_labels = len(GLUE_META_DATA[config['task']])
    student_config.finetuning_task = config['task']
    student_config.output_hidden_states = True
    student_model = BertForSequenceClassification.from_pretrained(config['student_model_type'], config=student_config)

    return teacher_model, student_model


def load_tokenizer(config: dict):
    tokenizer = BertTokenizer.from_pretrained(config['teacher_model_type'], do_lower_case=config['do_lower_case'])
    return tokenizer
