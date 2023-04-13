from typing import Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from evaluate import task_eval
from initialize import get_config, load_models
from pre_processing import get_data_loaders


def run_full_training(config_path):
    # Initialize with relevant parameters
    config = get_config(config_path)

    # Load initial models
    teacher, student = load_models(config)

    # Get Train and Dev DataLoaders
    train_dataloader, quiz_dataloader, val_dataloader, test_dataloader = get_data_loaders(config)

    # Run training
    final_teacher, final_student, train_loss, val_loss = train(config,
                                                               teacher,
                                                               student,
                                                               train_dataloader,
                                                               val_dataloader)

    # Get model metrics
    metrics = task_eval(final_student, val_dataloader, test_dataloader, config['task'])

    return final_teacher, final_student, metrics


def train(config: dict,
          teacher: Module,
          student: Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader) -> Tuple[Module, Module, float, float]:
    pass
