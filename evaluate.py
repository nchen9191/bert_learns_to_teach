from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def task_eval(student: Module, val_dataloader: DataLoader, task: str) -> Dict[str, float]:
    pass