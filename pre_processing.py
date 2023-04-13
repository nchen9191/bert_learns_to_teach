from typing import Tuple

import pandas as pd

import torch
from torch.utils.data import DataLoader


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    pass


def load_raw_data(data_path: str, task: str) -> pd.DataFrame:
    pass


def transform_data(data: pd.DataFrame, task: str) -> torch.Tensor:
    pass
