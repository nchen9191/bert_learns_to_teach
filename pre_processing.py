import json
from collections import namedtuple
from typing import Tuple, List, Dict
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from task_specific_utils import GLUE_META_DATA
from utils import truncate_seqs

DataExample = namedtuple("DataExample", ["uid", "text_a", "text_b", "label"])
Features = namedtuple("Features", ["input_ids", "input_mask", "segment_ids", "label_id"])


class LabelIdParams:
    """
    Class to hold all parameters related to how to generate inputs for BERT and other models
    """

    def __init__(self, cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]",
                 sep_token_extra=False, pad_on_left=False, pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0,
                 sequence_b_segment_id=1, mask_padding_with_zero=True, max_seq_length=128):
        self.cls_token_at_end = cls_token_at_end
        self.cls_token = cls_token
        self.cls_token_segment_id = cls_token_segment_id
        self.sep_token = sep_token
        self.sep_token_extra = sep_token_extra
        self.pad_on_left = pad_on_left
        self.pad_token = pad_token
        self.pad_token_segment_id = pad_token_segment_id
        self.sequence_a_segment_id = sequence_a_segment_id
        self.sequence_b_segment_id = sequence_b_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero
        self.max_seq_length = max_seq_length

    def to_dict(self):
        return {
            "cls_token_at_end": self.cls_token_at_end,
            "cls_token": self.cls_token,
            "cls_token_segment_id": self.cls_token_segment_id,
            "sep_token": self.sep_token,
            "sep_token_extra": self.sep_token_extra,
            "pad_on_left": self.pad_on_left,
            "pad_token": self.pad_token,
            "pad_token_segment_id": self.pad_token_segment_id,
            "sequence_a_segment_id": self.sequence_a_segment_id,
            "sequence_b_segment_id": self.sequence_b_segment_id,
            "mask_padding_with_zero": self.mask_padding_with_zero,
            "max_seq_length": self.max_seq_length,
        }


def get_data_loaders(config: dict, tokenizer, quiz=True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    task = config['task']
    label_id_params = LabelIdParams(**config['label_id_params'])
    folder = config['task'] if task != 'mnli-mm' else 'mnli'
    data_path = Path(config['data_path'], folder)
    has_header = GLUE_META_DATA[task]['has_header']

    # Train and Quiz
    train_data_path = Path(data_path, "train.tsv")
    train_dataset = load_features_to_dataset(train_data_path, "train", task, tokenizer, label_id_params, has_header)

    if quiz:
        quiz_num = int(len(train_dataset) * 0.1)
        # quiz_num = 128
        train_num = len(train_dataset) - quiz_num
        train_dataset, quiz_dataset = random_split(train_dataset, [train_num, quiz_num])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'])
        quiz_dataloader = DataLoader(quiz_dataset, shuffle=True, batch_size=config['batch_size'])
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'])
        quiz_dataloader = None

    # Dev
    dev_file = "dev.tsv" if "dev_file" not in GLUE_META_DATA[config['task']] else GLUE_META_DATA[task]['dev_file']
    dev_data_path = Path(data_path, dev_file)
    dev_dataset = load_features_to_dataset(dev_data_path, "dev", task, tokenizer, label_id_params, has_header)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config['batch_size'])

    return train_dataloader, quiz_dataloader, dev_dataloader


def load_raw_data(data_path: Path, task: str, uid_prefix: str, is_test: bool = False, has_header: bool = True) -> List:
    if is_test:
        return load_raw_data_test(data_path, task)

    indices = GLUE_META_DATA[task]['col_indices']

    with open(data_path, 'r') as fp:
        dataset = []
        for i, line in enumerate(csv.reader(fp, delimiter="\t", quotechar=None)):
            if has_header and i == 0:
                continue
            example = DataExample(uid=f"{uid_prefix}-{line[indices[0]] if indices[0] else i}",
                                  text_a=line[indices[1]],
                                  text_b=line[indices[2]] if indices[2] else None,
                                  label=line[indices[3]])
            dataset.append(example)

    return dataset


def load_raw_data_test(data_path: Path, task: str):
    indices = GLUE_META_DATA[task]['test_col_indices']
    dummy_label = GLUE_META_DATA[task]['labels'][0]
    dummy_label = dummy_label if dummy_label else 0.0

    with open(data_path, 'r') as fp:
        dataset = []
        for i, line in enumerate(csv.reader(fp, delimiter="\t", quotechar=None)):
            if i == 0:
                continue
            example = DataExample(uid=f"{line[indices[0]]}",
                                  text_a=line[indices[1]],
                                  text_b=line[indices[2]] if indices[2] else None,
                                  label=dummy_label)
            dataset.append(example)

    return dataset


def transform_data_to_features(dataset: List,
                               label_map: Dict,
                               tokenizer,
                               label_id_params: LabelIdParams,
                               output_mode: str) -> List:
    features = []

    for i, data_example in enumerate(dataset):
        if (i + 1) % 10000 == 0:
            print(f"{i + 1} of {len(dataset)} processed to feature")

        # Tokenize
        tokens_a = tokenizer.tokenize(text=data_example.text_a)
        tokens_b = tokenizer.tokenize(text=data_example.text_b) if data_example.text_b else []

        # Truncate to max sequence length
        special_tokens_count = (4 if tokens_b else 3) - (0 if label_id_params.sep_token_extra else 1)
        tokens_a, tokens_b = truncate_seqs(tokens_a, label_id_params.max_seq_length - special_tokens_count, tokens_b)

        # Combine sequences and add sep tokens as needed
        tokens = tokens_a + [label_id_params.sep_token]
        tokens += [label_id_params.sep_token] if label_id_params.sep_token_extra else []
        segment_ids = [label_id_params.sequence_a_segment_id] * len(tokens)

        tokens += (tokens_b + [label_id_params.sep_token]) if tokens_b else []
        segment_ids += [label_id_params.sequence_b_segment_id] * (len(tokens_b) + 1) if tokens_b else []

        # Add cls token
        if label_id_params.cls_token_at_end:
            tokens = tokens + [label_id_params.cls_token]
            segment_ids = segment_ids + [label_id_params.cls_token_segment_id]
        else:
            tokens = [label_id_params.cls_token] + tokens
            segment_ids = [label_id_params.cls_token_segment_id] + segment_ids

        # Convert to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Build mask
        input_mask = [int(label_id_params.mask_padding_with_zero)] * len(input_ids)

        # Pad rest of sequence
        pad_length = label_id_params.max_seq_length - len(input_ids)
        if label_id_params.pad_on_left:
            input_ids = [label_id_params.pad_token] * pad_length + input_ids
            input_mask = [int(not label_id_params.mask_padding_with_zero)] * pad_length + input_mask
            segment_ids = [label_id_params.pad_token_segment_id] * pad_length + segment_ids
        else:
            input_ids += [label_id_params.pad_token] * pad_length
            input_mask += [int(not label_id_params.mask_padding_with_zero)] * pad_length
            segment_ids += [label_id_params.pad_token_segment_id] * pad_length

        # Make sure everything is of max sequence length
        assert len(input_ids) == label_id_params.max_seq_length
        assert len(input_mask) == label_id_params.max_seq_length
        assert len(segment_ids) == label_id_params.max_seq_length

        # Build label ids
        if output_mode == "classification":
            label_id = label_map[data_example.label]
        elif output_mode == "regression":
            label_id = float(data_example.label)
        else:
            raise KeyError(output_mode)

        # Add to list
        features.append(Features(input_ids, input_mask, segment_ids, label_id))

    return features


def load_features_to_dataset(data_path: Path,
                             data_type: str,
                             task: str,
                             tokenizer,
                             label_id_params: LabelIdParams,
                             has_header: bool,
                             is_test: bool = False):
    # Cache path
    cache_path = Path(data_path).parent / "cache/" / data_type
    if cache_path.exists():
        # Load cache if exists
        all_input_ids = torch.load(Path(cache_path, 'all_input_ids.pt'))
        all_input_mask = torch.load(Path(cache_path, 'all_input_mask.pt'))
        all_segment_ids = torch.load(Path(cache_path, 'all_segment_ids.pt'))
        all_label_ids = torch.load(Path(cache_path, 'all_label_ids.pt'))
    else:
        # Load the raw data
        data_set = load_raw_data(data_path, task, data_type, is_test, has_header)

        # Compute params
        label_map = {label: i for i, label in enumerate(GLUE_META_DATA[task]["labels"])}
        output_mode = GLUE_META_DATA[task]['output_mode'] if 'output_mode' in GLUE_META_DATA[task] else "classification"
        label_type = torch.float if output_mode == 'regression' else torch.long

        # Transform data examples to feature tuples
        features = transform_data_to_features(data_set, label_map, tokenizer, label_id_params, output_mode)

        # Convert all features to tensors
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=label_type)

        cache_path.mkdir(parents=True, exist_ok=True)
        torch.save(all_input_ids, Path(cache_path, 'all_input_ids.pt'))
        torch.save(all_input_mask, Path(cache_path, 'all_input_mask.pt'))
        torch.save(all_segment_ids, Path(cache_path, 'all_segment_ids.pt'))
        torch.save(all_label_ids, Path(cache_path, 'all_label_ids.pt'))

        # Build DataSet
    tensor_data_set = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return tensor_data_set


if __name__ == '__main__':
    config_path = "./example_config.json"
    with open(config_path, 'r') as j:
        config = json.load(j)

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(config['teacher_model_type'], do_lower_case=config['do_lower_case'])

    get_data_loaders(config, tokenizer)
