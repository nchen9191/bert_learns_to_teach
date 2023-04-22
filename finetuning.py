# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer)

from evaluate import task_eval
from initialize import set_seed
from pre_processing import get_data_loaders
from task_specific_utils import GLUE_META_DATA

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def train(config, device):
    """ Train the model """
    set_seed(config['seed'])

    task = config['task']
    labels = GLUE_META_DATA[task]['labels']

    model_config = BertConfig.from_pretrained(config['finetuning_model_name'], num_labels=len(labels), finetuning_task=task)
    tokenizer = BertTokenizer.from_pretrained(config['finetuning_model_name'], do_lower_case=config['do_lower_case'])
    model = BertForSequenceClassification.from_pretrained(config['finetuning_model_name'], config=model_config)

    train_dataloader, quiz_dataloader, dev_dataloader = get_data_loaders(config, tokenizer, quiz=False)

    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=config['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config['warmup_steps'], t_total=len(train_dataloader))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", config['num_train_epochs'])

    model.zero_grad()

    results, dev_loss, _ = task_eval(model, dev_dataloader, task, device)
    _, tr_loss, _ = task_eval(model, train_dataloader, task, device)
    print(f"Epoch 0 (before training), Train Loss: {tr_loss}, Dev Loss: {dev_loss}, Task Metrics: {results}")

    train_iterator = trange(config['num_train_epochs'], desc="Epoch")
    for i in train_iterator:
        tr_loss = 0.0
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            tr_loss += loss.item()

            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()

        results, dev_loss, preds = task_eval(model, dev_dataloader, task, device)
        tr_loss /= len(train_dataloader)
        print(f"Epoch {i + 1}, Train Loss: {tr_loss}, Dev Loss: {dev_loss}, Task Metrics: {results}")

    return model, tokenizer


def main():
    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    #                     help="Path to pre-trained model or shortcut name selected in the list: ")
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    #
    # ## Other parameters
    # parser.add_argument("--config_name", default="", type=str,
    #                     help="Pretrained config name or path if not the same as model_name")
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Pretrained tokenizer name or path if not the same as model_name")
    # parser.add_argument("--cache_dir", default="", type=str,
    #                     help="Where do you want to store the pre-trained models downloaded from s3")
    # parser.add_argument("--max_seq_length", default=128, type=int,
    #                     help="The maximum total input sequence length after tokenization. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")
    # parser.add_argument("--do_train", action='store_true',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true',
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--evaluate_during_training", action='store_true',
    #                     help="Rul evaluation during training at each logging step.")
    # parser.add_argument("--do_lower_case", action='store_true',
    #                     help="Set this flag if you are using an uncased model.")
    #
    # parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for training.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    # parser.add_argument("--num_train_epochs", default=3.0, type=float,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--max_steps", default=-1, type=int,
    #                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="Linear warmup over warmup_steps.")
    #
    # parser.add_argument('--logging_steps', type=int, default=50,
    #                     help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=50,
    #                     help="Save checkpoint every X updates steps.")
    # parser.add_argument("--eval_all_checkpoints", action='store_true',
    #                     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    # parser.add_argument("--no_cuda", action='store_true',
    #                     help="Avoid using CUDA when available")
    # parser.add_argument('--overwrite_output_dir', action='store_true',
    #                     help="Overwrite the content of the output directory")
    # parser.add_argument('--overwrite_cache', action='store_true',
    #                     help="Overwrite the cached training and evaluation sets")
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="random seed for initialization")
    #
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    # parser.add_argument('--fp16_opt_level', type=str, default='O1',
    #                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #                          "See details at https://nvidia.github.io/apex/amp.html")
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="For distributed training: local_rank")
    # parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    # args = parser.parse_args()

    config_path = "./finetuning_config.json"
    with open(config_path, "r") as fp:
        config = json.load(fp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)

    # Training
    model, tokenizer = train(config, device)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    model.save_pretrained(os.path.join(config['output_dir'], config['task']))
    tokenizer.save_pretrained(os.path.join(config['output_dir'], config['task']))

    # Good practice: save your training arguments together with the trained model
    with open(os.path.join(config['output_dir'], config['task'], 'config.json'), "w") as fp:
        json.dump(config, fp)


if __name__ == "__main__":
    main()
