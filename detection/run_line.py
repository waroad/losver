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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import glob
import logging
import math
import os
import pickle
import random
import re
import shutil
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json



import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from datetime import datetime


cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification,
                          DistilBertTokenizer, RobertaModel, T5Config, T5EncoderModel)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'codet5': (T5Config, T5EncoderModel, RobertaTokenizer)
}


class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        return x + self.layers(x)  # Skip connection


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.args = args
        self.drop_out = 0.0

        self.newline_token_id = self.tokenizer.convert_tokens_to_ids('\n')

        # If tokenizer doesn't recognize '\n' directly, try alternate encodings
        if self.newline_token_id == self.tokenizer.unk_token_id:
            # Try with special token format
            self.newline_token_id = self.tokenizer.convert_tokens_to_ids('Ċ')  # Some tokenizers use this

            # If still not found, you can try to encode a string with a newline and check
            if self.newline_token_id == self.tokenizer.unk_token_id:
                newline_tokens = self.tokenizer.encode('\n', add_special_tokens=False)
                if newline_tokens:  # If not empty
                    self.newline_token_id = newline_tokens[0]

        print(f"Newline token ID: {self.newline_token_id}")
        # Single attention layer for global context
        self.line_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=self.drop_out
        )

        self.line_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            ResidualBlock(  # Optional residual block
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.drop_out)
                )
            ),
            nn.Linear(self.hidden_size, 1)
        )

    def focal_loss(self, logits, labels):
        pos_weight = torch.tensor(5, device=logits.device)

        # BCEWithLogitsLoss applies sigmoid internally
        ce_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction='none'
        )
        p_t = torch.sigmoid(logits) * labels + (1 - torch.sigmoid(logits)) * (1 - labels)
        alpha = 0.8
        gamma = 1.5
        alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)
        modulating_factor = (1.0 - p_t).pow(gamma)
        return alpha_factor * modulating_factor * ce_loss

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        line_logits = []
        for i in range(input_ids.size(0)):  # Batch loop
            line_indices = (input_ids[i] == self.newline_token_id).nonzero(as_tuple=True)[0]  #30790
            line_start_indices = torch.cat([torch.tensor([0], device=input_ids.device), line_indices + 1])
            code_end = line_start_indices[-1]  # Last line start is end of code

            sample_line_logits = []
            global_context = outputs[i, :code_end]
            for j in range(len(line_start_indices) - 1):
                start = line_start_indices[j]
                end = line_start_indices[j + 1] - 1

                if not end > start:
                    sample_line_logits.append(torch.tensor(-10, device=input_ids.device))
                    continue

                line_repr = outputs[i, start:end]
                attn_output, _ = self.line_attention(
                    line_repr.unsqueeze(0),
                    global_context.unsqueeze(0),
                    global_context.unsqueeze(0)
                )
                line_final = torch.cat([
                    line_repr.mean(0),
                    attn_output.squeeze(0).mean(0),
                ])
                logit = self.line_classifier(line_final)
                sample_line_logits.append(logit.squeeze(-1))

            if sample_line_logits:
                line_logits.append(torch.stack(sample_line_logits))
            else:
                line_logits.append(torch.zeros(1, device=input_ids.device))

        # Loss computation with focal loss
        if labels is not None:
            loss = 0.0
            total_valid = 0

            for i, logit in enumerate(line_logits):
                valid_length = min(logit.size(0), labels[i].size(0))
                valid_mask = (labels[i, :valid_length] != -1).float()
                valid_labels = labels[i, :valid_length].clamp(0, 1).float()
                focal_loss = self.focal_loss(
                    logit[:valid_length],
                    valid_labels,
                )
                loss += (focal_loss * valid_mask).sum()
                total_valid += valid_mask.sum()

            loss = loss / total_valid if total_valid > 0 else loss * 0.0
            line_logits=[torch.sigmoid(logits) for logits in line_logits]
            return loss, line_logits

        return [torch.sigmoid(logits) for logits in line_logits]


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = js['func']
    line_info = js['lines_ground_truth']
    lines = code.splitlines()

    sparse_matrix = [1 if x in line_info else 0 for x in range(len(lines))][:args.block_size]
    sparse_matrix+=[-1]*(args.block_size-len(lines))
    code_tokens = []
    for ind,line in enumerate(lines):
        line_token=tokenizer.tokenize(line+"\n")
        if len(code_tokens)+len(line_token)<=args.block_size - 2:
            code_tokens.extend(line_token)
        else:
            for i in range(ind,len(sparse_matrix)):
                sparse_matrix[i]=-1
            break

    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, sparse_matrix), js["idx"]


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, is_test=False):
        self.examples = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                js = json.loads(line.strip())
                temp= convert_examples_to_features(js, tokenizer, args)
                if temp:
                    self.examples.append(temp[0])
                    if is_test:
                        args.indexes.append(temp[1])
        print("Code Len:", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def add_label(jsonl_path,output_path, predicted_label, args):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        with open(jsonl_path, 'r') as file:
            pr_count=0
            for ind, js in enumerate(file):
                entry = json.loads(js)
                if entry["idx"] in args.indexes:
                    line_info=[]
                    for idx, p in enumerate(predicted_label[pr_count]):
                        if p==1:
                            line_info.append(idx)
                    entry['lines_predicted']=line_info
                    output_file.write(json.dumps(entry) + '\n')
                    pr_count+=1


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0.0
    model.zero_grad()

    # Initialize early stopping parameters at the start of training
    early_stopping_counter = 0
    best_loss = None

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        model.train()
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            loss, logits = model(inputs, labels=labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint

                    if results['f1'] > best_f1:
                        best_f1 = results['f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num

        # Check for early stopping condition
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:

                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break  # Exit the loop early


def top_k_accuracy(predictions, labels, k):
    correct = 0
    total = 0
    for pred, label in zip(predictions, labels):
        top_k_indices = np.argsort(pred)[-k:]
        total += np.sum(label != -1)  # Count valid lines
        correct += np.sum(label[top_k_indices] == 1)  # Check if faulty lines are in top K
    return correct / total


def evaluate(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []

    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            for sample_logit in logit:
                logits.append(sample_logit.cpu().numpy())  # Convert tensor to numpy and collect it
            for sample_label in label:
                labels.append(sample_label.cpu().numpy())  # Collect labels as numpy arrays
        nb_eval_steps += 1

    # No need to concatenate since `logits` is already a list of arrays
    logits = np.array(logits, dtype=object)  # Use dtype=object for variable-length sequences
    labels = np.array(labels, dtype=object)
    all_logits = np.concatenate(logits)

    result = {}
    kk = [1, 3, 5]  # Added -1 for raw predictions
    for k in kk:
        adjusted_logits = []
        adjusted_labels = []
        adjusted_preds = []
        top_k_hits = []

        for logit, label in zip(logits, labels):
            valid_indices = np.arange(len(logit))
            label = label[valid_indices]
            # Original top-k logic
            if k == 0:
                # For raw predictions, use 0.5 threshold
                raw_pred = (logit > 0.5).astype(int)
                top_k_pred = raw_pred
                hit = np.all((label == 1) == (raw_pred == 1))  # Exact match for raw predictions
            else:
                if len(logit) < k:
                    top_k_indices = np.argsort(logit)[::-1]
                else:
                    top_k_indices = np.argsort(logit)[::-1][:k]

                top_k_pred = np.zeros_like(logit)
                top_k_pred[top_k_indices] = 1
                hit = np.any((label == 1) & (top_k_pred == 1))

            top_k_hits.append(hit)
            adjusted_logits.append(logit)
            adjusted_labels.append(label)
            adjusted_preds.append(top_k_pred)

        flattened_labels = np.concatenate(adjusted_labels, axis=0)
        flattened_logits = np.concatenate(adjusted_logits, axis=0)

        # Threshold for binary classification
        flattened_preds = (flattened_logits > 0.5).astype(int)
        flattened_labels = flattened_labels.astype(np.int32)
        flattened_preds = flattened_preds.astype(np.int32)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(flattened_labels, flattened_preds).ravel()

        # Calculate FPR and FNR
        fpr = fp / (fp + tn)  # False Positive Rate
        fnr = fn / (fn + tp)  # False Negative Rate

        f1 = f1_score(flattened_labels, flattened_preds, average='weighted', zero_division=0)

        # Calculate top-k accuracy
        top_k_accuracy = np.mean(top_k_hits)
        if k == 1:
            print("sample preds:", adjusted_preds[0])
            print("sample label:", adjusted_labels[0])
            print("sample preds:", adjusted_preds[1])
            print("sample label:", adjusted_labels[1])

        print(f"Top-{k} Accuracy: {top_k_accuracy}")
        result[f"top_{k}"] = round(top_k_accuracy, 4)
        if k == 5:
            print(f"False Positive Rate: {fpr:.4f}")
            print(f"False Negative Rate: {fnr:.4f}")
            print(f"F1 Score: {f1}")
            result["f1"] = round(f1, 4)
    result["eval_loss"] = eval_loss

    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    args.indexes=[]
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file, is_test=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []

    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            for sample_logit in logit:
                logits.append(sample_logit.cpu().numpy())  # Convert tensor to numpy and collect it
            for sample_label in label:
                labels.append(sample_label.cpu().numpy())  # Collect labels as numpy arrays
        nb_eval_steps += 1

    # No need to concatenate since `logits` is already a list of arrays
    logits = np.array(logits, dtype=object)  # Use dtype=object for variable-length sequences
    labels = np.array(labels, dtype=object)
    print("----------------------",args.test_data_file,"----------------------")
    logger.info("***** Predicted Result of %s *****", args.test_data_file)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    with open('result_line.txt', 'a') as f:  # 'a' mode appends to existing content
        f.write(f"{args.test_data_file}\n")
    kk = [0, 1, 3, 5]  # Added -1 for raw predictions

    for k in kk:
        adjusted_logits = []
        adjusted_labels = []
        adjusted_preds = []
        top_k_hits = []

        for logit, label in zip(logits, labels):
            valid_indices = np.arange(len(logit))
            label = label[valid_indices]
            # Original top-k logic
            if k== 0:
                # For raw predictions, use 0.5 threshold
                raw_pred = (logit > 0.5).astype(int)
                top_k_pred = raw_pred
                hit = np.all((label == 1) == (raw_pred == 1))  # Exact match for raw predictions
            else:
                if len(logit) < k:
                    top_k_indices = np.argsort(logit)[::-1]
                else:
                    top_k_indices = np.argsort(logit)[::-1][:k]

                top_k_pred = np.zeros_like(logit)
                top_k_pred[top_k_indices] = 1
                hit = np.any((label == 1) & (top_k_pred == 1))

            top_k_hits.append(hit)
            adjusted_logits.append(logit)
            adjusted_labels.append(label)
            adjusted_preds.append(top_k_pred)


        # Save predictions with appropriate suffix
        if k==0:
            add_label(args.test_data_file, f"{args.output_dir}/{args.test_data_file.split('_')[0]}.jsonl", adjusted_preds, args)
        flattened_labels = np.concatenate(adjusted_labels, axis=0)
        flattened_logits = np.concatenate(adjusted_logits, axis=0)

        # Threshold for binary classification
        flattened_preds = (flattened_logits > 0.5).astype(int)
        flattened_labels = flattened_labels.astype(np.int32)
        flattened_preds = flattened_preds.astype(np.int32)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(flattened_labels, flattened_preds).ravel()

        # Calculate FPR and FNR
        fpr = fp / (fp + tn)  # False Positive Rate
        fnr = fn / (fn + tp)  # False Negative Rate

        f1 = f1_score(flattened_labels, flattened_preds, average='weighted', zero_division=0)

        # Calculate top-k accuracy
        top_k_accuracy = np.mean(top_k_hits)

        print(f"Top-{k} Accuracy: {top_k_accuracy}")
        with open('result_line.txt', 'a') as f:  # 'a' mode appends to existing content
            f.write(f"Top-{k} Accuracy: {top_k_accuracy}\n")
        if k==5:
            print(f"False Positive Rate: {fpr:.4f}")
            print(f"False Negative Rate: {fnr:.4f}")
            print(f"F1 Score: {f1}")
            with open('result_line.txt', 'a') as f:  # 'a' mode appends to existing content
                f.write(f"False Positive Rate: {fpr:.4f}\n")
                f.write(f"False Negative Rate: {fnr:.4f}\n")
                f.write(f"F1 Score: {f1}\n")


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", required=True, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", required=True, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", required=True, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run val on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run val on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', default=True,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--GPU", default=0, type=int,
                        help="..")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=40,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # Set the GPU(s) explicitly
        if isinstance(args.GPU, list):  # If multiple GPUs are specified
            selected_gpus = ",".join(map(str, args.GPU))
            torch.cuda.set_device(args.GPU[0])  # Set the first GPU as the primary device
            device = torch.device(f"cuda:{args.GPU[0]}")
            args.n_gpu = len(args.GPU)
        else:  # Single GPU case
            torch.cuda.set_device(args.GPU)
            device = torch.device(f"cuda:{args.GPU}")
            args.n_gpu = 1
    else:  # Distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)
    with open('result_line.txt', 'a') as f:  # 'a' mode appends to existing content
        f.write(f"-------------------------{datetime.now()}------------------------- \n")
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    if args.model_name_or_path:
        base_model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    else:
        base_model = model_class(config)

    model = Model(base_model, tokenizer=tokenizer, config=config, args=args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        test_names=[args.test_data_file, args.eval_data_file, args.train_data_file]
        for name in test_names:
            args.test_data_file = name

            checkpoint_prefix = 'checkpoint-best-f1/pytorch_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
            test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()

