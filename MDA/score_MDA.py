import csv
import torch.utils.data as data
import os
import json
import time
import random
import datetime
from LogME import LogME
from transformers import TrainingArguments, Trainer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from collections import Counter
from sklearn import metrics
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
import torch
import torch.nn.functional as FF
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel, TrainingArguments, Trainer, DebertaModel
from sklearn.metrics import classification_report
import logging
import subprocess
from SupCsTrainer import SupCsTrainer
import numpy as np
CUDA = "5"
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
device = torch.device("cuda")
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "sst2"
model_name = "bert-base-uncased"
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("csv", data_files={"train": 'train.tsv',
    "validation": 'dev.tsv'},delimiter='\t')
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
sentence1_key="sentence"
sentence2_key = None
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)

    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
    
encoded_dataset = dataset.map(preprocess_function, batched=True)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0],
    return metric.compute(predictions=predictions, references=labels)
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matche" if task == "mnli" else "validation"
train_dataset = encoded_dataset["train"]
test_dataset = encoded_dataset[validation_key]

def run_model( 
              model_name,
              train_dataset,
                  test_dataset):
    model = BertModel.from_pretrained('./cs_baseline_dropout0.10.05_8')   #load the disguised model
    model.eval()
    model.to(device)
    args = TrainingArguments(
            output_dir = './results1',
            per_device_eval_batch_size=32
        )


    trainer = SupCsTrainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer
            )

    feature,labels = trainer.get_feature(model,train_dataset)
    return feature,labels

feature,labels = run_model(   
             model_name,
             train_dataset,
             test_dataset
           )
score = LogME(feature, labels)  #calculate the LogME score
print("score:",score)




