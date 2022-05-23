import csv
import torch.utils.data as data
import os
import json
import time
import random
import datetime
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
CUDA = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
device = torch.device("cuda")
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "sst2"
model_name = "bert-base-uncased"
batch_size = 32
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
                  test_dataset, 
                  drop_out = [0.0,0.05,0.2],
                  temperature=0.05,
                  lr = 5e-05,
                  batch_size = 10,
                  epoch = 5,
                  warmup_steps=500,
                  logging_steps = 200,
                  evaluation_strategy = 'no',
                  supcontrast=False, 
                  save_name = "disguised_model"):
    model = BertModel.from_pretrained("bert-base-uncased")
    args = TrainingArguments(
            output_dir = './results8',
            save_total_limit = 1,
            num_train_epochs=epoch,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=64,
            evaluation_strategy = evaluation_strategy,
            logging_steps = logging_steps,
            learning_rate = lr,
            eval_steps = 200,
            warmup_steps=warmup_steps, 
            weight_decay=0.01,              
            logging_dir='./logs8',
        )
    
    if supcontrast:       #using supervised contrastive loss to perform MDA
        trainer = SupCsTrainer(
                    model,
                    args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer
                )
        trainer.set_views(drop_out, temperature)
    logging.basicConfig(level = logging.INFO)
    trainer.train()
    trainer.save_model('./' + save_name)   #save the disguised model
run_model(   
             model_name,
             train_dataset,
             test_dataset,
             drop_out = [0.05],          
             temperature = 0.05,          
             lr = 3e-5,                    
             save_name='cs_baseline_dropout0.10.05_8',   
             logging_steps = 50,
             warmup_steps = 100,           
             supcontrast= True,                 
             batch_size = 8,              
             epoch = 3
           )


