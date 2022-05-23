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
from transformers import AutoTokenizer,AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification, AutoModel, TrainingArguments, Trainer, DebertaModel
from sklearn.metrics import classification_report
import logging
import subprocess
from sklearn.cluster import KMeans
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
print(dataset['train'][0])
labels_ori=[]
f1=open("train.tsv","r")
m=0
for line in f1.readlines():
    if(m==0):
        m=m+1
        continue
    m=m+1
    line = line.strip().split('\t')
    labels_ori.append(int(line[1]))
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
                  test_dataset
                  ):
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to(device)
    args = TrainingArguments(
            per_device_eval_batch_size=32,
            output_dir = './results1'
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
feature =feature.detach().cpu().numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit_transform(feature)
dist=[]
labels=[]
for i in range(0,kmeans.shape[0]):
    dist.append(min(kmeans[i][0],kmeans[i][1]))
for i in range(0,kmeans.shape[0]):
    if(kmeans[i][0]<kmeans[i][1]):  #There are random factors when assigning the value of the cluster centroids (0/1) during performing k-means clustering; the values of corresponding cluster centroids are in the "labels" list in the following codes.
        labels.append(0)
    else:
        labels.append(1)
print("labels' len:",len(labels))
print("ori_labels's len:",len(labels_ori))
dist = np.array(dist)
indexes = np.argsort(dist)

f=open("index_sst2.txt","w")
for i in range(0,2000):
    if(labels[indexes[i]]==labels_ori[indexes[i]]): #perform filtering to ensure the closest features to the same cluster centroid are with the same target label
        f.write(str(indexes[i]))
        f.write('\n')
