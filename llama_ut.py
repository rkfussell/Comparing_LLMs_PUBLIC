import pandas as pd 
import numpy as np
import os 
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, model_selection
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from prompts import *
# bert imports
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,WeightedRandomSampler
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

# llama imports
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from copy import deepcopy
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoModelForSequenceClassification
from transformers import LlamaForSequenceClassification#, LlamaForQuestionAnswering
import torch
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import scipy

import random
import time
from utilities import compute_metrics

### LLaMA
def preprocessing_for_llama(code, train, val):

    train=train.rename(columns={code:"target"})
    
    pos_weights = len(train) / (2 * train.target.value_counts()[1])
    neg_weights = len(train) / (2 * train.target.value_counts()[0])

    max_words = train['Sentences'].str.split().str.len().max()
    
    val=val.rename(columns={code:"target"})
    
    data = DatasetDict({
        "train": Dataset.from_pandas(train),
        "test": Dataset.from_pandas(val)
        })
    col_to_delete = ['Sentences']
    return pos_weights, neg_weights, max_words, data, col_to_delete



def tokenize_for_llama(llama_checkpoint, data, col_to_delete, max_len, prompt="", hfToken=""):
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint, add_prefix_space=True, token=hfToken)
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print("adding prompt: {}".format(prompt))
    def llama_preprocessing_function(examples):
        #print(examples['Sentences'])
        return llama_tokenizer([prompt + ex for ex in examples['Sentences']], truncation=True, max_length=max_len)
    
    llama_tokenized_datasets = data.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
    llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
    llama_tokenized_datasets.set_format("torch")
    
    # Data collator for padding a batch of examples to the maximum length seen in the batch
    llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)
    return llama_tokenized_datasets, llama_data_collator, llama_tokenizer







def tokenize_for_llama_test(val, code, llama_tokenizer, col_to_delete, max_len):
    val=val.rename(columns={code:"target"})
    
    test_data = DatasetDict({"test": Dataset.from_pandas(val)})
    ##prepare the dataset the same way as the original one 
    def llama_preprocessing_function(examples):
        return llama_tokenizer(examples['Sentences'], truncation=True, max_length=max_len)
    llama_tokenized_datasets = test_data.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
    llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
    llama_tokenized_datasets.set_format("torch")
    # Data collator for padding a batch of examples to the maximum length seen in the batch
    llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)
    return llama_tokenized_datasets






class WeightedCELossTrainer(Trainer):
    def __init__(self,weights, *args, **kwargs):
        super(WeightedCELossTrainer, self).__init__(*args, **kwargs)
        self.neg_weights=weights[0]
        self.pos_weights=weights[1]
        print(self.pos_weights)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        #print(inputs)
    #print("compute loss")
     #   print(labels[0])
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([self.neg_weights, self.pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
 




class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
   


        
def set_llama_model(llama_checkpoint,hfToken="", ty='class'):
    #set up llama model
    if ty=='class':
        llama_model =  LlamaForSequenceClassification.from_pretrained(
            llama_checkpoint,
            device_map={"": 0},
            token=hfToken
        )
    elif ty=='question':
        llama_model =  LlamaForQuestionAnswering.from_pretrained(
            llama_checkpoint,
            device_map={"": 0},
            token=hfToken
        )
    llama_model.config.pad_token_id = llama_model.config.eos_token_id
    #LoRA
    llama_peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none", 
        target_modules=[
            "q_proj",
            "v_proj",  
        ],
    )
    llama_model = get_peft_model(llama_model, llama_peft_config)
    llama_model.print_trainable_parameters()
    llama_model = llama_model.cuda()
    return llama_model













def train_llama(llama_model, llama_tokenized_datasets, llama_data_collator, pos_weights, neg_weights,num_epochs = 5):
    lr = 1e-4
    batch_size = 16
    
    training_args = TrainingArguments(
        output_dir="llama-lora-token-classification",
        learning_rate=lr,
        lr_scheduler_type= "cosine",
        warmup_ratio= 0.1,
        max_grad_norm= 0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        report_to='none',
    )
    
    llama_trainer = WeightedCELossTrainer(
        weights = [neg_weights, pos_weights],
        model=llama_model,
        args=training_args,
        train_dataset=llama_tokenized_datasets['train'],
        eval_dataset=llama_tokenized_datasets["test"],
        data_collator=llama_data_collator,
        compute_metrics=compute_metrics
    )
    llama_trainer.add_callback(CustomCallback(llama_trainer))
    llama_trainer.train()
    return llama_trainer







def llama_logits(llama_model, dataset, batchsize = 1):
    batchsize=1
    with torch.no_grad():
        testset=dataset['test']
        llama_model=llama_model.eval()
        logits=torch.zeros(( len(testset), 2))
        
        for i in range(0, len(testset), batchsize):

            ts=testset[i:i+batchsize]
            maxlen=max([len(t) for t in ts['input_ids']])
            ex=min(batchsize, len(testset)-i)
            input_ids=torch.ones((ex,maxlen), dtype=int)*2
            attention_mask=torch.zeros((ex,maxlen ), dtype=int)
            for j in range(ex):
                input_ids[j, 0:len(testset[i+j]['input_ids'])]=testset[i+j]['input_ids']
                attention_mask[j, 0:len(testset[i+j]['attention_mask'])]=testset[i+j]['attention_mask']
#             print("llama logits input ids")
#             print(input_ids[0])
            with torch.no_grad():
                logits[ i:i+ex,:]=llama_model(**{'input_ids':input_ids.cuda(), 'attention_mask':attention_mask.cuda()}).logits.cpu()
            del input_ids
            del attention_mask
#         print(logits[0])
        return logits, torch.argmax(logits, axis=1)


def llama_logits2(llama_model, dataset, batchsize=1):
    model=model.eval()
    with torch.no_grad():
        print("help")