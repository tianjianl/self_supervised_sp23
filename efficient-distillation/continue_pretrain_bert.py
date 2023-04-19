# Code for continue pretraining BERT for MLM on both the teacher and the student

import time
import torch
import random
import wandb
import argparse
import numpy as np
import evaluate
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, DataCollatorForLanguageModeling
from init_data import data_to_df 
from rotk import Regularizer

class BertForMaskedLanguageModeling(nn.Module):

    def __init__(self, model_name, num_vocab, student_layer):
        super(BertForSequenceClassification, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        if 'small' in model_name:
            hidden_size = 512
        if 'base' in model_name:
            hidden_size = 768
        if 'large' in model_name:
            hidden_size = 1024
        self.classifier_t = nn.Linear(hidden_size, num_vocab)
        self.classifier_s = nn.Linear(hidden_size, num_vocab)
        self.student_layer = student_layer
    
    def forward(self, input_ids, attention_mask):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states

        assert torch.equal(hidden_states[-1], output.last_hidden_state)
        mean_t = torch.mean(hidden_states[-1], dim=1)
        mean_s = torch.mean(hidden_states[self.student_layer], dim=1)
        teacher_logits = self.classifier_t(mean_t)
        student_logits = self.classifier_s(mean_s)
        
        return teacher_logits, student_logits

def get_symm_kl(logits_a, logits_b):
    return (F.kl_div(
                F.log_softmax(logits_a, dim=-1, dtype=torch.float32),
                F.softmax(logits_b, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(logits_b, dim=-1, dtype=torch.float32),
                F.softmax(logits_a, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / logits_a.size(0)


class CustomLanguageModelingDataset(Dataset):

def train(epoch, tokenizer, model, device, loader, optimizer, args, scheduler=None, regularizer=None):
    
    model.train()
    start = time.time()
    loss_fn = nn.NLLLoss()
    wd_iter = args.weighted_dropout_iters

    #initializing parameter importance dictionary
    for iteration, data in enumerate(loader, 0):
        
        x = data['source_ids'].to(device, dtype=torch.long)
        x_mask = data['source_mask'].to(device, dtype=torch.long)
        y = data['label'].to(device)
        
        output, output_student = model(input_ids=x, attention_mask=x_mask)
        y_hat = output        
        y_hat = F.log_softmax(y_hat, dim=1)
        loss = loss_fn(y_hat, y)
        
        if args.use_sd:           
            #self-distillation: symmetric kl divergence between teacher and student logits
            loss += args.sd_alpha*get_symm_kl(output, output_student) 
        
        if iteration%50 == 0:
            wandb.log({"Training Loss": loss.item()})
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward() # backprop gradient 
        optimizer.step() # update parameters
        
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')

def valid(tokenizer, model, device, val_loader):
    
    model.eval()
    start = time.time()
    loss_fn = nn.NLLLoss()
    
    teacher_mlm_loss = []
    student_mlm_loss = []

    for iteration, data in enumerate(val_loader, 0):
        
        x = data['source_ids'].to(device, dtype=torch.long)
        x_mask = data['source_mask'].to(device, dtype=torch.long)
        y = data['label'].to(device)
        
        output, output_student = model(input_ids=x, attention_mask=x_mask)
        y_hat = output        
        y_hat = F.log_softmax(y_hat, dim=1)
        loss = loss_fn(y_hat, y)
        
        y_hat = output_student 
        y_hat = F.log_softmax(y_hat, dim=1)
        loss_student = loss_fn(y_hat, y)
        
        
    end = time.time()
    print(f'validation used {end-start} seconds')
    

    teacher_ppl = torch.exp(torch.stack(teacher_mlm_loss).mean())
    student_ppl = torch.exp(torch.stack(student_mlm_loss).mean())
    return teacher_ppl, student_ppl

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb_name = f"{args.lr}-{args.seed}-{args.student_layer}-{args.sd_alpha}"
    wandb.init(project=f"bert-continue-pretraine", entity="dogtooooth", name=wandb_name)

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
 
    num_labels = label_dict[args.task]
    model = BertForMaskedLanguageModeling(args.model_name, num_labels, args.student_layer)
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    wandb.watch(model, log="all")
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr) 
    
    train_loader = 
    for epoch in range(args.epoch):
        train(epoch, tokenizer, model, device, train_loader, optimizer, args)
        teacher_valid_ppl, student_val_ppl = validate(tokenizer, model, device, val_loader)
        print(f"teacher val ppl {teacher_val_ppl} | student val ppl {student_val_ppl}")

    model.push_to_hub("bert-large-bookcorpus")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic training arguments
    parser.add_argument("--epoch", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--seed", default=1104, type=int)
    
    # distillation related
    parser.add_argument("--use_sd", action='store_true', help="using teacher student self distillation")
    parser.add_argument("--sd_alpha", type=float, default=0.5, help="self-distillation loss scale")
    parser.add_argument("--student_layer", default=8, type=int)
    args = parser.parse_args()

    main(args)
