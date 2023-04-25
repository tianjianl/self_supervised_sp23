# Code for continue pretraining BERT for MLM on both the teacher and the student

import time
import torch
import random
import wandb
import argparse
import numpy as np
import datetime
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, DataCollatorForLanguageModeling
from init_data import data_to_df 
from rotk import Regularizer

from datasets import load_dataset

class BertForMaskedLanguageModeling(nn.Module):

    def __init__(self, model_name, num_vocab, student_layer):
        super(BertForMaskedLanguageModeling, self).__init__()
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
        
        # when attention_mask is [1, 1, 0, 1, 1, 0, 1], it means that the second and fifth positions need to be predicted 

        hidden_states = output.hidden_states
        
        assert torch.equal(hidden_states[-1], output.last_hidden_state)
        
        # hidden_states[layer_num] is a tensor with shape (batch_size, max_length, hidden_size)
        # we project hidden_states[layer_num] to (batch_size, max_length, vocabulary_size)
        # then we only select the masked indices in the second dimension to calculate the loss 
        
        teacher_logits = self.classifier_t(hidden_states[-1])
        student_logits = self.classifier_s(hidden_states[self.student_layer])
        
        
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

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], max_length = 128, truncation=True, padding=True)

def train(epoch, tokenizer, model, loader, optimizer, accelerator):
    
    model.train()
    start = time.time()
    loss_fct = nn.CrossEntropyLoss()
    

    #initializing parameter importance dictionary
    for iteration, data in enumerate(loader, 0):
        
        print(data)
        x = data['source_ids']
        x_mask = data['source_mask']
        y = data['label']
        
        output, output_student = model(input_ids=x, attention_mask=x_mask)
        labels = torch.where(x == tokenizer.mask_token_id, y, -100)
        loss = loss_fct(output.view(-1, output.size(-1)), labels.view(-1))
        loss += loss_fct(output_student.view(-1, output_student.size(-1)), labels.view(-1))
        
        #if args.use_sd:           
        #    #self-distillation: symmetric kl divergence between teacher and student logits
        #    loss += args.sd_alpha * get_symm_kl(output.view(-1), output_student.view(-1))

            
        if iteration%50 == 0:
            wandb.log({"Training Loss": loss.item()})
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step() # update parameters
        
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')

def validate(tokenizer, model, val_loader):
    
    model.eval()
    start = time.time()
    loss_fn = nn.NLLLoss()
    
    teacher_mlm_loss = []
    student_mlm_loss = []

    for _, data in enumerate(val_loader, 0):
        
        x = data['source_ids']
        x_mask = data['source_mask']
        y = data['label']
        
        output, output_student = model(input_ids=x, attention_mask=x_mask)
        labels = torch.where(x == tokenizer.mask_token_id, y, -100)
        loss_teacher = loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
        loss_student = loss_fn(output_student.view(-1, output_student.size(-1)), labels.view(-1))
    
        teacher_mlm_loss.append(loss_teacher)
        student_mlm_loss.append(loss_student)
        
        
    end = time.time()
    print(f'validation used {end-start} seconds')
    

    teacher_ppl = torch.exp(torch.stack(teacher_mlm_loss).mean())
    student_ppl = torch.exp(torch.stack(student_mlm_loss).mean())
    return teacher_ppl, student_ppl

def main(args):
    
    wandb_name = f"{args.lr}-{args.seed}-{args.student_layer}"
    wandb.init(project=f"bert-continue-pretrain", entity="dogtooooth", name=wandb_name)
    accelerator = Accelerator()
    device = accelerator.device
    
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    num_vocab = 30522
    model_name = 'bert-large-uncased'
    model = BertForMaskedLanguageModeling(model_name, num_vocab, args.student_layer)
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)

    wandb.watch(model, log="all")
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    dataset = load_dataset("bookcorpus")
    
    train_dataset = dataset['train']
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length = 512, truncation=True, padding=True)

    encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr) 
    
    train_loader = DataLoader(encoded_train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
    

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    for epoch in range(args.epoch):
        train(epoch, tokenizer, model, train_loader, optimizer, accelerator)
        print(f"teacher val ppl {teacher_valid_ppl} | student val ppl {student_val_ppl}")
        torch.save(model.state_dict(), f"/bert-checkpoints/bert-large-bookcorpus-{epoch}.pt")
    model.push_to_hub("bert-large-bookcorpus")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic training arguments
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--seed", default=1104, type=int)
    
    # distillation related arguments
    parser.add_argument("--student_layer", default=8, type=int)
    args = parser.parse_args()

    main(args)
