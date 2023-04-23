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
from transformers import BertModel, BertTokenizer
from init_data import data_to_df 

from rotk import Regularizer

class BertForSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        if 'small' in model_name:
            hidden_size = 512
        if 'base' in model_name:
            hidden_size = 768
        if 'large' in model_name:
            hidden_size = 1024
        self.classifier_t = nn.Linear(hidden_size, num_labels)
        self.classifier_s = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, teacher_layer, student_layer):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states

        assert torch.equal(hidden_states[-1], output.last_hidden_state)
        
        mean_t = torch.mean(hidden_states[teacher_layer-1], dim=1)
        mean_s = torch.mean(hidden_states[student_layer-1], dim=1)

        teacher_logits = self.classifier_t(mean_t)
        student_logits = self.classifier_s(mean_s)

        return teacher_logits, student_logits

class CustomClassificationDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.input = self.data.src
        self.label = self.data.label

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        src = str(self.input[index])
        src = ' '.join(src.split())
        source = self.tokenizer.batch_encode_plus([src], max_length=self.max_len, padding="max_length", truncation=True, return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        
        # list of numeric labels 
        tgt = self.label[index]

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'label': tgt
        }

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

def train(epoch, tokenizer, model, device, loader, optimizer, args, teacher_layer, student_layer):
    
    model.train()
    start = time.time()
    loss_fn = nn.NLLLoss()

    #initializing parameter importance dictionary
    for iteration, data in enumerate(loader, 0):
        
        x = data['source_ids'].to(device, dtype=torch.long)
        x_mask = data['source_mask'].to(device, dtype=torch.long)
        y = data['label'].to(device)
        
        output, output_student = model(input_ids=x, attention_mask=x_mask, teacher_layer=teacher_layer, student_layer=student_layer)
        y_hat = output        
        y_hat = F.log_softmax(y_hat, dim=1)
        loss = loss_fn(y_hat, y)
        
        if args.use_sd:           
            #self-distillation: symmetric kl divergence between teacher and student logits
            loss += args.sd_alpha*get_symm_kl(output, output_student) 
        
        if args.use_id:
            output_2, output_student_2 = model(input_ids=x, attention_mask=x_mask, teacher_layer=teacher_layer, student_layer=student_layer)
            loss += args.id_alpha*get_symm_kl(output, output_2)
        
        if iteration%200 == 0:
            wandb.log({"Training Loss": loss.item()})
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward() # backprop gradient 
        optimizer.step() # update parameters
        
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')
    
def validate(epoch, tokenizer, model, device, val_loader, teacher_layer, student_layer, student=False):

    model.eval()
    predictions = []
    actuals = []
    total_dev_loss = []
    loss_fn = nn.NLLLoss()
    with torch.no_grad():
        for index, data in enumerate(val_loader, 0):
            
            x = data['source_ids'].to(device, dtype = torch.long)
            x_mask = data['source_mask'].to(device, dtype = torch.long)
            y = data['label'].to(device, dtype = torch.long)

            output, output_student = model(input_ids = x, attention_mask = x_mask, teacher_layer=teacher_layer, student_layer=student_layer)
            if student:
                y_hat = output_student
            else:
                y_hat = output            
            y_hat = F.log_softmax(y_hat, dim=1)
            loss = loss_fn(y_hat, y)
            
            predictions.extend(torch.argmax(y_hat, dim=1))
            actuals.extend(y)
            total_dev_loss.append(loss.item())
    
    #print(predictions[:5])
    #print(actuals[:5])
    #print(f'evaluation used {now-start} seconds')
    loss = np.mean(total_dev_loss)
    return predictions, actuals, loss


def get_usadam_param_groups(model):
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    mask = ['attention.self', 'attention.output.dense', 'output.dense', 'intermediate.dense']

    mask_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in mask)]
    common_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in mask)]

    optimizer_parameters = [
        {'params': [p for n, p in common_params if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01,
         'params_type': 'common'},
        {'params': [p for n, p in mask_params if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01,
         'params_type': 'mask'},
        {'params': [p for n, p in common_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0,
         'params_type': 'common'},
        {'params': [p for n, p in mask_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0,
         'params_type': 'mask'},
        ]
    return optimizer_parameters

label_dict = {'rte': 2, 'mrpc': 2, 'cola': 2, 'sst-2': 2, 'sts-b': 2, 'qnli': 2, 'qqp': 2}

def main(args):

    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb_name = f"{args.lr}-{args.seed}-{args.task}-{args.student_layer}-{args.sd_alpha}"
    if args.sage:
        wandb_name += '-sage'
    if args.weighted_dropout_iters > 0:
        wandb_name += f'-wd-{args.weighted_dropout_iters}'
    wandb.init(project=f"bert-gradual", entity="dogtooooth", name=wandb_name)

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
 
    num_labels = label_dict[args.task]
    model = BertForSequenceClassification(args.model_name, num_labels)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    wandb.watch(model, log="all")
   
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr) 
    
    acc = evaluate.load("accuracy")
    matthews_metric = evaluate.load("matthews_correlation")
    f1_metric = evaluate.load("f1")

    loader_params = {
        'batch_size': args.bs,
        'shuffle': True,
        'num_workers': 0
    }
    
    train_dataset = data_to_df(task=args.task, language='en', split='train')
    
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    train_dataset = CustomClassificationDataset(train_dataset, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, **loader_params)
  
    val_dataset = data_to_df(task=args.task, language='en', split='dev')
    val_dataset = CustomClassificationDataset(val_dataset, tokenizer, args.max_len)
    val_loader = DataLoader(val_dataset, **loader_params)
    
    if args.regularizer != None:
        regularizer = Regularizer(model = model, alpha = 1, dataset = train_loader, regularizer_type = args.regularizer)
    
    if args.epoch == -1 or args.load_checkpoint:
        epoch = -1
        model.load_state_dict(torch.load(f"./{args.task}_latest.pth"))   
        print(f"loaded checkpoint at {args.task}_latest.pth")
        y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, val_loader)       
        if args.task == 'cola':
            result = matthews_metric.compute(references=y, predictions=y_hat) 
            print(f"epoch = {epoch} | mcc = {result['matthews_correlation']}")
        
        if args.task == 'mrpc' or args.task == 'qqp':
            f1_result = f1_metric.compute(references=y, predictions=y_hat)
            print(f"epoch = {epoch} | f1 = {f1_result['f1']}")

        result = acc.compute(references = y, predictions = y_hat)
        print(f"epoch = {epoch} | acc = {result['accuracy']}")
    
    
    schedule = []
    if args.gradual: # this overrides args.student_layer 
        seq = args.schedule.split(',')
        for index, layer in enumerate(seq):
            if index != 0:
                schedule.append((int(seq[index-1]), int(seq[index])))
                                   
        print(f"prune schedule = {schedule}")
        for teacher_layer, student_layer in schedule:
            for epoch in range(args.epoch):
                train(epoch, tokenizer, model, device, train_loader, optimizer, args, teacher_layer=teacher_layer, student_layer=student_layer)
                torch.save(model.state_dict(), f"/scratch4/cs601/tli104/checkpoints/{args.task}_gradual_{student_layer}.pth")
                y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, 
                                              val_loader, teacher_layer=teacher_layer, student_layer=student_layer, student=True)
                if args.task == 'cola':
                    result = matthews_metric.compute(references=y, predictions=y_hat) 
                    print(f"epoch = {epoch} | mcc = {result['matthews_correlation']}")
                if args.task == 'mrpc' or args.task == 'qqp':
                    f1_result = f1_metric.compute(references=y, predictions=y_hat)
                    print(f"epoch = {epoch} | f1 = {f1_result['f1']}")
                result = acc.compute(references = y, predictions = y_hat)
                print(f"epoch = {epoch} | student = {student_layer} | acc = {result['accuracy']}")
    else:
        for epoch in range(args.epoch):
            teacher_layer = 12 if args.model_name == 'bert-base-uncased' else 24
            train(epoch, tokenizer, model, device, train_loader, optimizer, args, teacher_layer=teacher_layer, student_layer=args.student_layer)
            torch.save(model.state_dict(), f"/scratch4/cs601/tli104/checkpoints/{args.task}_latest.pth")
            
            flag = True if args.use_sd else False
            y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, val_loader, teacher_layer=teacher_layer, student_layer=args.student_layer, 
                                        student=flag)       
            
            if args.task == 'cola':
                result = matthews_metric.compute(references=y, predictions=y_hat) 
                print(f"epoch = {epoch} | mcc = {result['matthews_correlation']}")

            if args.task == 'mrpc' or args.task == 'qqp':
                f1_result = f1_metric.compute(references=y, predictions=y_hat)
                print(f"epoch = {epoch} | f1 = {f1_result['f1']}")

            result = acc.compute(references = y, predictions = y_hat)
            print(f"epoch = {epoch} | acc = {result['accuracy']}")
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic training arguments
    parser.add_argument("--load_checkpoint", action='store_true')
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--task", default='xnli')
    parser.add_argument("--seed", default=1104, type=int)
    parser.add_argument("--model_name", default='bert-large-uncased')
    parser.add_argument("--max_len", default=256, type=int)

    # regularizers, tricks, plots...
    parser.add_argument("--sage", action='store_true')
    parser.add_argument("--regularizer", default=None)
    parser.add_argument("--weighted_dropout_iters", type=int, default=0, help="interval for calculating weighted dropout")
    
    # distillation related
    parser.add_argument("--use_sd", action='store_true', help="using teacher student self distillation")
    parser.add_argument("--sd_alpha", type=float, default=0.5, help="self-distillation loss scale")
    parser.add_argument("--use_id", action='store_true', help='using intra-distillation in teacher and student')
    parser.add_argument("--id_alpha", type=float, default=0.5, help="intra-distillation scale")
    parser.add_argument("--student_layer", default=8, type=int)
    parser.add_argument("--gradual", action='store_true', help="gradually prune the top layers of teacher: 24 - 20 - 16 - 12")
    parser.add_argument("--schedule", default='24,20,16,12')
    args = parser.parse_args()
    
    assert args.model_name in ['bert-base-uncased', 'bert-large-uncased'], "This code base only support bert-base-uncased and bert-large-uncased"

    main(args)
