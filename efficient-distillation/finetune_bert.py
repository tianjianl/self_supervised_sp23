import time
import torch
import random
import wandb
import argparse
import numpy as np
import evaluate
import datetime
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel
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

    def forward(self, input_ids, attention_mask, student_layer):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states

        assert torch.equal(hidden_states[-1], output.last_hidden_state)
        mean_t = torch.mean(hidden_states[-1], dim=1)
        mean_s = torch.mean(hidden_states[student_layer], dim=1)

        teacher_logits = self.classifier_t(mean_t)
        student_logits = self.classifier_s(mean_s)

        return teacher_logits, student_logits

    def get_symm_kl(self, logits_a, input_b):
        return (
            F.kl_div(
                F.log_softmax(logits_a, dim=-1, dtype=torch.float32),
                F.softmax(logits_b, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(logits_a, dim=-1, dtype=torch.float32),
                F.softmax(logits_b, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / noised_logits.size(0)

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


param_importance_by_layer = [[] for _ in range(24)]

def generate_binary_outcomes(probabilities):
    
    logits = torch.rand(len(probabilities))
    probs = probabilities.clone().detach()
    mask = logits >= probs
    return mask.to('cuda')

def weighted_dropout(params, probs):
    
    """
    returns a params tensor that is the original value or 0 depending on its probability 
    """
    param_shape = params.shape
    params = params.clone().view(-1)
    mask = generate_binary_outcomes(probs)
    params = params * mask
    return params.view(param_shape)

def train(epoch, tokenizer, model, device, loader, optimizer, scheduler=None, regularizer=None, param_importance_dict=None, weighted_dropout_iters=0):
    
    model.train()
    start = time.time()
    loss_fn = nn.NLLLoss()
    
    #initializing parameter importance dictionary
    for iteration, data in enumerate(loader, 0):
        
        x = data['source_ids'].to(device, dtype = torch.long)
        x_mask = data['source_mask'].to(device, dtype = torch.long)
        y = data['label'].to(device)
        
        output = model(input_ids = x, attention_mask = x_mask)
        y_hat = output.logits
        y_hat = F.log_softmax(y_hat, dim=1)
        loss = loss_fn(y_hat, y)
        
        if regularizer != None:
            loss += regularizer.penalty(model, input_ids = x, attention_mask = x_mask) 
        if iteration%50 == 0:
            wandb.log({"Training Loss": loss.item()})
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if weighted_dropout_iters == 0:
            continue

        if iteration % weighted_dropout_iters == 0:
            for name, params in model.named_parameters():
                if 'embeddings' in name:
                    continue
                    
                if params.requires_grad:
                    grad = params.grad.clone().detach().view(-1)
                    p = params.clone().detach().view(-1)
                    scores = torch.abs(grad*p)
                    scores = scores.to("cpu")
                    scaled_scores = scores - scores.min()
                    scaled_scores /= scores.max()
                    #perform weighted dropout, the probability that a certain parameter is masked is larger if it is more important
                    params = weighted_dropout(params, scaled_scores)
        """   
        if iteration%200 == 0 and param_importance_dict != None:
            
            importances = [[] for _ in range(24)]
            #update param_importance_dict
            for name, params in model.named_parameters():
                layer_num = name.split('.')
                if len(layer_num) <= 3 or layer_num[3].isnumeric() == False:
                    continue
                else:
                    layer_num = int(layer_num[3])
                if params.requires_grad:
                    grad = params.grad.clone().detach().view(-1)
                    params = params.clone().detach().view(-1)
                    score = torch.abs(grad*params)
                    score = score.to("cpu")
                    importances[layer_num - 1].extend(score.tolist())        
                    # mu = torch.mean(score)
                    # sigma = torch.std(score)
                    # param_importance_dict[name].append((mu.item(), sigma.item()))
            
            # temp = sorted(param_importance_dict.items(), key=lambda x:x[1][-1][0])
            
            # for item in temp:
            
            #    print(item[0])
            #    print(f"mean = {param_importance_dict[item[0]][-1][0]}")
            #    print(f"std = {param_importance_dict[item[0]][-1][1]}")
            
            normalized_importances = []          
            for i in range(1, 25):
                layer_importance = np.mean(importances[i-1])
                param_importance_by_layer[i-1].append(layer_importance)
                print(f"layer {i}, score {layer_importance}")
                normalized_importances.append(layer_importance)
            
            normalized_importances = torch.tensor(normalized_importances, device='cuda')
            normalized_importances -= normalized_importances.min()
            normalized_importances /= normalized_importances.max()
            normalized_importances *= 0.9
            
            for name, params in model.named_parameters():
                layer_num = name.split('.')
                if len(layer_num) <= 3 or layer_num[3].isnumeric() == False:
                    continue
                else:
                    layer_num = int(layer_num[3])
                if name == 'roberta.encoder.layer.1.output.LayerNorm.weight':
                    print("before dropout")
                    print(params[0:5])
                if params.requires_grad:
                    params = F.dropout(params, p=normalized_importances[layer_num-1].item())
                if name == 'roberta.encoder.layer.1.output.LayerNorm.weight':
                    print("after dropout")
                    print(params[0:5])
        """
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')
    
def validate(epoch, tokenizer, model, device, val_loader):
    
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

            output = model(input_ids = x, attention_mask = x_mask)
            y_hat = output.logits
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

label_dict = {'rte': 2, 'mrpc': 2, 'cola': 2, 'sst': 2, 'sts-b': 2, 'qnli': 2, 'qqp': 2}

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb_name = f"{args.lr}-{args.seed}-{args.task}-{args.regularizer}"
    if args.sage:
        wandb_name += '-sage'
    if args.weighted_dropout_iters > 0:
        wandb_name += f'-wd-{args.weighted_dropout_iters}'
    wandb.init(project=f"bert-distillation", entity="dogtooooth", name=wandb_name)

    num_labels = label_dict[args.task]
    model = BertForSequenceClassification(args.model_name, num_labels)
    model.to(device)
    
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    param_importance_dict = None
    if args.plot_params:
        print(f"--plot_params argument detected, printing the param importance while training")
        param_importance_dict = {n: [] for n, p in model.named_parameters() if p.requires_grad}
    
    if args.sage:
        print(f"--sage marker detected, using AdamW with lr adaptive to param importance")
        from bert_optim import UnstructAwareAdamW
        optimizer_parameters = get_usadam_param_groups(model)
        optimizer = UnstructAwareAdamW(params=optimizer_parameters, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr) 
    
    acc = evaluate.load("accuracy")
    loader_params = {
        'batch_size': args.bs,
        'shuffle': True,
        'num_workers': 0
    }
    
    train_dataset = data_to_df(task=args.task, split='train')
    
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    train_dataset = CustomClassificationDataset(train_dataset, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, **loader_params)
  
    val_dataset = data_to_df(task=args.task, split='dev')
    val_loader = DataLoader(val_dataset, **loader_params)

    regularizer = None
    if args.regularizer != None:
        regularizer = Regularizer(model = model, alpha = 1, dataset = train_loader, regularizer_type = args.regularizer)
    
    wandb.watch(model, log="all")
    if args.epoch == -1 or args.load_checkpoint:
        epoch = -1
        model.load_state_dict(torch.load(f"./{args.task}_latest.pth"))   
        print(f"loaded checkpoint at {args.task}_latest.pth")
        y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, val_loader)       
        result = acc.compute(references = y, predictions = y_hat)
        print(f"epoch = {epoch} | acc = {result['accuracy']}")
                   
    for epoch in range(args.epoch):
        train(epoch, tokenizer, model, device, train_loader, optimizer, param_importance_dict=param_importance_dict, wd_iter=args.weighted_dropout_iters)
        torch.save(model.state_dict(), f"{args.task}_latest.pth")
        
        y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, val_loader)       
        result = acc.compute(references = y, predictions = y_hat)
        print(f"epoch = {epoch} | acc = {result['accuracy']}")

    if args.plot_params:
        num_layers = 12 if 'base' in args.model_name else 24
        for i in range(num_layers):
            print(param_importance_by_layer[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic training arguments
    parser.add_argument("--load_checkpoint", action='store_true')
    parser.add_argument("--epoch", default=15, type=int)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--task", default='xnli')
    parser.add_argument("--seed", default=1104, type=int)
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--max_len", default=256, type=int)
    
    # regularizers, tricks, plots...
    parser.add_argument("--sage", action='store_true')
    parser.add_argument("--plot_params", action='store_true')
    parser.add_argument("--regularizer", default=None)
    parser.add_argument("--weighted_dropout_iters", type=int, help="interval for calculating weighted dropout")
    
    # distillation related
    parser.add_argument("--student_layer", defaut=8, type=int)
    args = parser.parse_args()
    
    assert args.model_name in ['bert-base-uncased', 'bert-large-uncased'], "This code base only support bert-base-uncased and bert-large-uncased"

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    main(args)
