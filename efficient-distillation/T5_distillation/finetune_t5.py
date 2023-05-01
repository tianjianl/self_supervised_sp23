# Importing stock libraries
import os
import sys
import json
import re
import torch
import time
import wandb
import evaluate
import argparse

import numpy as np
import pandas as pd
import torch.nn.functional as F

from datetime import date
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda

def add_sentinel_tokens(text, spaces=True):
    
    final = ""
    if spaces == False:
        text = " ".join(text)
    text = text.strip().split()
    
    for index, token in enumerate(text):
        final += f"<extra_id_{index}> " + token + " "
    
    return final

def remove_sentinel_tokens(text):
    
    return re.sub('<.*?>', '', text)

def init_data(args, split):

    src_doc = f"/scratch4/cs601/tli104/WikiLingua_data_splits/english/{split}.src.en"
    tgt_doc = f"/scratch4/cs601/tli104/WikiLingua_data_splits/english/{split}.tgt.en"
    src = []
    tgt = []
    f = open(src_doc, 'r')
    for line in f:
        src.append(line.strip())
    f.close()
    f = open(tgt_doc, 'r')
    for line in f:
        tgt.append(line.strip())
    f.close()
    assert len(src) == len(tgt), f"size of source and target mismatch"
    df = pd.DataFrame(list(zip(src, tgt)), columns=['src', 'tgt'])
    print(df.head(5))
    if split != 'train':
        return df.head(200)
    else:
        return df

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.src = self.data.src
        self.tgt = self.data.tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = str(self.src[index])
        src = ' '.join(src.split())

        tgt = str(self.tgt[index])
        tgt = ' '.join(tgt.split())

        source = self.tokenizer.batch_encode_plus([src], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([tgt], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

param_importance_dict = {'encoder': [[] for _ in range (24)], 'decoder':[[] for _ in range(24)]}
def train(epoch, tokenizer, model, device, loader, optimizer, task='tg', val_loader=None):

    start = time.time()
    for iteration, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        
        if iteration%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        if iteration%50 == 0:
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')
        if iteration%100 == 0 and val_loader != None:
            #printing the validation loss to a file
            _ = validate(1, tokenizer, model, device, val_loader, get_val_loss=True, task=task)
        

        optimizer.zero_grad()
        loss.backward()
        
        # Getting parameter importance by layer
        if iteration % 500 == 0:
            for n, p in model.named_parameters():
                n = n.split('.')
                if len(n) >= 4:
                    component = n[0]
                    layer = int(n[2])
                else:
                    continue
                grad = p.grad.detach().clone()
                params = p.detach().clone()
                scores = torch.abs(grad*params)
                scores = scores.view(-1)
                scores = scores.to('cpu')

                param_importance_dict[component][layer].append(scores.tolist())
                

            for comp in ['encoder', 'decoder']:
                for i in range(24):
                    print(f"{iteration} | {comp} | {i} | {np.mean(param_importance_dict[comp][i])}")
                    param_importance_dict[comp][i] = []
        optimizer.step()
        end = time.time()
    
    print(f'Epoch: {epoch} used {end-start} seconds')

def validate(epoch, tokenizer, model, device, loader, get_val_loss=False):
    
    """
    If get_val_loss is set to be True, then this function returns the average validation loss
    """

    model.eval()
    predictions = []
    actuals = []
    start = time.time()
    with torch.no_grad():
        loss_total = []
        for index, data in enumerate(loader, 0):
            
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            
            if get_val_loss == True:
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
                val_loss = outputs[0]
                loss_total.append(val_loss.item())
                continue

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=64, 
                num_beams=4,
                length_penalty=0.6,
            )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            #printing sample outputs
            if index == 0 or index == 1 or index == 2:
                print(f"pred = {''.join(remove_sentinel_tokens(preds[0]))}")
                print(f"target = {''.join(remove_sentinel_tokens(target[0]))}")
            if index%100==0:
                now = time.time()
                print(f'evaluation used {now-start} seconds')
                print(f'Completed {index}')

            predictions.extend(preds)
            actuals.extend(target)
    
    if get_val_loss:
        f = open(f'./dev-loss-cnndm.txt', 'a+')
        print(f"{np.mean(loss_total)}", file=f)
        print(f"Dev set loss = {np.mean(loss_total)}")
        return np.mean(loss_total)
    
    return predictions, actuals

def evaluate_func(epoch, predictions, actuals, tokenizer, model, prev_best):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    preds = [' '.join(str(encoding) for encoding in tokenizer.encode(remove_sentinel_tokens(pred))) for pred in predictions]
    acts = [' '.join(str(encoding) for encoding in tokenizer.encode(remove_sentinel_tokens(act))) for act in actuals]
    r1 = []
    r2 = []
    rl = []
    for pred, target in zip(preds, acts):
        scores = scorer.score(pred, target)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeLsum'].fmeasure)
            
    if np.mean(rl) > prev_best:
        print(f"found best validation rouge-l score {np.mean(r2)} at epoch {epoch}")
        prev_best = np.mean(rl)
        torch.save(model.state_dict(), f"./t5_cnndm_best.pth")
        print(f"epoch = {epoch}|r1 = {np.mean(r1)} r2 = {np.mean(r2)} rl = {np.mean(rl)}")
    
    return prev_best

def freeze_encoder(model):
    print("Freezing the encoder parameters")
    for name, param in model.named_parameters():
        if 'encoder' in name and param.requires_grad:
            param.requires_grad = False

def main(args):

    global config
    device = 'cuda' if cuda.is_available() else 'cpu'
    wandb.init(project=f"t5_large_{args.task}", entity="dogtooooth", name=f"{args.lr}-{args.seed}-{args.task}")
    
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    config = wandb.config
    # Defining some key variablces that will be used later on in the training
    config.TRAIN_BATCH_SIZE = args.bs   # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = args.bs   # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = args.epoch    # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = args.lr   
    config.SEED = args.seed           
    config.MAX_LEN = args.input_len
    config.SUMMARY_LEN = args.output_len
    #Setting finetuned model path
    
    print(f"lr = {config.LEARNING_RATE}")

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
     
    train_dataset = init_data(args, 'train')
    val_dataset = init_data(args, 'val') 
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    model.to(device)
    
    if args.freeze:
        freeze_encoder(model)  

    if config.TRAIN_EPOCHS == 0:
        model.load_state_dict(torch.load(f"./{args.task}_best.pth"))   
        print(f"loaded checkpoint at {args.task}_best.pth")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.LEARNING_RATE) 
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    
    wandb.watch(model, log="all")
    # Training loop
    prev_best = 0.0
    
    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        _ = evaluate_func(epoch, predictions, actuals, tokenizer, model)
        final_df = pd.DataFrame({'Generated Text': predictions,'Actual Text': actuals})
        print(final_df.head(5))
    
    if config.TRAIN_EPOCHS != 0:
        model.load_state_dict(torch.load(f"./{args.task}_best.pth"))   
        print(f"loaded checkpoint at {args.task}_best.pth")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #basic arguments
    parser.add_argument("--epoch", default=10, type=int, help="the training epochs")
    parser.add_argument("--lr", default=0.000001, type=float, help="learning rate")
    parser.add_argument("--bs", default=8, type=int, help="batch size")
    parser.add_argument("--task", default='pos', help="the task you want to finetune on")
    parser.add_argument("--output_len", default=64, type=int, help="the ouptut length")
    parser.add_argument("--input_len", default=512, type=int, help="the input length")
    parser.add_argument("--seed", type=int, default=1005, help="the random seed")
   
    #prefix tuning specific arguments
    parser.add_argument("--freeze", dest="freeze", action="store_true")
    parser.add_argument("--print_val_loss", dest="print_val_loss", action="store_true")
    
    args = parser.parse_args()
    main(args)
