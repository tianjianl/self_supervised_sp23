import pandas as pd
import json
from glob import glob

def data_to_df(task, language, split):
    
    if task == 'cola':
        filename = f"/scratch/tli104/glue_data/CoLA/{split}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for line in f:
            cols = line.strip().split('\t')
            label = int(cols[1])
            context = cols[-1]
            src.append(context)
            labels.append(label)
        df = pd.DataFrame({"src": src, "label": labels})
        #df = df.sample(frac=1, ignore_index=True)
        return df

    elif task == 'qnli' or task == 'rte':
        task = task.upper()
        filename = f"/scratch/tli104/glue_data/{task}/{split}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for index, line in enumerate(f):
            if index == 0:
                continue
            cols = line.strip().split('\t')
            question = cols[1]
            context = cols[2]
            label = 1 if cols[3] == 'entailment' else 0
            src.append(context + " " + question)
            labels.append(label)

        df = pd.DataFrame({"src": src, "label": labels})
        return df
    
    elif task == 'mnli':
        task = task.upper()
        if split == 'dev' or split == 'test':
            split = split + '_matched'

        filename = f"glue_data/{task}/{split}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for index, line in enumerate(f):
            if index == 0:
                continue
            cols = line.strip().split('\t')
            question = cols[8]
            context = cols[9]
            if cols[11] == 'entailment': 
                label = 1
            elif cols[11] == 'contradiction':
                label = 0
            else:
                label = 2

            src.append(context + " " + question)
            labels.append(label)

        df = pd.DataFrame({"src": src, "label": labels})
        return df
     
    elif task == 'qqp':
        filename = f"/scratch/tli104/glue_data/QQP/{split}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for index, line in enumerate(f):
            if index == 0:
                continue
            cols = line.strip().split('\t')
            question = cols[3]
            context = cols[4]
            label = int(cols[5])
            src.append(context + " " + question)
            labels.append(label)

        df = pd.DataFrame({"src": src, "label": labels})
        return df
    
    elif task == 'sst-2':
        filename = f"/scratch/tli104/glue_data/SST-2/{split}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for index, line in enumerate(f):
            if index == 0:
                continue
            cols = line.strip().split('\t')
            sentence = cols[0]
            label = int(cols[1])
            src.append(sentence)
            labels.append(label)
        
        df = pd.DataFrame({"src": src, "label": labels})
        return df 
    
    elif task == 'sts-b':
        filename = f"glue_data/STS-B/{split}.tsv"
        f = open(filename, 'r')
        s1 = []
        s2 = []
        scores = []

        for index, line in enumerate(f):
            if index == 0:
                continue
            cols = line.strip().split('\t')
            s1.append(cols[7])
            s2.append(cols[8])
            scores.append(cols[9])

        df = pd.Dataframe({"s1": s1, "s2": s2, "score": scores})
        return df

    elif task == 'mrpc':
        if split != 'train':
            split = 'test'
        filename = f"/scratch/tli104/glue_data/MRPC/msr_paraphrase_{split}.txt"
        f = open(filename, 'r')
        src = []
        labels = []
        for index, line in enumerate(f):
            if index == 0:
                continue 
            cols = line.strip().split('\t')
            label = int(cols[0])
            sent1 = cols[3]
            sent2 = cols[4]
            src.append(sent1 + " " + sent2)
            labels.append(label)
        
        df = pd.DataFrame({"src": src, "label": labels})
        return df
            
    elif task == "xnli":
        filename = f"../download/xnli/{split}-{language}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for line in f:
            cols = line.strip().split('\t')
            premise = cols[0]
            hypothesis = cols[1]
            if cols[2] == 'neutral':
                label = 0
            elif cols[2] == 'entailment':
                label = 1
            elif cols[2] == 'contradiction':
                label = 2
                
            src.append(premise + " " + hypothesis)
            labels.append(label)

        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        print(df.head(5)) 
        return df
    
    elif task == 'tydiqa':
        src_doc = f"../download/tydiqa/tydiqa-goldp-v1.1-{split}/tydiqa.goldp.{language}.{split}.json"
        
        f = open(src_doc, 'r')
        src = []
        tgt = []
        dataset = json.load(f)['data']
        for paragraphs in dataset:
            for paragraph in paragraphs['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    src.append(context + " " + question)
                    answers = [answer['text'] for answer in qa['answers']]
                    answers_indices = [(answer['answer_start'], answer['answer_start'] + len(answer['text'].split(' '))- 1) for answer in qa['answers']]
                    tgt.append(answers_indices[0])

        df = pd.DataFrame({"src": src, "label": tgt})
        print(df.head(5))
        return df
    
    elif task == 'pawsx':
        filename = f"../download/pawsx/{split}-{language}.tsv"
        f = open(filename, 'r')
        src = []
        labels = []
        for line in f:
            cols = line.strip().split('\t')
            src.append(cols[0] + ' ' + cols[1])
            labels.append(int(cols[2]))
        
        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        print(df.head(5))
        return df
    
    elif task == 'ape':
        mt_filenames = glob('../ape/*/*.mt')
        pe_filenames = glob('../ape/*/*.pe')
        src = []
        labels = []
        for mt_file in mt_filenames:
            if split not in mt_file:
                continue
            f = open(mt_file, 'r')
            for line in f:
                line = line.strip()
                src.append(line)
                labels.append(0)

        for pe_file in pe_filenames:
            if split not in pe_file:
                continue
            f = open(pe_file, 'r')
            for line in f:
                line = line.strip()
                src.append(line)
                labels.append(1)

        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        print(df.head(5))
        return df.head(20000)

    elif task == 'td':
        src = []
        labels = []
        f = open('../de_original', 'r')
        for line in f:
            line = line.strip().split('.')
            for sentence in line:
                if len(sentence) <= 16:
                    continue
                src.append(sentence + ' . ')
                labels.append(0)
            
            if len(labels) >= 50000:
                break
        
        f = open('../de_translated_from_en', 'r')
        for line in f:
            line = line.strip()
            src.append(line)
            labels.append(1)
            if len(labels) >= 100000:
                break
       
        if split == 'train':
            src = src[:80000]
            labels = labels[:80000]
        else:
            src = src[80000:]
            labels = labels[80000:]
        print(split)
        print(len(src))
        print(len(labels))
        
        df = pd.DataFrame({"src": src, "label": labels})
        df = df.sample(frac=1, ignore_index=True)
        return df
