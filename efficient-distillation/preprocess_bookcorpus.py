from transformers import BertTokenizerFast
from datasets import load_dataset

dataset = load_dataset("bookcorpus")
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
train_dataset = dataset['train']
    
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, padding="max_length")
encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_train_dataset.save_to_disk("/scratch/tli104/bookcorpus_preprocessed")


