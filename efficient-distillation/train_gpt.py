import torch
from transformers import GPT2Model, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

class WikiDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(tokenizer.encode(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

def evaluate_perplexity(model, tokenizer, path):
    # Load validation dataset
    val_dataset = WikiDataset(path, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables
    total_loss = 0
    total_count = 0

    # Evaluate on validation dataset
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch[:, :-1].to(model.device)
            labels = batch[:, 1:].to(model.device)
            loss, *_ = model(input_ids, labels=labels)
            total_loss += loss.item() * labels.numel()
            total_count += labels.numel()

    # Calculate perplexity
    perplexity = torch.exp(total_loss / total_count)

    return perplexity

def main():
    
    # Define model and tokenizer
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Define dataset and dataloader
    train_dataset = WikiDataset("wiki.train.tokens", tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Define training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(3):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch[:, :-1].to(model.device)
            labels = batch[:, 1:].to(model.device)
            loss, *_ = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
        
        val_path = "wiki.valid.tokens"
        perplexity = evaluate_perplexity(model, tokenizer, val_path)
        print(f"Validiation Perplexity: {perplexity:.2f}")

    model.save_pretrained("fine-tuned-gpt2-wiki103")

if __name__ == '__main__':
    main()
