import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn 
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification, T5Model
import argparse
import subprocess
import matplotlib.pyplot as plt


class T5ForSequenceClassification(nn.Module):

    def __init__(self, t5model, num_labels=2):
        super(T5ForSequenceClassification, self).__init__()
        self.model = T5Model.from_pretrained(t5model)
        if 'base' in t5model:
            hidden_dim = 768
        elif 'small' in t5model:
            hidden_dim = 512
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        last_hidden_state = output.last_hidden_state
   scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, call loss.backward() to compute the gradients.
            Then, call optimizer.step()  to update the model parameters.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            groundtruth = batch['labels'].to(device)
            
            if 't5' in model_name:
                predictions = mymodel(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
            else:
                output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
                predictions = output.logits
            
            model_loss = loss(predictions, groundtruth)
            model_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = torch.argmax(predictions, dim=1)

            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=batch['labels'])
            
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        train_acc = train_accuracy.compute()
        print(f" - Average training metrics: accuracy={train_acc}")
        train_acc_list.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device, model_name)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
        test_accuracy = evaluate_model(mymodel, test_dataloader, device, model_name)
        print(f" - Average test metrics: accuracy={test_accuracy}")
    
    """
    plt.plot(epoch_list, train_acc_list, 'b', label='train')
    plt.plot(epoch_list, dev_acc_list, 'g', label='valid')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/home/tli104/self_supervised_sp23/hw6/overfitting.pdf')
    """

def pre_process(model_name, batch_size, device, small_subset):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        dataset_dev_subset = dataset['train'][:10]
        dataset_test_subset = dataset['train'][:10]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:8000]
        dataset_dev_subset = dataset['validation']
        dataset_test_subset = dataset['train'][8000:]

    print("Size of the loaded dataset:")
    print(f" - train: {len(dataset_train_subset['passage'])}")
    print(f" - dev: {len(dataset_dev_subset['passage'])}")
    print(f" - test: {len(dataset_test_subset['passage'])}")

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    if 't5' in model_name:
        pretrained_model = T5ForSequenceClassification(model_name, num_labels=2)
    else:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"

    # load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset)
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, args.device, args.lr, args.model)
    
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device, args.model)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device, args.model)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
