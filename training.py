import time
import torch
import spacy
import config
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from IMDBDataset import IMDBDataset, IMDBCollate
from sklearn.model_selection import train_test_split
from model import RNN, LSTM
from evaluate import compute_accuracy

## Config Variables
RANDOM_SEED = config.RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
LEARNING_RATE = config.LEARNING_RATE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = config.EMBEDDING_DIM
HIDDEN_DIM = config.HIDDEN_DIM
NUM_CLASSES = config.NUM_CLASSES
NUM_LAYERS = config.NUM_LAYERS

# Get the tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Load Dataset
df = pd.read_csv('movie_data.csv')

# split the dataset into train, test and validation sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Create the dataset
train_dataset = IMDBDataset(train_df, tokenizer, min_freq=5, vocabulary=None)
val_dataset = IMDBDataset(val_df, tokenizer, min_freq=None, vocabulary=train_dataset.vocab)
test_dataset = IMDBDataset(test_df, tokenizer, min_freq=None, vocabulary=train_dataset.vocab)

# Initialize Vocabulary Size
VOCABULARY_SIZE = train_dataset.vocab.__len__()

# Create DataLoader
train_dataloader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True,
                              collate_fn=IMDBCollate(pad_idx=train_dataset.vocab['<PAD>']), 
                              num_workers=multiprocessing.cpu_count())
val_dataloader = DataLoader(val_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False, 
                            collate_fn=IMDBCollate(pad_idx=train_dataset.vocab['<PAD>']), 
                            num_workers=multiprocessing.cpu_count())
test_dataloader = DataLoader(test_dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=False, 
                             collate_fn=IMDBCollate(pad_idx=train_dataset.vocab['<PAD>']), 
                             num_workers=multiprocessing.cpu_count())

model = RNN(input_dim=VOCABULARY_SIZE,
            embedding_dim=EMBEDDING_DIM, 
            hidden_dim=HIDDEN_DIM, 
            output_dim=NUM_CLASSES, 
            num_layers=NUM_LAYERS)

model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

def train():
    start_time = time.time()
    training_accuracy = '0.00%'
    validation_accuracy = "0.00%"

    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                text, labels = batch_data
                text = text.to(DEVICE)
                labels = labels.to(DEVICE)
                ### FORWARD AND BACK PROP
                logits = model(text)
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()                
                loss.backward()                
                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), training_accuracy=training_accuracy, validation_accuracy=validation_accuracy)

        training_accuracy = f'{compute_accuracy(model, train_dataloader, DEVICE):.2f}%'
        validation_accuracy = f'{compute_accuracy(model, val_dataloader, DEVICE):.2f}%'
    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
    print(f'Test accuracy: {compute_accuracy(model, test_dataloader, DEVICE):.2f}%')

if __name__ == "__main__":
    train()