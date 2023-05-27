import os
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import ParameterGrid
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import logging
import json
from models import ConvClassifier
from sequence_utils import hot_dna
from plotting_functions import plot_training_results
from training_functions import train_network


# Set path and parameters
msa_file_path = '../data/bacillus.aln'
alignment_length = 4500

# Set up a logger
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Define SequenceDataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, index):
        return {
            "sequence": self.sequences[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.sequences)

# Read and preprocess data
sequences = []
taxonomy_labels = []

with open(msa_file_path) as handle:
    for record in SeqIO.parse(handle, 'fasta'):
        label = str(record.description)  # corrected this line
        encoded_dna = hot_dna(str(record.seq)[10:alignment_length+10], label)
        if len(encoded_dna.onehot) == alignment_length:
            sequences.append(torch.tensor(encoded_dna.onehot).float())
            taxonomy_labels.append(encoded_dna.taxonomy)

# Convert the list of sequences into a tensor
sequences_tensor = pad_sequence(sequences, batch_first=True)

# Create LabelEncoder and encode labels at specified taxonomy level
taxonomy_level = 5  # specify the taxonomy level you want to train on
taxonomy_labels_level = [labels[taxonomy_level] for labels in taxonomy_labels]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(taxonomy_labels_level)
num_classes = len(set(encoded_labels))

# Split the data into training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(sequences_tensor, encoded_labels):
    X_train, X_test = sequences_tensor[train_index], sequences_tensor[test_index]
    y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

# Create separate datasets and dataloaders for training and test sets
train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the device, model and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvClassifier(input_length=alignment_length, num_classes=num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()

# Define the hyperparameters
param_grid = {
    'lr': [0.01,0.001, 0.0001, 0.00001],
    'n_epoch': [100, 500, 1000, 2000]
}

grid = ParameterGrid(param_grid)

# To store results
results = []

for params in grid:
    # Reset the model and optimizer
    model = ConvClassifier(alignment_length, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Create a subdirectory for this set of parameters
    directory = f"lr={params['lr']}_n_epoch={params['n_epoch']}"
    os.makedirs(directory, exist_ok=True)

    logging.info(f"Starting training with lr={params['lr']}, n_epoch={params['n_epoch']}")
    print(f"Training with lr={params['lr']}, n_epoch={params['n_epoch']}")

    train_losses, test_losses = train_network(params['n_epoch'], model, optimizer, criterion, train_dataloader, test_dataloader, device, os.path.join(directory, 'model.pth'), directory)    
    plot_training_results(model, train_losses, test_losses, test_dataloader, device, label_encoder, directory) 
    # Save results
    results.append({
        'lr': params['lr'],
        'n_epoch': params['n_epoch'],
        'train_loss': train_losses[-1],  # Final training loss
        'test_loss': test_losses[-1],  # Final test loss
    })

    # Log the results
    logging.info(f"lr={params['lr']}, n_epoch={params['n_epoch']}, train_loss={train_losses[-1]}, test_loss={test_losses[-1]}")

    # Save the results to a JSON file in the subdirectory
    with open(os.path.join(directory, 'results.json'), 'w') as f:
        json.dump(results[-1], f)

# Save the overall results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f)

# Print the results
for result in results:
    print(f"lr={result['lr']}, n_epoch={result['n_epoch']}, train_loss={result['train_loss']}, test_loss={result['test_loss']}")
