import os
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, ParameterGrid
import logging
import json
from models import ConvClassifier
import sequence_utils
from sequence_utils import SequenceDataset
from plotting_functions import plot_training_results
from training_functions import train_network

# Set path and parameters
msa_file_path = '../data/bacillus.aln'
alignment_length = 4500

# Set up a logger
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Read and summarize the MSA
sequence_utils.summarize_msa(msa_file_path)

### Read data
flatted_sequence = list()
sequence_labels = list()

with open(msa_file_path) as handle:
  for record in SeqIO.parse(handle, 'fasta'):
    label = str(record.description).rsplit(';', 1)[-1]
    seq_hot = sequence_utils.hot_dna(str(record.seq)[10:alignment_length+10]).onehot

    if len(seq_hot) == alignment_length:
      flatted_sequence.append(seq_hot)
      sequence_labels.append(label)

## Encode labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(sequence_labels)
num_classes = len(set(encoded_labels))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(flatted_sequence, encoded_labels, test_size=0.2)

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
    'n_epoch': [100, 500, 1000]
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
