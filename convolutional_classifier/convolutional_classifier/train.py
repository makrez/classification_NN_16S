import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import ParameterGrid
import numpy as np
import logging
import json
from models import ConvClassifier, ConvClassifier2, ConvClassifierBacillus
from training_functions import train_network
import pickle
import sys
import torch.nn.functional as F
from model_summarise_functions import plot_confusion_matrix, plot_train_test_curves, print_f1_and_classification_report
import yaml
from dataset_classes import hot_dna

data_folder="/scratch/mk_cas/datasets"
batch_size = 32
alignment_length = 50000

class SequenceDataset(Dataset):
    def __init__(self, folder_path):
        self.pt_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]

        # Load labels into a dictionary
        with open(os.path.join(folder_path, 'labels.pkl'), 'rb') as f:
            labels = pickle.load(f)
        self.labels = {label['sequence_id']: label['label'] for label in labels}

        self.folder_path = folder_path

    def __getitem__(self, index):
        sequence_file = self.pt_files[index]
        sequence_id = os.path.splitext(sequence_file)[0]
        sequence_dict = torch.load(os.path.join(self.folder_path, sequence_file))
        sequence = sequence_dict['sequence_tensor']
        label = torch.tensor(self.labels[sequence_id], dtype=torch.long)
        return {'sequence': sequence, 'label': label}

    def __len__(self):
        return len(self.pt_files)

# Create datasets
train_dataset = SequenceDataset(os.path.join(data_folder, 'train'))
test_dataset = SequenceDataset(os.path.join(data_folder, 'test'))

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
num_classes = len(set(train_dataset.labels.values()))

model = ConvClassifier2(input_length=alignment_length, num_classes=num_classes)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

criterion = nn.CrossEntropyLoss()

# Load the configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the hyperparameters from the config file
lr_list = config['lr']
n_epoch_list = config['n_epoch']

# Define the hyperparameters
param_grid = {
    'lr': lr_list,
    'n_epoch': n_epoch_list
}

grid = ParameterGrid(param_grid)

# To store results
results = []

for params in grid:
    
    # Create a subdirectory for this set of parameters
    directory = f"lr={params['lr']}_n_epoch={params['n_epoch']}"
    os.makedirs(directory, exist_ok=True)
    
    # Create a subdirectory for model evaluation
    model_evaluation_dir = os.path.join(directory, 'model_evaluation')
    os.makedirs(model_evaluation_dir, exist_ok=True)
    
    # Set up a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(os.path.join(directory, 'training.log'))
    handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    # Reset the model and optimizer
    model = ConvClassifier2(alignment_length, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    logging.info(f"Starting training with lr={params['lr']}, n_epoch={params['n_epoch']}")
    print(f"Training with lr={params['lr']}, n_epoch={params['n_epoch']}")

    train_losses, test_losses, y_true, y_pred = train_network(params['n_epoch'], model, optimizer, criterion, train_dataloader, test_dataloader, device, directory, logger)

    print(y_true)
    print(y_pred)
    # Save losses and predictions for later analysis
    np.save(os.path.join(model_evaluation_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(model_evaluation_dir, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(model_evaluation_dir, 'y_true.npy'), np.array(y_true))
    np.save(os.path.join(model_evaluation_dir, 'y_pred.npy'), np.array(y_pred))

    # Log the results
    logging.info(f"lr={params['lr']}, n_epoch={params['n_epoch']}, train_loss={train_losses[-1]}, test_loss={test_losses[-1]}")

    # Save results
    results.append({
        'lr': params['lr'],
        'n_epoch': params['n_epoch'],
        'train_loss': train_losses[-1],  # Final training loss
        'test_loss': test_losses[-1],  # Final test loss
        'model_architecture' : str(model),
        'batch_size': batch_size,
    })

    # Save the results to a JSON file in the model_evaluation subdirectory
    with open(os.path.join(model_evaluation_dir, 'results.json'), 'w') as f:
        json.dump(results[-1], f)
    
    torch.save(model.state_dict(), os.path.join(directory, 'model_weights.pt'))