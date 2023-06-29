import time
import torch
import sys
import json
import logging
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import csv
from torch.utils.data import Dataset
import numpy as np
import os

def train_network(n_epochs, model, optimizer, criterion, train_dataloader, test_dataloader, device, params_save_path, logger):

    # Save all parameters
    params = {
        "n_epochs": n_epochs,
        "optimizer": str(optimizer),
        "criterion": str(criterion),
        "train_dataloader_length": len(train_dataloader.dataset),
        "test_dataloader_length": len(test_dataloader.dataset),
        "device": str(device),
    }

    train_losses = []
    test_losses = []
    start_time = time.time()

    for epoch in range(1, n_epochs+1):
        epoch_start_time = time.time()

        # Training phase
        train_loss = 0.0
        model.train()
        for i, batch in enumerate(train_dataloader, 1):
            sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(sequence_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequence_data.size(0)

            # Log the progress
            curr_time = time.time()
            elapsed_time = curr_time - epoch_start_time
            processed_sequences = i * batch['sequence'].size(0)
            percentage_complete = (processed_sequences / len(train_dataloader.dataset)) * 100
            logger.info(f"Epoch {epoch}/{n_epochs} | Processed {processed_sequences}/{len(train_dataloader.dataset)} sequences ({percentage_complete:.2f}%) | Training Loss: {train_loss/processed_sequences:.6f} | Elapsed Time: {elapsed_time:.2f}s")

        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # Test phase
        test_loss = 0.0
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader, 1):
                sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
                labels = batch["label"].to(device)
            
                outputs = model(sequence_data)
                _, preds = torch.max(outputs, 1)

                y_pred.extend(preds.tolist())
                y_true.extend(labels.tolist())

                loss = criterion(outputs, labels)
                test_loss += loss.item() * sequence_data.size(0)

        test_loss /= len(test_dataloader.dataset)
        test_losses.append(test_loss)

        correct_preds = torch.eq(torch.max(F.softmax(outputs, dim=-1), dim=-1)[1], labels).float().sum()
        params['correct_preds'] = correct_preds.item()
        total_preds = torch.FloatTensor([labels.size(0)])
        params['total_preds'] = total_preds.item()
        correct_preds = correct_preds.to(device)
        total_preds = total_preds.to(device)
        accuracy = correct_preds / total_preds
        params['accuracy'] = accuracy.item()
        logger.info(f'Accuracy: {accuracy.item():.4f}')

        with open(f"{params_save_path}/parameters.json", 'w') as f:
            json.dump(params, f)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time

        logger.info(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTest Loss: {test_loss:.6f} \tEpoch Time: {epoch_duration:.2f}s \tTotal Time: {total_duration:.2f}s")

    # Save the final model
    torch.save(model.state_dict(), f"{params_save_path}/final_model.pt")

    total_duration = time.time() - start_time
    logger.info(f"Total training time: {total_duration:.2f}s")

    return train_losses, test_losses, y_true, y_pred

# Define the function for taxonomy filtering
def filter_by_taxonomy(full_labels, taxonomic_level, taxonomic_group, 
                       classification_level, minimum_samples):
    taxonomic_levels = ['ncbi_identifier', 'Domain', 'Kingdom', 'Phylum', 'Order', 'Family', 'Genus', 'Species']

    # Make sure provided taxonomic_level is valid
    assert taxonomic_level in taxonomic_levels, f"Invalid taxonomic_level. Must be one of {taxonomic_levels}"
    assert classification_level in taxonomic_levels, f"Invalid classification_level. Must be one of {taxonomic_levels}"
    
    level_index = taxonomic_levels.index(taxonomic_level)
    filtered_labels = []
    
    for label in full_labels:
        current_label = label['label'][level_index]
        if current_label == taxonomic_group:
            filtered_labels.append(label)
    
    classification_level_index = taxonomic_levels.index(classification_level)
    labels_counter = Counter([label['label'][classification_level_index] for label in filtered_labels])
    labels_with_minimum_samples = [label for label in filtered_labels if labels_counter[label['label'][classification_level_index]] >= minimum_samples]
    
    # Count the occurrences of the classification levels in the filtered labels
    classification_counts = Counter([label['label'][classification_level_index] for label in labels_with_minimum_samples])
    num_classes = len(classification_counts)
    
    return labels_with_minimum_samples, labels_counter, classification_counts, num_classes


def encode_labels(labels, classification_level):
    le = LabelEncoder()
    
    taxonomic_levels = ['ncbi_identifier', 'Domain', 'Kingdom', 'Phylum', 'Order', 'Family', 'Genus', 'Species']
    classification_level_index = taxonomic_levels.index(classification_level)
    
    original_labels = [label['label'][classification_level_index] for label in labels]
    encoded_labels = le.fit_transform(original_labels) 
    
    label_map = {original: encoded for original, encoded in zip(original_labels, encoded_labels)}
    encoded_labels_dict = [{'sequence_id': label['sequence_id'], 'label': label_map[label['label'][classification_level_index]]} for label in labels]
    return encoded_labels_dict, label_map, le


def split_data(encoded_labels_dict):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    indices = np.arange(len(encoded_labels_dict))
    labels = [label_dict['label'] for label_dict in encoded_labels_dict]
    for train_index, temp_index in sss.split(indices, labels):
        train_data = [encoded_labels_dict[i] for i in train_index]
        temp_data = [encoded_labels_dict[i] for i in temp_index]
        
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    temp_labels = [label_dict['label'] for label_dict in temp_data]
    for valid_index, test_index in sss.split(np.arange(len(temp_data)), temp_labels):
        valid_data = [temp_data[i] for i in valid_index]
        test_data = [temp_data[i] for i in test_index]
    
    return train_data, valid_data, test_data

class SequenceDataset(Dataset):
    def __init__(self, folder_path, data):
        # Construct full file paths
        self.pt_files = [os.path.join(folder_path, str(f['sequence_id']) + '.pt') for f in data]

        # Use provided labels
        self.labels = {label_dict['sequence_id']: label_dict['label'] for label_dict in data}

    def __getitem__(self, index):
        sequence_file = self.pt_files[index]
        sequence_id = os.path.splitext(os.path.basename(sequence_file))[0]  # Get the sequence_id from the file name
        sequence_dict = torch.load(sequence_file)
        sequence = sequence_dict['sequence_tensor']
        label = torch.tensor(self.labels[sequence_id], dtype=torch.long)
        return {'sequence': sequence, 'label': label}

    def __len__(self):
        return len(self.pt_files)
    
def write_counts_to_csv(counts, filepath):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Classification Level", "Count"])
        for key, count in counts.items():
            writer.writerow([key, count])