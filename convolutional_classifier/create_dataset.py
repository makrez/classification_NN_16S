import os
import torch
import numpy as np
import pickle
import logging
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from dataset import one_hot, 
import pandas as pd
import ast
import json

# Define logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler('create_dataset.log')
handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console_handler)

# Define parameters
msa_file_path = '../data/Actinobacteria_10000_seqs.fasta'
alignment_length = 50000
taxonomy_level = 5  # specify the taxonomy level you want to train on
dataset_dir = './datasets'

# Ensure dataset directory exists
os.makedirs(dataset_dir, exist_ok=True)

# Helper function to get list of file paths in a directory
def get_file_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Define SequenceDataset
class SequenceDataset(Dataset):
    def __init__(self, sequence_paths, labels):
        self.sequence_paths = sequence_paths
        self.labels = labels

    def __getitem__(self, index):
        sequence = torch.load(self.sequence_paths[index])
        label = self.labels[index]
        return {
            "sequence": sequence,
            "label": label,
        }

    def __len__(self):
        return len(self.sequence_paths)

# First pass: compute labels and their frequencies
taxonomy_labels = []
full_taxonomy_labels = []
class_counts = {}
with open(msa_file_path) as handle:
    for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
        label = record.description.split(';')  # labels are separated by ';'
        if len(str(record.seq)[:alignment_length]) == alignment_length and len(label) > taxonomy_level:
            current_label = label[taxonomy_level]
            if current_label in class_counts:
                class_counts[current_label] += 1
            else:
                class_counts[current_label] = 1
            
            # Only add labels with count greater than 1 to taxonomy_labels and full_taxonomy_labels
            if class_counts[current_label] > 1:
                taxonomy_labels.append(current_label)
                full_taxonomy_labels.append(label)

        if i % 1000 == 0:  # log every 1000 sequences
            logging.info(f'Processed {i} sequences')

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(taxonomy_labels)

# Create a DataFrame for the class summary
class_summary = pd.DataFrame(label_encoder.classes_, columns=['Class'])
class_summary['Count'] = np.bincount(encoded_labels)
class_summary.to_csv('class_summary.csv', index=False)

# Compute train/test/valid split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42) # 30% for validation and test
train_index, temp_index = next(sss.split(np.zeros_like(encoded_labels), encoded_labels)) # temp is test+valid

# Then split the temp data between validation and test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) # 50% of 30% for validation and test each
valid_index, test_index = next(sss.split(np.zeros_like(encoded_labels[temp_index]), encoded_labels[temp_index]))

# Then you remap the indices to original
valid_index = temp_index[valid_index]
test_index = temp_index[test_index]

# Create paths for your train/validation/test sequence tensors
train_path = os.path.join(dataset_dir, 'train')
valid_path = os.path.join(dataset_dir, 'valid')
test_path = os.path.join(dataset_dir, 'test')

# Make directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Second pass: process sequence data in chunks and split into train/test/valid sets
labels_train = []
labels_valid = []
labels_test = []

full_labels_train = []
full_labels_valid = []
full_labels_test = []

train_counter = 0
valid_counter = 0
test_counter = 0

with open(msa_file_path) as handle:
    for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
        label = record.description.split(';')  # labels are separated by ';'
        if len(str(record.seq)[:alignment_length]) == alignment_length and len(label) > taxonomy_level:
            current_label = label[taxonomy_level]

            # Skip if this label only occurs once
            if class_counts[current_label] < 2:
                continue

            encoded_dna = hot_dna(str(record.seq)[:alignment_length], record.description) #label)
            sequence_tensor = torch.tensor(encoded_dna.onehot).float()

            if train_counter < len(train_index):
                torch.save(sequence_tensor, f'{train_path}/{train_counter}.pt')
                labels_train.append(encoded_dna.taxonomy)
                full_labels_train.append(full_taxonomy_labels[train_counter])
                train_counter += 1
            elif valid_counter < len(valid_index):
                torch.save(sequence_tensor, f'{valid_path}/{valid_counter}.pt')
                labels_valid.append(encoded_dna.taxonomy)
                full_labels_valid.append(full_taxonomy_labels[valid_counter])
                valid_counter += 1
            elif test_counter < len(test_index):
                torch.save(sequence_tensor, f'{test_path}/{test_counter}.pt')
                labels_test.append(encoded_dna.taxonomy)
                full_labels_test.append(full_taxonomy_labels[test_counter])
                test_counter += 1

        if i % 1000 == 0:  # log every 1000 sequences
            logging.info(f'Processed {i} sequences')    

# Save labels as .npy files for train/validation/test sets
np.save(f'{train_path}/labels.npy', np.array(labels_train))
np.save(f'{valid_path}/labels.npy', np.array(labels_valid))
np.save(f'{test_path}/labels.npy', np.array(labels_test))

# Save full labels as .npy files for train/validation/test sets
np.save(f'{train_path}/full_labels.npy', np.array(full_labels_train))
np.save(f'{valid_path}/full_labels.npy', np.array(full_labels_valid))
np.save(f'{test_path}/full_labels.npy', np.array(full_labels_test))


# # Load the sequences and labels as SequenceDatasets
# train_path = os.path.join(dataset_dir, 'train')
# valid_path = os.path.join(dataset_dir, 'valid')
# test_path = os.path.join(dataset_dir, 'test')

# train_files = get_file_paths(train_path)
# valid_files = get_file_paths(valid_path)
# test_files = get_file_paths(test_path)

# labels_train = np.load(os.path.join(dataset_dir, 'labels_train.npy'))
# labels_valid = np.load(os.path.join(dataset_dir, 'labels_valid.npy'))
# labels_test = np.load(os.path.join(dataset_dir, 'labels_test.npy'))

# train_dataset = SequenceDataset(train_files, labels_train)
# valid_dataset = SequenceDataset(valid_files, labels_valid)
# test_dataset = SequenceDataset(test_files, labels_test)
