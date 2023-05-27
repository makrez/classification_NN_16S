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
from sequence_utils import hot_dna

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('create_dataset.log')
handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)

msa_file_path = '../data/Ngt10_filtered_Actinovacteria_SILVA_138.1_SSURef_tax_silva_full_align_trunc.fasta'
alignment_length = 50000
taxonomy_level = 5  # specify the taxonomy level you want to train on
dataset_dir = './datasets'
dataset_filename = 'actinobacteria_sequence_dataset.pkl'

os.makedirs(dataset_dir, exist_ok=True)


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


# First pass: compute labels and their frequencies
taxonomy_labels = []

with open(msa_file_path) as handle:
    for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
        label = str(record.description)
        encoded_dna = hot_dna(str(record.seq)[:alignment_length], label)
        if len(encoded_dna.onehot) == alignment_length:
            taxonomy_labels.append(encoded_dna.taxonomy)

        if i % 1000 == 0:  # log every 1000 sequences
            logging.info(f'Processed {i} sequences')

taxonomy_labels_level = [labels[taxonomy_level] for labels in taxonomy_labels]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(taxonomy_labels_level)

# Compute train/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_index, test_index = next(sss.split(np.zeros_like(encoded_labels), encoded_labels))

# Compute train/validation split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_index, valid_index = next(sss.split(np.zeros_like(encoded_labels[train_index]), encoded_labels[train_index]))

# Second pass: process sequence data in chunks and split into train/test/valid sets
sequences_train = []
sequences_valid = []
sequences_test = []
labels_train = []
labels_valid = []
labels_test = []

with open(msa_file_path) as handle:
    for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
        label = str(record.description)
        encoded_dna = hot_dna(str(record.seq)[:alignment_length], label)
        if len(encoded_dna.onehot) == alignment_length:
            sequence_tensor = torch.tensor(encoded_dna.onehot).float()

            if i in train_index:
                sequences_train.append(sequence_tensor)
                labels_train.append(encoded_dna.taxonomy)
            elif i in valid_index:
                sequences_valid.append(sequence_tensor)
                labels_valid.append(encoded_dna.taxonomy)
            elif i in test_index:
                sequences_test.append(sequence_tensor)
                labels_test.append(encoded_dna.taxonomy)

        if i % 1000 == 0:  # log every 1000 sequences
            logging.info(f'Processed {i} sequences')

# Create train/test/valid datasets
train_dataset = SequenceDataset(pad_sequence(sequences_train, batch_first=True), labels_train)
valid_dataset = SequenceDataset(pad_sequence(sequences_valid, batch_first=True), labels_valid)
test_dataset = SequenceDataset(pad_sequence(sequences_test, batch_first=True), labels_test)

# Save datasets
with open(os.path.join(dataset_dir, dataset_filename), 'wb') as f:
    pickle.dump((train_dataset, valid_dataset, test_dataset, label_encoder.classes_), f)
