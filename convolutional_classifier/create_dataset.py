import os
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
from sequence_utils import hot_dna
import logging

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

sequences = []
taxonomy_labels = []

with open(msa_file_path) as handle:
    for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
        label = str(record.description)
        encoded_dna = hot_dna(str(record.seq)[:alignment_length], label)
        if len(encoded_dna.onehot) == alignment_length:
            sequences.append(torch.tensor(encoded_dna.onehot).float())
            taxonomy_labels.append(encoded_dna.taxonomy)

        if i % 1000 == 0:  # log every 1000 sequences
            logging.info(f'Processed {i} sequences')

sequences_tensor = pad_sequence(sequences, batch_first=True)

taxonomy_labels_level = [labels[taxonomy_level] for labels in taxonomy_labels]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(taxonomy_labels_level)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(sequences_tensor, encoded_labels):
    X_train, X_test = sequences_tensor[train_index], sequences_tensor[test_index]
    y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

with open(os.path.join(dataset_dir, dataset_filename), 'wb') as f:
    pickle.dump((train_dataset, test_dataset, label_encoder.classes_), f)
