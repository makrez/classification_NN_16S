import os
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from dataset import SequenceDataset, hot_dna
import pandas as pd
import torch
import pickle
import re

def load_sequences(msa_file_path, alignment_length, taxonomy_level):
    taxonomy_labels = []
    full_taxonomy_labels = []
    class_counts = {}
    original_indices = []  # To keep track of original indices

    with open(msa_file_path) as handle:
        for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
            label = record.description.split(';')
            if len(str(record.seq)[:alignment_length]) == alignment_length and len(label) > taxonomy_level:
                current_label = label[taxonomy_level]
                if current_label in class_counts:
                    class_counts[current_label] += 1
                else:
                    class_counts[current_label] = 1
                
                taxonomy_labels.append(current_label)
                full_taxonomy_labels.append(label)
                original_indices.append(i)

    return taxonomy_labels, full_taxonomy_labels, class_counts, original_indices


def encode_labels(taxonomy_labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(taxonomy_labels)
    return encoded_labels, label_encoder

def split_data(encoded_labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_index, temp_index = next(sss.split(np.zeros_like(encoded_labels), encoded_labels))
    
    print(train_index[0])
    print(temp_index[0])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_index, test_index = next(sss.split(np.zeros_like(encoded_labels[temp_index]), encoded_labels[temp_index]))
    
    valid_index = temp_index[valid_index]
    test_index = temp_index[test_index]
    
    return train_index, valid_index, test_index


def process_sequences(msa_file_path, alignment_length, taxonomy_level, class_counts, train_index, valid_index, test_index, train_path, valid_path, test_path, full_taxonomy_labels, original_indices):
    labels_train = []
    labels_valid = []
    labels_test = []

    full_labels_train = []
    full_labels_valid = []
    full_labels_test = []

    # Create dictionaries mapping original index to shuffled index
    train_index_dict = {original_idx: i for i, original_idx in enumerate(train_index)}
    valid_index_dict = {original_idx: i for i, original_idx in enumerate(valid_index)}
    test_index_dict = {original_idx: i for i, original_idx in enumerate(test_index)}

    with open(msa_file_path) as handle:
        for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
            print(f"Processing record {i}: sequence {record.seq[:50]}, label {record.description.split(';')[taxonomy_level]}")
            if i not in original_indices:
                continue

            label = record.description.split(';')
            if len(str(record.seq)[:alignment_length]) == alignment_length and len(label) > taxonomy_level:
                current_label = label[taxonomy_level]

                encoded_dna = hot_dna(str(record.seq)[:alignment_length], record.description)
                sequence_tensor = torch.tensor(encoded_dna.onehot).float()

                original_index = original_indices.index(i)
                if original_index in train_index_dict:
                    print(f"Saving to train set: original index {i}, new index {train_index_dict[i]}")
                    torch.save(sequence_tensor, f'{train_path}/{train_index_dict[original_index]}.pt')
                    labels_train.append([original_index, encoded_dna.taxonomy])  # Append original index with label
                    full_labels_train.append([original_index, full_taxonomy_labels[original_index]])  # Append original index with label
                elif original_index in valid_index_dict:
                    print(f"Saving to validation set: original index {i}, new index {valid_index_dict[i]}")
                    torch.save(sequence_tensor, f'{valid_path}/{valid_index_dict[original_index]}.pt')
                    labels_valid.append([original_index, encoded_dna.taxonomy])
                    full_labels_valid.append([original_index, full_taxonomy_labels[original_index]])
                elif original_index in test_index_dict:
                    print(f"Saving to test set: original index {i}, new index {test_index_dict[i]}")
                    torch.save(sequence_tensor, f'{test_path}/{test_index_dict[original_index]}.pt')
                    labels_test.append([original_index, encoded_dna.taxonomy])
                    full_labels_test.append([original_index, full_taxonomy_labels[original_index]])

    pickle.dump(labels_train, open(f'{train_path}/labels.pkl', 'wb'))
    pickle.dump(labels_valid, open(f'{valid_path}/labels.pkl', 'wb'))
    pickle.dump(labels_test, open(f'{test_path}/labels.pkl', 'wb'))

    pickle.dump(full_labels_train, open(f'{train_path}/full_labels.pkl', 'wb'))
    pickle.dump(full_labels_valid, open(f'{valid_path}/full_labels.pkl', 'wb'))
    pickle.dump(full_labels_test, open(f'{test_path}/full_labels.pkl', 'wb'))

