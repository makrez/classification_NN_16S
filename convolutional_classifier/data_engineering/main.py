import os
import logging
from preprocessing import load_sequences, encode_labels, split_data, process_sequences
from dataset import SequenceDataset, hot_dna
import re

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
msa_file_path = '../../data/Actinobacteria_10000_seqs.fasta'
alignment_length = 50000
taxonomy_level = 5  # specify the taxonomy level you want to train on
dataset_dir = './datasets'

# Ensure dataset directory exists
os.makedirs(dataset_dir, exist_ok=True)

taxonomy_labels, full_taxonomy_labels, class_counts, original_indices = load_sequences(msa_file_path, alignment_length, taxonomy_level)
print(taxonomy_labels[0])
encoded_labels = encode_labels(taxonomy_labels)
print(encoded_labels[0])
train_index, valid_index, test_index = split_data(encoded_labels[0])

# Create paths for your train/validation/test sequence tensors
train_path = os.path.join(dataset_dir, 'train')
valid_path = os.path.join(dataset_dir, 'valid')
test_path = os.path.join(dataset_dir, 'test')

# Make directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

process_sequences(msa_file_path, alignment_length, taxonomy_level, class_counts, train_index, valid_index, test_index, train_path, valid_path, test_path, full_taxonomy_labels, original_indices)
