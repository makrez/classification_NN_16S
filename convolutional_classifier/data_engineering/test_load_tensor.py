import os
import pickle
import torch
from Bio.Seq import Seq
from dataset_classes import hot_dna

taxonomy_level = 6

def load_sequence_and_labels(index, data_path):
    # Load the LabelEncoder and LabelMap
    label_encoder = pickle.load(open(f'{data_path}/label_encoder.pkl', 'rb'))
    label_map = pickle.load(open(f'{data_path}/label_map.pkl', 'rb'))

    inverse_label_map = {v: k for k, v in label_map.items()}

    sequence_tensor = torch.load(f'{data_path}/{index}.pt')[1458:1468]

    # Initialize hot_dna class with empty sequence and taxonomy (for using the _onehot_decode method)
    dna_decoder = hot_dna('', '')

    # Decode the sequence tensor to the original sequence
    original_sequence = dna_decoder._onehot_decode(sequence_tensor)[1458:1468]

    # Load the labels and full labels
    labels = pickle.load(open(f'{data_path}/labels.pkl', 'rb'))
    full_labels = pickle.load(open(f'{data_path}/full_labels.pkl', 'rb'))

    # Find the entry with the matching index
    label_entry = next((item for item in labels if item[0] == index), None)
    full_label_entry = next((item for item in full_labels if item[0] == index), None)

    # If found, extract the actual label and full label
    if label_entry is not None:
        encoded_label = label_entry[1]
        original_label = label_encoder.inverse_transform([encoded_label])[0]
    else:
        encoded_label = None
        original_label = None

    if full_label_entry is not None:
        full_label = full_label_entry[1]
        taxonomy_of_encoded_label = full_label[taxonomy_level]
    else:
        full_label = None
        taxonomy_of_encoded_label = None

    return sequence_tensor, original_sequence, encoded_label, original_label, full_label, taxonomy_of_encoded_label

# Load the sequence tensor, original sequence, encoded label, original label, full taxonomy, 
# and taxonomy of the encoded label for the tensor at index 5 in the training set

sequence_tensor, original_sequence, encoded_label, original_label, full_label, taxonomy_of_encoded_label = \
     load_sequence_and_labels(1022, './datasets_bacillus/train')

print(f"Encoded sequence: {sequence_tensor}")
print(f"Original sequence: {original_sequence}")
print(f"Encoded label: {encoded_label}")
print(f"Original label: {original_label}")
print(f"Full taxonomy: {full_label}")
print(f"Taxonomy of the encoded label: {taxonomy_of_encoded_label}")
