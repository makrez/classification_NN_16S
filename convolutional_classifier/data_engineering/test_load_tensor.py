import os
import pickle
import torch
from Bio.Seq import Seq
from dataset_classes import hot_dna

taxonomy_level = 6

def load_sequence_and_labels(sequence_id, data_path):
    # Load the LabelEncoder and LabelMap
    label_encoder = pickle.load(open(os.path.join(data_path, 'label_encoder.pkl'), 'rb'))
    label_map = pickle.load(open(os.path.join(data_path, 'label_map.pkl'), 'rb'))

    inverse_label_map = {v: k for k, v in label_map.items()}

    # Load sequence tensor data
    sequence_data = torch.load(os.path.join(data_path, f'{sequence_id}.pt'))
    sequence_tensor = sequence_data["sequence_tensor"]

    # Initialize hot_dna class with empty sequence and taxonomy (for using the _onehot_decode method)
    dna_decoder = hot_dna('', '')

    # Decode the sequence tensor to the original sequence
    original_sequence = dna_decoder._onehot_decode(sequence_tensor)

    # Load the labels and full labels
    labels = pickle.load(open(os.path.join(data_path, 'labels.pkl'), 'rb'))
    full_labels = pickle.load(open(os.path.join(data_path, 'full_labels.pkl'), 'rb'))

    # Find the entry with the matching sequence_id
    label_entry = next((item for item in labels if item["sequence_id"] == sequence_id), None)
    full_label_entry = next((item for item in full_labels if item["sequence_id"] == sequence_id), None)

    # If found, extract the actual label and full label
    if label_entry is not None:
        encoded_label = label_entry["label"]
        original_label = inverse_label_map[encoded_label]
    else:
        encoded_label = None
        original_label = None

    if full_label_entry is not None:
        full_label = full_label_entry["label"]
        taxonomy_of_encoded_label = full_label[taxonomy_level]
    else:
        full_label = None
        taxonomy_of_encoded_label = None

    return sequence_tensor, original_sequence, encoded_label, original_label, full_label, taxonomy_of_encoded_label

# Load the sequence tensor, original sequence, encoded label, original label, full taxonomy, 
# and taxonomy of the encoded label for the tensor at index 5 in the training set

sequence_id = "1022"
sequence_tensor, original_sequence, encoded_label, original_label, full_label, taxonomy_of_encoded_label = \
     load_sequence_and_labels(sequence_id, './datasets_bacillus/train')

print(f"Encoded sequence: {sequence_tensor}")
print(f"Original sequence: {original_sequence}")
print(f"Encoded label: {encoded_label}")
print(f"Original label: {original_label}")
print(f"Full taxonomy: {full_label}")
print(f"Taxonomy of the encoded label: {taxonomy_of_encoded_label}")
