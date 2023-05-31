import numpy as np
import torch
import pickle

def test_load_tensor():
    dataset_dir = './datasets'
    tensor_index = 2590  # Specify the index of the tensor you want to load

    # Load the tensor
    sequence_tensor = torch.load(f'{dataset_dir}/train/{tensor_index}.pt')

    # Load the labels
    encoded_labels = pickle.load(open(f'{dataset_dir}/train/labels.pkl', 'rb'))
    full_taxonomy_labels = pickle.load(open(f'{dataset_dir}/train/full_labels.pkl', 'rb'))

    # Find the corresponding label for the tensor using original index
    encoded_label_data = next((x for x in encoded_labels if x[0] == tensor_index), None)
    full_taxonomy_label_data = next((x for x in full_taxonomy_labels if x[0] == tensor_index), None)

    if encoded_label_data is not None and full_taxonomy_label_data is not None:
        original_index = encoded_label_data[0]
        encoded_label = encoded_label_data[1]
        full_taxonomy_label = full_taxonomy_label_data[1]

        print(f'Sequence tensor:\n{sequence_tensor}')
        print(f'Encoded label: {encoded_label}')
        print(f'Full taxonomy label: {full_taxonomy_label}')
    else:
        print(f"No data found for tensor index {tensor_index}")

test_load_tensor()
