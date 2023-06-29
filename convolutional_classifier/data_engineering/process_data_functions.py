from Bio import SeqIO
import torch
import os
import pickle
from dataset_classes import hot_dna

def load_sequences(msa_file_path, alignment_length):
    full_taxonomy_labels = []
    original_indices = []  # To keep track of original indices

    with open(msa_file_path) as handle:
        for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
            # Modified the splitting operation
            first_label, *rest_labels = record.description.split(';')
            rest_labels[0:0] = first_label.split()
            label = rest_labels

            if len(str(record.seq)[:alignment_length]) == alignment_length:
                full_taxonomy_labels.append(label)
                original_indices.append(i)

    return full_taxonomy_labels, original_indices


def process_sequences(msa_file_path, alignment_length, sequence_path, full_taxonomy_labels, original_indices):
    full_labels = []

    with open(msa_file_path) as handle:
        for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
            if i not in original_indices:
                continue

            if len(str(record.seq)[:alignment_length]) == alignment_length:
                encoded_dna = hot_dna(str(record.seq)[:alignment_length], record.description)
                sequence_tensor = torch.tensor(encoded_dna.onehot).float()

                original_index = original_indices.index(i)

                sequence_id = f"{original_index}"  # Unique identifier for each sequence
                torch.save({"sequence_id": sequence_id, "sequence_tensor": sequence_tensor}, f'{sequence_path}/{sequence_id}.pt')  # Save with sequence_id
                full_labels.append({"sequence_id": sequence_id, "label": full_taxonomy_labels[original_index]})

    pickle.dump(full_labels, open(f'{sequence_path}/full_labels.pkl', 'wb'))
