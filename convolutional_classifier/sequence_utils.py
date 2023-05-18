from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Bio import AlignIO
import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset


def summarize_msa(file_path, num_lines_to_display=5):
    alignment = AlignIO.read(file_path, 'fasta')
    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()

    # Count occurrences of each base and gaps
    counts = Counter()
    total_bases = num_sequences * alignment_length
    for record in alignment:
        counts.update(record.seq)

    # Calculate the percentages
    percentages = {base: (count / total_bases) * 100 for base, count in counts.items()}

    print(f"Number of sequences: {num_sequences}")
    print(f"Alignment length: {alignment_length}")
    print("\nPercentages of bases and gaps:")
    for base, percentage in percentages.items():
        if percentage > 0:
            print(f"{base}: {percentage:.2f}%")

    print("\nFirst few lines of the alignment:")
    for idx, record in enumerate(alignment):
        if idx < num_lines_to_display:
             print(f"{record.id}: {str(record.seq)[:100]}")
        else:
            break

class hot_dna:
    ### Class for One Hot Encoding
    def __init__(self, sequence):
        sequence = sequence.upper()
        self.sequence = self._preprocess_sequence(sequence)
        self.category_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, '-': 4, 'N': 5}
        self.onehot = self._onehot_encode(self.sequence)

    def _preprocess_sequence(self, sequence):
        ambiguous_bases = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'}
        new_sequence = ""
        for base in sequence:
            if base in ambiguous_bases:
                new_sequence += 'N'
            else:
                new_sequence += base
        return new_sequence

    def _onehot_encode(self, sequence):
        integer_encoded = np.array([self.category_mapping[char] for char in sequence]).reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # Fill missing channels with zeros
        full_onehot_encoded = np.zeros((len(sequence), 6))
        full_onehot_encoded[:, :onehot_encoded.shape[1]] = onehot_encoded

        return full_onehot_encoded

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, index):
        return {
            "sequence": torch.tensor(self.sequences[index]).float(),
            "label": torch.tensor(self.labels[index]).long(),
        }

    def __len__(self):
        return len(self.sequences)