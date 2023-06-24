import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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


class hot_dna:
    ### Class for One Hot Encoding
    def __init__(self, sequence, taxonomy):
        sequence = sequence.upper()
        self.sequence = self._preprocess_sequence(sequence)
        self.category_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, '-': 4, 'N': 5}
        self.onehot = self._onehot_encode(self.sequence)
        self.taxonomy = taxonomy.split(';')  # splitting by ';' to get each taxonomy level

    def _preprocess_sequence(self, sequence):
        ambiguous_bases = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', '.',}
        new_sequence = ""
        for base in sequence:
            if base in ambiguous_bases:
                new_sequence += 'N'
            else:
                new_sequence += base
        # replace sequences of four or more '-' characters with 'N' characters
        new_sequence = re.sub('(-{4,})', lambda m: 'N' * len(m.group(1)), new_sequence)
        return new_sequence

    def _onehot_encode(self, sequence):
        integer_encoded = np.array([self.category_mapping[char] for char in sequence]).reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # Fill missing channels with zeros
        full_onehot_encoded = np.zeros((len(sequence), 6))
        full_onehot_encoded[:, :onehot_encoded.shape[1]] = onehot_encoded

        return full_onehot_encoded