from torch.utils.data import Dataset
import torch
import pickle

class SequenceDataset(Dataset):
    def __init__(self, sequence_paths, labels):
        self.sequence_paths = sequence_paths
        self.labels = labels

    def __getitem__(self, index):
        sequence_path = self.sequence_paths[index]
        sequence = torch.load(sequence_path)
        label = self.labels[index]
        return {
            "sequence": sequence,
            "label": label,
        }

    def __len__(self):
        return len(self.sequence_paths)

# Load datasets
with open('./datasets/actinobacteria_sequence_dataset.pkl', 'rb') as f:
    train_dataset, valid_dataset, test_dataset, label_classes = pickle.load(f)

# Print the first item in the train_dataset
print(train_dataset[0])
