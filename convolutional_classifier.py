import os
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Bio import AlignIO
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchviz import make_dot
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix


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

# Read and summarize the MSA
msa_file_path = 'data/bacillus.aln'
summarize_msa(msa_file_path)



### Class for One Hot Encoding

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class hot_dna:
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



### Read data

flatted_sequence = list()
sequence_labels = list()

alignment_length = 1500

with open('data/bacillus.aln') as handle:
  for record in SeqIO.parse(handle, 'fasta'):
    label = str(record.description).rsplit(';', 1)[-1]
    seq_hot = hot_dna(str(record.seq)[10:alignment_length+10]).onehot

    if len(seq_hot) == alignment_length:
      flatted_sequence.append(seq_hot)
      sequence_labels.append(label)



## Define the model

class ConvClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ConvClassifier, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )

        # Calculate the size of the final feature map after convolutional and pooling layers
        feature_map_size = input_length // (2 * 2 * 2)

        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * feature_map_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

## Change dataset to include labels

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

## Encode labels into integers

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(sequence_labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(flatted_sequence, encoded_labels, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(encoded_labels))
model = ConvClassifier(input_length=alignment_length, num_classes=num_classes)
model.to(device)

print(model)





# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(flatted_sequence, encoded_labels, test_size=0.2, random_state=42)

# Create separate datasets and dataloaders for training and test sets
train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(np.unique(encoded_labels))

criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 50

train_losses = []
test_losses = []

for epoch in range(1, n_epochs+1):
    # Training loop
    train_loss = 0.0
    model.train()
    for batch in train_dataloader:
        sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(sequence_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * sequence_data.size(0)

    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)

    # Test loop
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
            labels = batch["label"].to(device)

            outputs = model(sequence_data)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * sequence_data.size(0)

    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)

    print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTest Loss: {test_loss:.6f}")

# Plot the training and test errors
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

## Create a confusion matrix



# Get predictions for the test set
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_dataloader:
        sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
        labels = batch["label"].to(device)

        outputs = model(sequence_data)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * sequence_data.size(0)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = label_encoder.classes_



def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names)
plt.show()
