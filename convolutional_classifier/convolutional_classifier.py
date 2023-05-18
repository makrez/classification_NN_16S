import os
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix
from models import ConvClassifier
import sequence_utils
from sequence_utils import SequenceDataset

# Read and summarize the MSA
msa_file_path = '../data/bacillus.aln'
sequence_utils.summarize_msa(msa_file_path)

### Read data
flatted_sequence = list()
sequence_labels = list()

alignment_length = 4500

with open(msa_file_path) as handle:
  for record in SeqIO.parse(handle, 'fasta'):
    label = str(record.description).rsplit(';', 1)[-1]
    seq_hot = sequence_utils.hot_dna(str(record.seq)[10:alignment_length+10]).onehot

    if len(seq_hot) == alignment_length:
      flatted_sequence.append(seq_hot)
      sequence_labels.append(label)

## Encode labels into integers
print(sequence_labels[1:5])
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(sequence_labels)
print(encoded_labels[1:5])

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
