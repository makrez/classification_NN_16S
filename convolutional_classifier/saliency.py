import os
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, ParameterGrid
import logging
import json
from models import ConvClassifier
import sequence_utils
from sequence_utils import SequenceDataset
from plotting_functions import plot_training_results
from training_functions import train_network
import matplotlib.pyplot as plt
import pandas as pd



# Set path and parameters
msa_file_path = '../data/bacillus.aln'
alignment_length = 4500

### Read data
flatted_sequence = list()
sequence_labels = list()

with open(msa_file_path) as handle:
  for record in SeqIO.parse(handle, 'fasta'):
    label = str(record.description).rsplit(';', 1)[-1]
    seq_hot = sequence_utils.hot_dna(str(record.seq)[10:alignment_length+10]).onehot

    if len(seq_hot) == alignment_length:
      flatted_sequence.append(seq_hot)
      sequence_labels.append(label)

## Encode labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(sequence_labels)
num_classes = len(set(encoded_labels))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(flatted_sequence, encoded_labels, test_size=0.2)

# Create separate datasets and dataloaders for training and test sets
train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the device, model and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvClassifier(input_length=alignment_length, num_classes=num_classes)
model.load_state_dict(torch.load("lr=1e-05_n_epoch=100/final_model.pt"))
model.to(device)  # Ensure the model is on the same device as the input data





def calculate_saliency_maps(model, inputs, device):
    """
    Calculate saliency maps.

    :param model: Trained model
    :param inputs: Inputs to the model
    :param device: Torch device
    :return: Saliency maps for inputs
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # We want to calculate the gradient of the output with respect to the input,
    # so we need to make sure that the input requires gradient
    inputs.requires_grad = True

    # Forward pass through the model to get the outputs
    outputs = model(inputs)

    # Get the maximum predicted output
    _, preds = outputs.max(1)

    # We’re going to calculate the gradients of the output with respect to the input
    outputs.backward(torch.ones(outputs.shape).to(device))

    # Get the gradients of the inputs
    saliency_maps = inputs.grad.data

    return saliency_maps

selected_inputs = torch.stack([torch.from_numpy(seq) for seq in X_train[:526]]).float().to(device)

# Reshape selected_inputs
selected_inputs = selected_inputs.permute(0, 2, 1)

# Calculate saliency maps
saliency_maps = calculate_saliency_maps(model, selected_inputs, device)
# Calculate the average saliency across sequences for each position
average_saliency = saliency_maps.abs().mean(dim=0).mean(dim=0)


# Plot the average saliency
plt.figure(figsize=(10,5))
plt.plot(average_saliency.cpu().numpy())
plt.xlabel('Sequence Position')
plt.ylabel('Average Saliency')
plt.title('Saliency Map')

# Save the plot
plt.savefig('saliency_map.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# ...

# Assuming that 'saliency_maps' is a 2D Tensor where 
# the first dimension is the sequence index and 
# the second dimension is the position in the sequence.

# Remove the channel dimension and take the absolute value
saliency_maps_abs = saliency_maps.squeeze().abs()

# Downsample the saliency map to reduce the number of positions
# Assuming that saliency_maps_abs is of shape (batch_size, channels, sequence_length)
downsampled_saliency = torch.nn.functional.avg_pool1d(saliency_maps_abs, 100).squeeze()

# Convert to a pandas DataFrame for easier plotting
downsampled_saliency_2d = downsampled_saliency.reshape(526, -1)
df_saliency = pd.DataFrame(downsampled_saliency_2d.cpu().numpy())

# Convert encoded labels back to original form
decoded_labels = label_encoder.inverse_transform(y_train[:len(df_saliency)])


# Add labels to the dataframe
df_saliency['label'] = decoded_labels
# Use seaborn to create a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df_saliency.groupby('label').mean())
plt.xlabel('Downsampled Position')
plt.ylabel('Label')
plt.title('Saliency Map')
plt.savefig('saliency_heatmap.png', dpi=300, bbox_inches='tight')
