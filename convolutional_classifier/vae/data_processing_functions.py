import torch
import logging
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import csv
from torch.utils.data import Dataset
import numpy as np
import os
import random
import fire
import pickle
import matplotlib.pyplot as plt

def filter_by_taxonomy(full_labels, taxonomic_level, taxonomic_group, 
                       classification_level, minimum_samples, fraction):
    taxonomic_levels = ['ncbi_identifier', 'Domain', 'Kingdom', 'Phylum', 'Order', 'Family', 'Genus', 'Species']
    filter_words = ['uncultured', 'unidentified', 'metagenome', 'bacterium']  # Corrected filter words

    # Make sure provided taxonomic_level and classification_level are valid
    assert taxonomic_level in taxonomic_levels, f"Invalid taxonomic_level. Must be one of {taxonomic_levels}"
    assert classification_level in taxonomic_levels, f"Invalid classification_level. Must be one of {taxonomic_levels}"
    
    level_index = taxonomic_levels.index(taxonomic_level)
    classification_level_index = taxonomic_levels.index(classification_level)
    filtered_labels = []
    
    for label in full_labels:
        current_label = label['label'][level_index]
        classification_label = label['label'][classification_level_index]
        if (current_label == taxonomic_group and not any(word in classification_label for word in filter_words) 
            and not (classification_label.endswith('sp.') and 'subsp.' not in classification_label)):
            filtered_labels.append(label)
    
    # Shuffle the labels
    random.shuffle(filtered_labels)
    
    # Determine the number of sequences to use based on the fraction
    total_sequences = len(filtered_labels)
    num_sequences_to_use = int(fraction * total_sequences)
    
    # Slice the labels to select the desired fraction
    filtered_labels = filtered_labels[:num_sequences_to_use]
    
    labels_counter = Counter([label['label'][classification_level_index] for label in filtered_labels])
    labels_with_minimum_samples = [label for label in filtered_labels if labels_counter[label['label'][classification_level_index]] >= minimum_samples]
    
    # Count the occurrences of the classification levels in the filtered labels
    classification_counts = Counter([label['label'][classification_level_index] for label in labels_with_minimum_samples])
    num_classes = len(classification_counts)
    
    return labels_with_minimum_samples, labels_counter, classification_counts, num_classes



def encode_labels(labels, classification_level):
    le = LabelEncoder()
    
    taxonomic_levels = ['ncbi_identifier', 'Domain', 'Kingdom', 'Phylum', 'Order', 'Family', 'Genus', 'Species']
    classification_level_index = taxonomic_levels.index(classification_level)
    
    original_labels = [label['label'][classification_level_index] for label in labels]
    encoded_labels = le.fit_transform(original_labels) 
    
    label_map = {original: encoded for original, encoded in zip(original_labels, encoded_labels)}
    encoded_labels_dict = [{'sequence_id': label['sequence_id'], 'label': label_map[label['label'][classification_level_index]]} for label in labels]
    return encoded_labels_dict, label_map, le


def split_data(encoded_labels_dict):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    indices = np.arange(len(encoded_labels_dict))
    labels = [label_dict['label'] for label_dict in encoded_labels_dict]
    for train_index, temp_index in sss.split(indices, labels):
        train_data = [encoded_labels_dict[i] for i in train_index]
        temp_data = [encoded_labels_dict[i] for i in temp_index]
        
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    temp_labels = [label_dict['label'] for label_dict in temp_data]
    for valid_index, test_index in sss.split(np.arange(len(temp_data)), temp_labels):
        valid_data = [temp_data[i] for i in valid_index]
        test_data = [temp_data[i] for i in test_index]
    
    return train_data, valid_data, test_data

class SequenceDataset(Dataset):
    def __init__(self, folder_path, data):
        # Construct full file paths
        self.pt_files = [os.path.join(folder_path, str(f['sequence_id']) + '.pt') for f in data]

        # Use provided labels
        self.labels = {label_dict['sequence_id']: label_dict['label'] for label_dict in data}

    def __getitem__(self, index):
        sequence_file = self.pt_files[index]
        sequence_id = os.path.splitext(os.path.basename(sequence_file))[0]  # Get the sequence_id from the file name
        sequence_dict = torch.load(sequence_file)
        sequence = sequence_dict['sequence_tensor']
        label = torch.tensor(self.labels[sequence_id], dtype=torch.long)
        return {'sequence': sequence, 'label': label}

    def __len__(self):
        return len(self.pt_files)
    
def write_counts_to_csv(counts, filepath):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Classification Level", "Count"])
        for key, count in counts.items():
            writer.writerow([key, count])

def plot_classification_histogram(classification_counts, save_path=None):
    # Extract classification levels and counts
    classification_levels = list(classification_counts.keys())
    counts = list(classification_counts.values())

    # Sort the classification levels and counts in descending order
    sorted_levels, sorted_counts = zip(*sorted(zip(classification_levels, counts), key=lambda x: x[1], reverse=True))

    # Calculate the figure size based on the number of levels
    num_levels = len(classification_levels)
    fig_width = max(num_levels * 0.5, 6)  # Adjust the multiplier and minimum width as needed
    fig_height = 6  # Set the height of the figure

    # Create the figure with the calculated size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.subplots_adjust(bottom=0.4)  # Adjust the bottom spacing as needed

    # Create the histogram with sorted data
    ax.bar(range(len(sorted_levels)), sorted_counts)
    ax.set_xticks(range(len(sorted_levels)))
    ax.set_xticklabels(sorted_levels, rotation=90, ha='right')  # Rotate and align the labels
    plt.xlabel('Classification Level')
    plt.ylabel('Count')
    plt.title('Histogram of Classification Counts')

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Display the plot
    plt.show()



def main(labels_file, taxonomic_level, taxonomic_group, 
         classification_level, minimum_samples, 
         fraction, output_path):
    base_out = '_'.join([taxonomic_level, taxonomic_group,classification_level, str(minimum_samples),str(fraction)])
    out_csv = os.path.join(output_path, base_out + ".csv")
    out_png = os.path.join(output_path, base_out + ".png")
    # Load labels from the pickle file
    with open(labels_file, 'rb') as f:
        full_labels = pickle.load(f)
    
    # Filter labels based on taxonomic criteria
    _, _, classification_counts, _ = filter_by_taxonomy(full_labels, taxonomic_level, taxonomic_group, classification_level, minimum_samples, fraction)
    
    write_counts_to_csv(classification_counts, out_csv)
    plot_classification_histogram(classification_counts, out_png)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments using Fire CLI
    fire.Fire(main)