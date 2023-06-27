import pickle
from collections import Counter
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

output_path = "dataset_stats_bacillus"
os.makedirs(output_path, exist_ok=True)

def load_labels_and_get_counts(dataset_path):
    labels = pickle.load(open(f'{dataset_path}/labels.pkl', 'rb'))
    # extract class labels
    class_labels = [label[1] for label in labels]
    # count number of sequences per class
    class_counts = Counter(class_labels)
    return class_counts


def create_counts_dataframe(train_counts, valid_counts, test_counts, label_encoder):
    # Convert each Counter object to a DataFrame
    train_df = pd.DataFrame.from_records(list(train_counts.items()), columns=['Encoded Taxonomy', 'Train'])
    valid_df = pd.DataFrame.from_records(list(valid_counts.items()), columns=['Encoded Taxonomy', 'Validation'])
    test_df = pd.DataFrame.from_records(list(test_counts.items()), columns=['Encoded Taxonomy', 'Test'])

    # Merge all dataframes
    df = pd.merge(train_df, valid_df, on='Encoded Taxonomy', how='outer')
    df = pd.merge(df, test_df, on='Encoded Taxonomy', how='outer')

    # Replace NaN values with zeros
    df.fillna(0, inplace=True)

    # Convert counts to integers
    df['Train'] = df['Train'].astype(int)
    df['Validation'] = df['Validation'].astype(int)
    df['Test'] = df['Test'].astype(int)

    # Add taxonomy names
    df['Taxonomy'] = df['Encoded Taxonomy'].apply(lambda x: label_encoder.inverse_transform([x])[0])

    # Set Taxonomy as index
    df.set_index('Taxonomy', inplace=True)

    return df

train_path = "datasets_bacillus/train"
valid_path = "datasets_bacillus/valid"
test_path = "datasets_bacillus/test"

label_encoder = pickle.load(open(f'{train_path}/label_encoder.pkl', 'rb'))

train_counts = load_labels_and_get_counts(train_path)
valid_counts = load_labels_and_get_counts(valid_path)
test_counts = load_labels_and_get_counts(test_path)

counts_df = create_counts_dataframe(train_counts, valid_counts, test_counts, label_encoder)
# Save to csv
counts_df.to_csv(os.path.join(output_path, 'dataset_counts.csv'))

# create a heatmap

rel_df = counts_df.div(counts_df.sum(axis=0), axis=1)

# Drop the 'Encoded Taxonomy' column
rel_df = rel_df.drop('Encoded Taxonomy', axis=1)
counts_df = counts_df.drop('Encoded Taxonomy', axis=1)

# Generate a heatmap
fig, ax = plt.subplots(figsize=(10, 20))  

# Use relative abundance for color coding but absolute counts for annotation
sns.heatmap(rel_df, cmap='YlOrRd', annot=counts_df.values, fmt='d', ax=ax)

# Reduce the size of the color bar
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Relative Abundance')
cbar.ax.yaxis.label.set_size(14)
cbar.ax.tick_params(labelsize=12)

plt.title('Relative Abundances of Taxonomies in Each Dataset')

# Save the figure
plt.savefig(os.path.join(output_path, 'heatmap.png'))