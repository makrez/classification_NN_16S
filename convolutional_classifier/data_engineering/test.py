from preprocessing import encode_labels, load_sequences, split_data
from sklearn.preprocessing import LabelEncoder

msa_file_path = '../../data/Actinobacteria_10000_seqs.fasta'
alignment_length = 50000
taxonomy_level = 5 

taxonomy_labels, full_taxonomy_labels, class_counts, original_indices = load_sequences(msa_file_path, 
                                                                     alignment_length, 
                                                                     taxonomy_level)

print(taxonomy_labels[200])
print(full_taxonomy_labels[200])
print(original_indices[200])

# encoded_labels, label_encoder = encode_labels(taxonomy_labels)

# train_index, valid_index, test_index = split_data(encoded_labels)

# print(train_index[0], valid_index[0], test_index[0])
# print(encoded_labels[train_index[0]], encoded_labels[valid_index[0]], encoded_labels[test_index[0]])
# print(taxonomy_labels[train_index[0]], taxonomy_labels[valid_index[0]], taxonomy_labels[test_index[0]])
