import sequence_utils
from sequence_utils import hot_dna


sequence="ATGCN--YATG"
print(sequence)

my_hot_dna = hot_dna(sequence)

preprocessed_sequence = my_hot_dna._preprocess_sequence(sequence)

print(preprocessed_sequence)
print("one hot encoded")

onehot_encoded_sequence = my_hot_dna._onehot_encode(preprocessed_sequence)
print(onehot_encoded_sequence)
