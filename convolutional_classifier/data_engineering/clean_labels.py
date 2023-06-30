import pickle
import os

path = "/scratch/mk_cas/full_silva_dataset/sequences/"

def split_dictionary(base_path=path, dict_file = "full_labels.pkl"):
    # Load the data
    with open(os.path.join(base_path, dict_file), 'rb') as f:
        data_list = pickle.load(f)

    # Initialize new lists
    eight_labels_list = []
    not_eight_labels_list = []

    # Iterate over list and split items
    for data in data_list:
        if len(data['label']) == 8:
            eight_labels_list.append(data)

        else:
            not_eight_labels_list.append(data)
    print(len(eight_labels_list))
    print(eight_labels_list[1:3])
    print(len(not_eight_labels_list))
    print(not_eight_labels_list[1:3])
    # Save lists as pickle files
    with open(os.path.join(base_path, 'full_labels_conform.pkl'), 'wb') as f:
        pickle.dump(eight_labels_list, f)
    with open(os.path.join(base_path, 'full_labels_non_conform.pkl'), 'wb') as f:
        pickle.dump(not_eight_labels_list, f)

# Call the function
split_dictionary()
