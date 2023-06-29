import os
import fire
from process_data_functions import load_sequences, process_sequences


def main(msa_file_path='../../data/Actinobacteria_10000_seqs.fasta', 
         alignment_length=50000, 
         dataset_dir='/scratch/mk_cas/datasets2'):
    
    """
    This script pre-processes sequence data and saves it for later usage.

    Args:
    msa_file_path (str): The path to the file with the sequence data. Default is '../../data/Actinobacteria_10000_seqs.fasta'.
    alignment_length (int): The length of the sequence alignment. Default is 50000.
    dataset_dir (str): The directory where the processed data will be saved. Default is '/scratch/mk_cas/datasets'.
    
    Example usage:
    python your_script.py --msa_file_path=../../data/Actinobacteria_10000_seqs.fasta --alignment_length=50000 --dataset_dir=/scratch/mk_cas/datasets
    """
    # Ensure dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    full_taxonomy_labels, original_indices = load_sequences(msa_file_path, alignment_length)

    # Create paths for your sequence tensors
    sequence_path = os.path.join(dataset_dir, 'sequences')

    # Make directories if they don't exist
    os.makedirs(sequence_path, exist_ok=True)

    process_sequences(msa_file_path, alignment_length, sequence_path, full_taxonomy_labels, original_indices)

if __name__ == '__main__':
    fire.Fire(main)
