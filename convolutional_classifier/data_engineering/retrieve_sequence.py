import os
import pickle
import torch
from Bio import SeqIO
import fire

class RetrieveSequenceInfo:
    def __init__(self, sequence_id, sequence_path, msa_file_path):
        """
        Initialize with the sequence ID, sequence tensor path and the original FASTA file path.

        Args:
        sequence_id (str): The unique identifier for the sequence of interest.
        sequence_path (str): The path to the directory where sequence tensors are saved.
        msa_file_path (str): The path to the original MSA FASTA file.
        """
        self.sequence_id = int(sequence_id)
        self.sequence_path = sequence_path
        self.msa_file_path = msa_file_path
        self.label_dict = pickle.load(open(f"{sequence_path}/full_labels.pkl", "rb"))

        # Debug print
        print("First few items in label_dict:", self.label_dict[:5])

    def retrieve(self):
        """
        Retrieve and print the tensor, full label and original FASTA header for the given sequence ID.
        """
        # Load tensor
        tensor_data = torch.load(f"{self.sequence_path}/{self.sequence_id}.pt")

        # Retrieve full label
        label_match = [item["label"] for item in self.label_dict if int(item["sequence_id"]) == self.sequence_id]
        if not label_match:
            raise ValueError(f"No match found for sequence_id {self.sequence_id}")
        full_label = label_match[0]

        # Retrieve original FASTA header
        original_header = ""
        with open(self.msa_file_path) as handle:
            for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
                if i == self.sequence_id:
                    original_header = record.description
                    break

        print("Tensor:", tensor_data["sequence_tensor"])
        print("Full label:", full_label)
        print("Original FASTA header:", original_header)
        
def main(sequence_id, sequence_path='/scratch/mk_cas/datasets2/sequences', msa_file_path='../../data/Actinobacteria_10000_seqs.fasta'):
    """
    Given a sequence ID, this script retrieves the tensor, full label, and original FASTA header.

    Args:
    sequence_id (str): The unique identifier for the sequence of interest.
    sequence_path (str): The path to the directory where sequence tensors are saved. Default is '/scratch/mk_cas/datasets/sequences'.
    msa_file_path (str): The path to the original MSA FASTA file. Default is '../../data/Actinobacteria_10000_seqs.fasta'.
    
    Example usage:
    python your_script.py --sequence_id=0 --sequence_path=/scratch/mk_cas/datasets/sequences --msa_file_path=../../data/Actinobacteria_10000_seqs.fasta
    """
    retriever = RetrieveSequenceInfo(sequence_id, sequence_path, msa_file_path)
    retriever.retrieve()


if __name__ == '__main__':
    fire.Fire(main)
