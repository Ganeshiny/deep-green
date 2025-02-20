import torch
from pathlib import Path
import os
import numpy as np
import argparse
import glob
import csv
from transformers import BertTokenizer, BertModel
import scipy.sparse as sp
from model import GCN, RareLabelGNN
from utils import write_seqs_from_cifdir, read_seqs_file, write_annot_npz
import gc
import json
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.5

# Load ProtBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
protbert_model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
protbert_model.gradient_checkpointing_enable()  # Reduces memory usage
protbert_model.to(device).eval()

# Dictionaries for residue properties
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

POLARITY = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0, 'Q': 1, 'E': 1,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
    'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
    'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

def compute_residue_features(sequence):
    """Compute residue-level features: hydrophobicity, polarity, and charge."""
    return [HYDROPHOBICITY.get(res, 0) for res in sequence], \
           [POLARITY.get(res, 0) for res in sequence], \
           [CHARGE.get(res, 0) for res in sequence]

def seq2onehot(seq):
    """Convert sequence to one-hot encoding."""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = {char: idx for idx, char in enumerate(chars)}
    vocab_one_hot = np.eye(len(chars), dtype=int)
    return np.array([vocab_one_hot[vocab_embed.get(v, vocab_embed['X'])] for v in seq])

def seq2protbert(seq):
    """Get ProtBERT embeddings for a sequence."""
    inputs = tokenizer(' '.join(seq), return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = protbert_model(**inputs)
    return outputs.last_hidden_state.detach().cpu().numpy()

def get_adjacency_info(distance_matrix, threshold=8.0):
    """Convert distance matrix to adjacency matrix."""
    adjacency_matrix = (distance_matrix <= threshold).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)
    edge_indices = np.nonzero(adjacency_matrix)
    return torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)

def process_structures(struct_dir, sequence_file):
    """
    Process structure files to generate sequences and contact maps.
    Skips processing if the tmp_cmap_files directory already exists and contains .npz files.
    """
    tmp_cmap_dir = os.path.join(struct_dir, 'tmp_cmap_files')
    
    # Check if the tmp_cmap_files directory exists and contains .npz files
    if os.path.exists(tmp_cmap_dir) and glob.glob(os.path.join(tmp_cmap_dir, '*.npz')):
        print("tmp_cmap_files directory already exists and contains .npz files. Skipping processing.")
        return True  # Indicate that processing is not needed

    # Create the tmp_cmap_files directory if it doesn't exist
    os.makedirs(tmp_cmap_dir, exist_ok=True)

    print("Extracting sequences... This might take a while...")
    write_seqs_from_cifdir(struct_dir, sequence_file)
    
    # Load sequence data in a memory-efficient way
    pdb2seq = read_seqs_file(sequence_file)

    # Gather already processed chains to avoid redundant processing
    npz_pdb_chains = {Path(chain).stem for chain in glob.glob(os.path.join(tmp_cmap_dir, '*.npz'))}

    # Identify structures that still need to be processed
    to_be_processed = set(pdb2seq.keys()).difference(npz_pdb_chains)
    print(f"Number of PDBs to be processed = {len(to_be_processed)}")

    # Process sequentially without multiprocessing
    print("Processing sequentially to reduce memory usage.")
    for prot in to_be_processed:
        write_annot_npz(prot, pdb2seq, struct_dir)
        gc.collect()

    # Explicitly release memory after processing
    del pdb2seq, npz_pdb_chains, to_be_processed
    gc.collect()

    return True  # Indicate successful processing
def run_predictions(struct_dir, model, output_file, gonames, goids, batch_size=8):
    """Run predictions on protein structures and save results to CSV."""
    npz_pdb_chains = [Path(chain).stem for chain in glob.glob(os.path.join(struct_dir, 'tmp_cmap_files', '*.npz'))]
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PDB_ID', 'GO_IDs', 'GO_Names'])
        for i in range(0, len(npz_pdb_chains), batch_size):
            batch = npz_pdb_chains[i:i+batch_size]
            for id in batch:
                try:
                    path = os.path.join(struct_dir, 'tmp_cmap_files', f'{id}.npz')
                    data = np.load(path)
                    ca_dist = data['C_alpha']
                    seq = data['seqres'].item()

                    # Extract features
                    onehot_features = torch.tensor(seq2onehot(seq), dtype=torch.float).to(device)
                    protbert_features = torch.tensor(seq2protbert(seq), dtype=torch.float).to(device).squeeze(0)
                    additional_features = torch.tensor(np.stack(compute_residue_features(seq), axis=1), dtype=torch.float).to(device)
                    node_features = torch.cat([protbert_features, additional_features], dim=1)
                    adjacency_info = get_adjacency_info(ca_dist).to(device)

                    # Run model inference
                    with torch.no_grad():
                        out = model(node_features, adjacency_info, torch.tensor([0] * len(seq), dtype=torch.long, device=device))
                    pred = torch.sigmoid(out) > threshold
                    true_indices = torch.nonzero(pred, as_tuple=False).cpu().numpy()

                    # Write predictions to CSV
                    if true_indices.ndim == 2 and true_indices.shape[1] >= 2:
                        class_indices = true_indices[:, 1]
                        go_names = [gonames[i] for i in class_indices.tolist()]
                        go_terms = [goids[i] for i in class_indices.tolist()]
                        writer.writerow([id, go_terms, go_names])

                    # Cleanup
                    del onehot_features, protbert_features, adjacency_info, out, pred
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"Error processing {id}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-struc_dir', type=str, default='examples/structure_files', help='Directory containing cif files')
    parser.add_argument('-seqs', type=str, default='examples/predictions_seqs.fasta', help='FASTA file containing sequences')
    parser.add_argument('-model_path', type=str, default="model_and_weight_files/hpc.pth", help='Path to the trained model weights')
    parser.add_argument('-output', type=str, default='examples/predictionshpc.csv', help='Output CSV file for predictions')
    parser.add_argument('-annot_dict', type=str, default='preprocessing/data/annot_dict_hpc.pkl', help='Path to the annotation dictionary')
    args = parser.parse_args()
    annot_dict = args.annot_dict

    struct_dir = args.struc_dir
    sequence_file = args.seqs

    # Process structure files to generate sequence and contact map data
    process_structures(struct_dir, sequence_file)

    # Load GCN model
    with open("model_and_weight_files/model_info_2_layers_hpc.json", 'r') as f:
        model_info = json.load(f)

    # Initialize the model with the correct parameters
    model = RareLabelGNN(
        input_size=model_info['input_size'],
        hidden_sizes=model_info['hidden_sizes'],
        output_size=model_info['output_size']
    )

    # Load the state dictionary
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # Load GO annotations
    with open(args.annot_dict, 'rb') as f:
        data = pickle.load(f)
    goterms = data['goterms']['biological_process']
    gonames = data['gonames']['biological_process']

    # Run predictions
    run_predictions(struct_dir, model, args.output, gonames, goterms)