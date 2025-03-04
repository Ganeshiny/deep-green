import torch
from pathlib import Path
import os
import numpy as np
import argparse
import glob
import csv
from transformers import BertTokenizer, BertModel
import scipy.sparse as sp
from model import GCN2, GCN
from utils import write_seqs_from_cifdir, read_seqs_file, write_annot_npz
import gc
import json
import pickle
BASE_PATH = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.7

# Load ProtBERT model and tokenizer with gradient checkpointing for memory efficiency
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
protbert_model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
protbert_model.gradient_checkpointing_enable()  # Helps with memory usage
protbert_model.to(device).eval()

ontology = "biological_process"

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
    hydrophobicity = [HYDROPHOBICITY.get(res, 0) for res in sequence]
    polarity = [POLARITY.get(res, 0) for res in sequence]
    charge = [CHARGE.get(res, 0) for res in sequence]
    return hydrophobicity, polarity, charge


def seq2onehot(seq):
    """
    Convert sequence to one-hot encoding.
    """
    #print("seq: ", seq)
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = {char: idx for idx, char in enumerate(chars)}
    #print("vocab embed: ", vocab_embed)
    vocab_one_hot = np.eye(len(chars), dtype=int)
    #print("vocab_one_hot: ", vocab_one_hot)

    # Ensure `seq` is a string before iterating over it
    if isinstance(seq, np.ndarray):
        seq = seq.item()  # Convert NumPy 0-d array to string if needed
    
    if isinstance(seq, str):
        seqs_x = np.array([vocab_one_hot[vocab_embed.get(v, vocab_embed['X'])] for v in seq])
        return seqs_x
    else:
        raise ValueError(f"Invalid sequence format: {type(seq)}")

def seq2protbert(seq):
    seq = ' '.join(seq)
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():  # No need for gradients during inference
        # Get ProtBERT embeddings
        outputs = protbert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

    # Detach and move tensors back to CPU to free GPU memory
    embeddings = embeddings.detach().cpu().numpy()
    attention_mask = attention_mask.detach().cpu().numpy()

    features = []
    for seq_num in range(len(embeddings)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        if seq_len > 2:
            seq_emd = embeddings[seq_num][1:seq_len-1]  # without [CLS] and [SEP]
            features.append(seq_emd)

    # Cleanup to free memory
    del input_ids, attention_mask, outputs
    torch.cuda.empty_cache()  # Free GPU memory
    gc.collect()

    return np.array(features)

def get_adjacency_info(distance_matrix, threshold = 8.0):
    adjacency_matrix = (distance_matrix <= threshold).astype(int)
    #print(adjacency_matrix)
    np.fill_diagonal(adjacency_matrix, 0)
    edge_indices = np.nonzero(adjacency_matrix)

    coo_matrix = sp.coo_matrix((np.ones_like(edge_indices[0]), edge_indices))
    return torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)

# Function to generate sequence file and contact maps
def process_structures(struct_dir, sequence_file):
    print("Extracting sequences... This might take a while...")
    write_seqs_from_cifdir(struct_dir, sequence_file)
    
    # Load sequence data in a memory-efficient way
    pdb2seq = read_seqs_file(sequence_file)

    # Gather already processed chains to avoid redundant processing
    npz_pdb_chains = {Path(chain).stem for chain in glob.glob(os.path.join(struct_dir, 'tmp_cmap_files', '*.npz'))}

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

# Main prediction function with batch processing
def run_predictions(struct_dir, model, output_file, gonames, goids, batch_size=8):
    npz_pdb_chains = glob.glob(os.path.join(struct_dir, 'tmp_cmap_files', '*.npz'))
    npz_pdb_chains = [Path(chain).stem for chain in npz_pdb_chains]
    # Open CSV to write predictions
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PDB_ID', 'GO_IDs', 'GO_Names'])

        # Process each protein in batches to reduce memory footprint
        for i in range(0, len(npz_pdb_chains), batch_size):
            batch = npz_pdb_chains[i:i+batch_size]
            for id in batch:
                path = os.path.join(struct_dir, 'tmp_cmap_files', f'{id}.npz')
                data = np.load(path)

                ca_dist = data['C_alpha']
                seq = data['seqres'].item()
                #print("This is the sequence", seq)

                # One-hot encoding
                onehot_features = torch.tensor(seq2onehot(seq), dtype=torch.float).to(device)

                # ProtBERT embeddings
                protbert_features = torch.tensor(seq2protbert(seq), dtype=torch.float).to(device).squeeze(0)

                # Compute additional features (hydrophobicity, polarity, charge)
                hydrophobicity, polarity, charge = compute_residue_features(seq)
                additional_features = torch.tensor(
                    np.stack([hydrophobicity, polarity, charge], axis=1),
                    dtype=torch.float
                ).to(device)

                node_features = torch.cat([protbert_features, additional_features], dim=1)

                # Get adjacency info (move adjacency_info to the same device)
                adjacency_info = get_adjacency_info(ca_dist).to(device)

                # Pass the data to the model
                with torch.no_grad():  # No need to compute gradients during inference
                    # Ensure the sequence length is used properly
                    out = model(node_features, adjacency_info, torch.tensor([0] * len(seq), dtype=torch.long, device=device))

                # Convert logits to probabilities and get predictions
                pred = torch.sigmoid(out) > threshold 

                true_indices = torch.nonzero(pred, as_tuple=False).cpu().numpy()  # Remove squeeze

                # Check if true_indices is valid and extract class indices
                if true_indices.ndim == 2 and true_indices.shape[1] >= 2:
                    # Assume second column contains class indices for 2D pred
                    class_indices = true_indices[:, 1]  # Adjust based on pred shape
                    go_names = [gonames[i] for i in class_indices.tolist()]
                    go_terms = [goids[i] for i in class_indices.tolist()]
                else:
                    print(f"Unexpected true_indices shape: {true_indices.shape}")
                    go_names = []
                    go_terms = []

                
                # Save the PDB ID and the indices of the True predictions to CSV
                writer.writerow([id, go_terms, go_names])
                print(f"id: {id}, number of predicted terms: {len(go_names)}")
                go_names = []
                go_terms = []

                # Cleanup memory
                del onehot_features, protbert_features, adjacency_info, out, pred
                torch.cuda.empty_cache()
                gc.collect()

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-struc_dir', type=str, default=f'{BASE_PATH}/examples/structure_files', help='Directory containing cif files')
    parser.add_argument('-seqs', type=str, default=f'{BASE_PATH}/examples/predictions_seqs.fasta', help='FASTA file containing sequences')
    parser.add_argument('-model_path', type=str, default=f"/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/model_and_weight_files/model_weights_adjusted_for_local_minima_converging.pth", help='Path to the trained model weights')
    parser.add_argument('-output', type=str, default=f'{BASE_PATH}/examples/predictions_new.csv', help='Output CSV file for predictions')
    parser.add_argument('-annot_dict', type=str, default=f'{BASE_PATH}/preprocessing/data/annot_dict.pkl', help='Path to the annotation dictionary')
    args = parser.parse_args()
    annot_dict = args.annot_dict

    struct_dir = args.struc_dir
    sequence_file = args.seqs

    # Process structure files to generate sequence and contact map data
    process_structures(struct_dir, sequence_file)

    # Load GCN model
    model = torch.load(args.model_path)
    # Path to the model info JSON file
    '''    MODEL_INFO_PATH = f"{BASE_PATH}/model_and_weight_files/model_info_2_layers.json"

        # Step 1: Load the model information from the JSON file
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)'''

    '''    # Extract the model parameters from the JSON file
        input_size = model_info['input_size']
        hidden_sizes = model_info['hidden_sizes']
        output_size = model_info['output_size']'''

    dataset_save_path = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachinthadata/datasets.pkl"

    with open(dataset_save_path, 'rb') as f:
        datasets = pickle.load(f)

    pdb_protBERT_dataset_train = datasets['train']
    pdb_protBERT_dataset_test = datasets['test']
    pdb_protBERT_dataset_valid = datasets['valid']

    print(f"Loaded datasets: Train={pdb_protBERT_dataset_train[0].x[0]}, Test={pdb_protBERT_dataset_test[0].x[0]}, Valid={pdb_protBERT_dataset_valid[0].x[0]}")
    # Model Setup
    input_size = len(pdb_protBERT_dataset_train[0].x[0])
    hidden_sizes = [1027, 912, 512, 256]
    output_size = pdb_protBERT_dataset_train.num_classes
    # Step 2: Initialize the GCN model using the loaded parameters
    # Assuming the GCN constructor accepts input_size, hidden_sizes, and output_size
    model = GCN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)

    # Step 3: Load the saved state dictionary into the GCN model
    model.load_state_dict(torch.load(args.model_path))

    # Step 4: Move the model to the GPU or CPU
    model.to(device).eval()

    with open(annot_dict, 'rb') as input_file:
        data = pickle.load(input_file)

    prot2annot = data['prot2annot']
    goterms = data['goterms']['biological_process']
    gonames = data['gonames']['biological_process']
    #print(gonames[ontology])
    prot_list = data['prot_list']

    # Run predictions and save to CSV
    run_predictions(struct_dir, model, args.output, gonames, goterms)
