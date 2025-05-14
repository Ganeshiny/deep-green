import torch
import pickle
from pathlib import Path
import os 
from Bio.PDB.MMCIFParser import MMCIFParser
import gzip
from Bio.SeqUtils import seq1
import obonet
import numpy as np
import argparse
import glob
import multiprocessing
import csv

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

def compute_metrics(y_true, y_pred):
    metrics = {}

    # Identify labels with positive samples
    valid_labels = np.where(y_true.sum(axis=0) > 0)[0]
    y_true_filtered = y_true[:, valid_labels]
    y_pred_filtered = y_pred[:, valid_labels]

    # AUPR
    metrics['aupr_micro'] = average_precision_score(y_true_filtered, y_pred_filtered, average='micro')
    metrics['aupr_macro'] = average_precision_score(y_true_filtered, y_pred_filtered, average='macro')
    metrics['aupr_per_label'] = average_precision_score(y_true_filtered, y_pred_filtered, average=None)

    # Fmax and threshold sweep
    thresholds = np.linspace(0, 1, 101)
    fmax_values = []
    fmax_per_label = np.zeros(len(valid_labels))

    for t in thresholds:
        y_pred_bin = (y_pred_filtered >= t).astype(int)
        micro_f1 = f1_score(y_true_filtered, y_pred_bin, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true_filtered, y_pred_bin, average='macro', zero_division=0)
        fmax_values.append((t, micro_f1, macro_f1))

    # Best micro/macro
    best_micro = max(fmax_values, key=lambda x: x[1])
    best_macro = max(fmax_values, key=lambda x: x[2])
    metrics['fmax_micro'], metrics['best_t_micro'] = best_micro[1], best_micro[0]
    metrics['fmax_macro'], metrics['best_t_macro'] = best_macro[1], best_macro[0]

    # Per-label Fmax
    for i in range(len(valid_labels)):
        label_true = y_true_filtered[:, i]
        label_pred = y_pred_filtered[:, i]
        best_f1 = 0
        for t in thresholds:
            label_pred_bin = (label_pred >= t).astype(int)
            f1 = f1_score(label_true, label_pred_bin, zero_division=0)
            best_f1 = max(best_f1, f1)
        fmax_per_label[i] = best_f1

    metrics['fmax_per_label'] = fmax_per_label
    metrics['thresholds'] = thresholds
    metrics['fmax_values'] = fmax_values
    metrics['valid_label_indices'] = valid_labels

    return metrics, thresholds

def calculate_class_weights(dataset, device, epsilon=1e-6):
    # Get number of classes
    num_classes = dataset[0].y.size(1)

    # Initialize counter
    class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_samples = 0

    for data in dataset:
        y = data.y.to(device)
        class_counts += y.sum(dim=0).float()
        total_samples += y.size(0)

    class_freq = class_counts / (total_samples + epsilon)
    class_weights = 1.0 / (class_freq + epsilon)

    class_weights = class_weights / class_weights.mean()

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)
    return class_weights


def save_alpha_weights(alpha, filename):
    with open(filename, 'wb') as f:
        pickle.dump(alpha, f)
    print(f'Alpha weights saved to {filename}')

def load_alpha_weights(filename):
    with open(filename, 'rb') as f:
        alpha_weights = pickle.load(f)
    return alpha_weights

def get_seqs(fname):
    with gzip.open(fname, "rt") as handle:
        parser = MMCIFParser()
        pdb_id = os.path.split(fname)[1].split(".")[0] 
        structure = parser.get_structure(pdb_id, handle)
        chains = {f"{pdb_id}_{chain.id}":seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
    return chains

def write_seqs_from_cifdir(dirpath, fname):
    structure_dir = Path(dirpath)
    seqs_file = open(fname, "w")
    for file in structure_dir.glob("*"):
        chain_dir = get_seqs(file)
        for key in chain_dir:
            #unknown_percentage = chain_dir[key].count("X")/len(chain_dir[key])
            #print(f"seq:{chain_dir[key]}, percentage:{unknown_percentage}")
            #if unknown_percentage <= 0.2:
            seqs_file.write(f">{key}\n{chain_dir[key]}\n")
    return seqs_file

def make_distance_maps(file_path):
    """
    Reads a compressed .cif.gz file and extracts atomic coordinates.
    
    Args:
        file_path (str): Path to the .cif.gz file.
    
    Returns:
        dict: A dictionary where keys are chain IDs and values contain:
              - 'C_alpha': Distance matrix for alpha carbons (Cα)
              - 'C_beta': Distance matrix for beta carbons (Cβ)
    """
    # Load the CIF file using Bio.PDB
    with gzip.open(file_path, 'rt') as f:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", f)
    
    distance_matrices = {}

    for model in structure:  # Usually only one model in PDB/MMCIF files
        for chain in model:
            chain_id = chain.id
            ca_coords, cb_coords = [], []

            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].coord)
                if 'CB' in residue:  # Cβ exists in all residues except Glycine
                    cb_coords.append(residue['CB'].coord)
                else:
                    cb_coords.append(residue['CA'].coord)  # Use Cα for Glycine

            # Convert lists to NumPy arrays
            ca_coords = np.array(ca_coords)
            cb_coords = np.array(cb_coords)

            # Compute pairwise Euclidean distance matrices
            ca_dist_map = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
            cb_dist_map = np.linalg.norm(cb_coords[:, None, :] - cb_coords[None, :, :], axis=-1)

            distance_matrices[chain_id] = {
                "C_alpha": ca_dist_map,
                "C_beta": cb_dist_map
            }

    return distance_matrices

def cif2cmap(pdb, chain, pdir):
    distance_matrices = make_distance_maps(os.path.join(pdir, pdb + '.cif.gz'))
    return distance_matrices[chain]['C_alpha'], distance_matrices[chain]['C_beta']

def write_annot_npz(prot, prot2seq, struct_dir, is_csm=True):
    print("Debug prot:", prot)
    
    if is_csm:
        # pdb should be everything before the last underscore
        pdb = '_'.join(prot.split('_')[:-1])
        # chain should be the part after the last underscore
        chain = prot.split('_')[-1]
    else:
        # In case `is_csm` is False, split `prot` normally
        pdb, chain = prot.split('_')
    
    tmp_dir = os.path.join(struct_dir, 'tmp_cmap_files')

    # Ensure the tmp_cmap_files directory exists
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        print("Processing", pdb, chain)
        # Call cif2cmap (assuming it's defined elsewhere in your code)
        print(struct_dir)
        A_ca, A_cb = cif2cmap(pdb, chain, pdir=struct_dir)
        
        # Save the results in a compressed .npz file
        np.savez_compressed(os.path.join(tmp_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot])
    except Exception as e:
        print("Exception occurred:", e)


def read_seqs_file(seqs_file):
    pdb2seq = {}
    with open(seqs_file, "r") as fasta_handle:
        for line in fasta_handle:
            if ">" in line:
                key  = line.strip().replace(">", "")
            else:
                unknown_percentage = line.strip().count("X")/len(line.strip())
                if unknown_percentage <= 0.2:
                    pdb2seq[key] = line.strip() 
                #else:
                    #print(f"X character percentage of {pdb2seq[key]} is: ", unknown_percentage)
    return pdb2seq

def load_go_graph(fname):
    go_graph = obonet.read_obo(fname)
    #print(f"DEBUG: {go_graph}, and the number of nodes: {len(go_graph.nodes)}")
    return go_graph


