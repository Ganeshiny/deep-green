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

def calculate_class_weights(dataset, device):
    # Calculate the number of classes in the dataset
    num_classes = dataset[0].y.size(1)
    print("Number of classes:", num_classes)

    # Initialize class counters
    class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)


    # Count the number of examples in each class
    for data in dataset:
        class_counts += data.y.sum(dim=0).float().to(device)
        #print(class_counts)
        

    # Calculate class weights by taking the inverse of class frequency
    class_weights = 1.0 / (class_counts / class_counts.sum())
    print(class_weights)
    return class_weights.to(device)

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
        A_ca, A_cb = cif2cmap(pdb, chain, prot2seq[prot], pdir=struct_dir)
        
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


