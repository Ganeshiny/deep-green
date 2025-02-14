import os
import gzip
from pathlib import Path
from extract_seqs_from_cif import read_seqs_file
import numpy as np
import argparse
import glob
import multiprocessing
import csv
import numpy as np
import os
import gzip
import numpy as np
from Bio.PDB import MMCIFParser
import matplotlib.pyplot as plt

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

def load_GO_annot(filename):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        for ont in onts:
            next(reader, None)  
            goterms[ont] = next(reader)
            next(reader, None)  
            gonames[ont] = next(reader)

        next(reader, None)  
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                prot2annot[prot][onts[i]] = [goterm for goterm in prot_goterms[i].split(',') if goterm != '']
    return prot2annot, goterms, gonames


def cif2cmap(pdb, chain, pdir):
    distance_matrices = make_distance_maps(os.path.join(pdir, pdb + '.cif.gz'))
    return distance_matrices[chain]['C_alpha'], distance_matrices[chain]['C_beta']

def write_annot_npz(prot, prot2seq, struct_dir):
    pdb, chain = prot.split('_')
    tmp_dir = os.path.join(struct_dir, 'tmp_cmap_files')

    # Ensure the tmp_cmap_files directory exists
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        A_ca, A_cb = cif2cmap(pdb, chain, pdir=struct_dir)
        np.savez_compressed(os.path.join(tmp_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachinthadata/_annot.tsv', help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-seqs', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/seqs_from_structure_dir.fasta', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=20, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-struc_dir', type=str, default='/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/structure_files', help='Directory containing cif files')

    args = parser.parse_args()
    struct_dir = args.struc_dir

    prot2goterms, _, _ = load_GO_annot(args.annot)
    print("### number of annotated proteins: %d" % (len(prot2goterms)))

    prot2seq = read_seqs_file(args.seqs)
    print("### number of proteins sequences: %d" % (len(prot2seq)))

    npz_pdb_chains = glob.glob(os.path.join(struct_dir, 'tmp_cmap_files', '*.npz'))
    npz_pdb_chains = [Path(chain).stem for chain in npz_pdb_chains]

    to_be_processed = list(prot2goterms.keys())
    to_be_processed = list(set(to_be_processed).difference(npz_pdb_chains))
    print("Number of pdbs to be processed =", len(to_be_processed))
    print(to_be_processed)

    nprocs = args.num_threads
    nprocs = min(nprocs, multiprocessing.cpu_count())

    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.starmap(write_annot_npz, zip(to_be_processed, [prot2seq]*len(to_be_processed), [struct_dir]*len(to_be_processed)))
    else:
        for prot in to_be_processed:
            write_annot_npz(prot, prot2seq, struct_dir)
