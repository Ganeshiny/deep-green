import os
import gzip
from pathlib import Path
from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder
from extract_seqs_from_cif import read_seqs_file
import numpy as np
import argparse
import glob
import multiprocessing
import csv

def make_distance_maps(pdbfile, chain=None, sequence=None):
    print(sequence, chain)
    # Check if the file is gzipped
    if pdbfile.endswith('.gz'):
        with gzip.open(pdbfile, 'rt') as handle:  # 'rt' mode opens as text
            structure_data = handle.read()
    else:
        with open(pdbfile, 'r') as handle:
            structure_data = handle.read()

    structure_container = build_structure_container_for_pdb(structure_data, chain).with_seqres(sequence)
    print(sequence, chain)

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)

    return ca.chains, cb.chains

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

def cif2cmap(pdb, chain, seq, pdir):
    ca, cb = make_distance_maps(os.path.join(pdir, pdb + '.cif.gz'), chain=chain, sequence=seq)
    return ca[chain]['contact-map'], cb[chain]['contact-map']

def write_annot_npz(prot, prot2seq, struct_dir):
    pdb, chain = prot.split('_')
    tmp_dir = os.path.join(struct_dir, 'tmp_cmap_files')

    # Ensure the tmp_cmap_files directory exists
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        A_ca, A_cb = cif2cmap(pdb, chain, prot2seq[prot], pdir=struct_dir)
        np.savez_compressed(os.path.join(tmp_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='./preprocessing/data/pdb2go.tsv', help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-seqs', type=str, default='./preprocessing/data/pdb2sequences.fasta', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=20, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-struc_dir', type=str, default='./preprocessing/data/structure_files', help='Directory containing cif files')

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
