import os
import csv
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import pickle

#Inializing here
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

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
    return (
        [HYDROPHOBICITY.get(res, 0) for res in sequence],
        [POLARITY.get(res, 0) for res in sequence],
        [CHARGE.get(res, 0) for res in sequence]
    )

def seq2onehot(seq):
    """Convert sequence to one-hot encoding."""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = {char: idx for idx, char in enumerate(chars)}
    vocab_one_hot = np.eye(len(chars), dtype=int)
    return np.array([vocab_one_hot[vocab_embed[v]] for v in seq])

def seq2protbert(seq):
    """Get ProtBERT embeddings for a protein sequence."""
    seq = ' '.join(seq)  # Add spaces between amino acids
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True, padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    embeddings = embeddings.detach().cpu().numpy()
    mask = attention_mask.detach().cpu().numpy()

    # Extract embeddings, removing CLS and SEP tokens
    return np.array([embeddings[i][1:np.sum(mask[i]) - 1] for i in range(len(embeddings))])

def read_list_file(filename):
    """Read list of PDB IDs from a file."""
    with open(filename, "r") as file:
        return [line.strip().replace("-", "_") for line in file.readlines()]

class PDB_Dataset(Dataset):
    def __init__(self, root, annot_file, num_shards=20, selected_ontology=None, transform=None, pre_transform=None, model="protBERT", pdb_split_set_file=None):
        self.model = model
        self.npz_dir = root
        self.num_shards = num_shards
        self.selected_ontology = selected_ontology
        self.transform = transform
        self.pre_transform = pre_transform

        # Read annotation data
        self.prot2annot, self.goterms, self.gonames, self.prot_list = self.annot_file_reader(annot_file)
        self.y_labels = self.goterms[selected_ontology]

        # Read list of proteins if a specific split is given
        self.pdb_split_list = read_list_file(pdb_split_set_file) if pdb_split_set_file else self.prot_list
        self.pdb_split_list = [prot for prot in self.pdb_split_list if os.path.exists(os.path.join(root, f'{prot}.npz'))]

        print(f" Loaded dataset with {len(self.pdb_split_list)} proteins for {selected_ontology}")

        super(PDB_Dataset, self).__init__(root, transform, pre_transform)

    @classmethod
    def annot_file_reader(cls, annot_filename):
        onts = ['molecular_function', 'biological_process', 'cellular_component']
        prot2annot = {}
        goterms = {ont: [] for ont in onts}
        gonames = {ont: [] for ont in onts}
        prot_list = []

        with open(annot_filename, mode='r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for ont in onts:
                next(reader, None)  # Skip headers
                goterms[ont] = next(reader)
                next(reader, None)  # Skip headers
                gonames[ont] = next(reader)

            next(reader, None)  # Skip headers
            for row in reader:
                prot = row[0].replace("-", "_")  # Ensure ID format consistency
                prot2annot[prot] = {ont: np.zeros(len(goterms[ont]), dtype=np.int64) for ont in onts}
                for i, ont in enumerate(onts):
                    goterm_indices = [goterms[ont].index(goterm) for goterm in row[i+1].split(',') if goterm]
                    prot2annot[prot][ont][goterm_indices] = 1.0
                prot_list.append(prot)

        return prot2annot, goterms, gonames, prot_list

    @property
    def num_classes(self):
        return len(self.y_labels)

    @property
    def processed_file_names(self):
        """Returns a list of processed filenames."""
        if self.pdb_split_list:
            return [f'data_{i}.pt' for i in range(len(self.pdb_split_list))]
        else:
            return [f'data_{i}.pt' for i in range(len(self.prot_list))]

    def process(self):
        data_list = []
        for index, prot_id in tqdm(enumerate(self.pdb_split_list), total=len(self.pdb_split_list)):
            data = self._load_data(prot_id)
            if data:
                data_list.append(data)
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
        return data_list

    def _load_data(self, prot_id):
        pdb_file = os.path.join(self.npz_dir, f'{prot_id.replace("-", "_")}.npz')
        if not os.path.isfile(pdb_file):
            print(f" File not found: {pdb_file}")
            return None

        cmap = np.load(pdb_file)
        sequence = str(cmap['seqres'])

        onehot_features = torch.tensor(seq2onehot(sequence), dtype=torch.float).squeeze(0)
        protbert_features = torch.tensor(seq2protbert(sequence), dtype=torch.float).squeeze(0)

        hydrophobicity, polarity, charge = compute_residue_features(sequence)
        additional_features = torch.tensor(np.stack([hydrophobicity, polarity, charge], axis=1), dtype=torch.float)

        residue_count = min(protbert_features.shape[0], onehot_features.shape[0])
        additional_features = additional_features[:residue_count]

        node_features = torch.cat([protbert_features, additional_features], dim=1) if self.model == "protBERT" else torch.cat([onehot_features, additional_features], dim=1)

        edge_index = self._get_adjacency_info(cmap['C_alpha'])
        labels = self._get_labels(prot_id)
        length = torch.tensor(len(sequence), dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, u=prot_id, y=labels, length=length)

    def _get_labels(self, prot_id):
        labels = {ont: torch.tensor(self.prot2annot.get(prot_id, {}).get(ont, np.zeros(len(self.y_labels), dtype=np.int64)), dtype=torch.long) for ont in ['molecular_function', 'biological_process', 'cellular_component']}

        for ont, label in labels.items():
            if label.dim() == 1:
                labels[ont] = label.unsqueeze(0)  # Add batch dimension
        
        return labels.get(self.selected_ontology, torch.zeros(len(self.y_labels), dtype=torch.long))

    def _get_adjacency_info(self, distance_matrix, threshold=8.0):
        adjacency_matrix = (distance_matrix <= threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        edge_indices = np.nonzero(adjacency_matrix)
        return torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)

    def len(self):
        return len(self.pdb_split_list)

    def get(self, idx):
        return self._load_data(self.pdb_split_list[idx])


'''# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

root = '/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/structure_files/tmp_cmap_files'
annot_file = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachintha data/nrPDB-GO_2025.02.04_annot.tsv"
num_shards = 20

test_file = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachintha data/nrPDB-GO_2025.02.04_test.txt"
train_file = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachintha data/nrPDB-GO_2025.02.04_train.txt"
valid_file = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachintha data/nrPDB-GO_2025.02.04_valid.txt"

torch.manual_seed(12345)
pdb_protBERT_dataset_test = PDB_Dataset(root=root, annot_file=annot_file, num_shards=num_shards, selected_ontology="biological_process", transform=None, pre_transform=None, model="protBERT", pdb_split_set_file=test_file)
pdb_protBERT_dataset_train = PDB_Dataset(root=root, annot_file=annot_file, num_shards=num_shards, selected_ontology="biological_process", transform=None, pre_transform=None, model ="protBERT", pdb_split_set_file=train_file)
pdb_protBERT_dataset_valid = PDB_Dataset(root=root, annot_file=annot_file, num_shards=num_shards, selected_ontology="biological_process", transform=None, pre_transform=None, model="protBERT",  pdb_split_set_file=valid_file)

print(f"Train: {len(pdb_protBERT_dataset_train)}, Test: {len(pdb_protBERT_dataset_test)}, Valid: {len(pdb_protBERT_dataset_valid)}")
print(len(pdb_protBERT_dataset_train), len(pdb_protBERT_dataset_valid[0].x[0]), pdb_protBERT_dataset_train.num_classes, pdb_protBERT_dataset_valid.num_classes)
# Paths to save the datasets
dataset_save_path = "/home/hpc_users/2019s17273@stu.cmb.ac.lk/ganeshiny/protein-go-predictor/preprocessing/data/sachintha data/pdb_datasets.pkl"

# Save datasets to a pickle file
with open(dataset_save_path, 'wb') as f:
    pickle.dump({
        'train': pdb_protBERT_dataset_train,
        'test': pdb_protBERT_dataset_test,
        'valid': pdb_protBERT_dataset_valid
    }, f)

print(f"Datasets saved to {dataset_save_path}")'''
