from create_cmaps import load_GO_annot
import networkx as nx
import obonet
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder



ontology = 'biological_process'
annot_file = './preprocessing/data/pdb2go.tsv'
DIMENSIONS = 128
WALK_LENGTH = 80
NUM_WALKS = 200
WORKERS = 1
EMBEDDING_FILENAME = './preprocessing/data/go_node2vec_embeddings'
EDGES_EMBEDDING_FILENAME = './preprocessing/data/go_edges_embeddings'


prot2goterms, goterms, gonames = load_GO_annot(annot_file)

#create a subgraph with only the ontology of interest
go_label_space_of_interest = goterms[ontology]

# Creatign a graph using the .obo file
go_graph = obonet.read_obo('preprocessing/data/go-basic_2024-06-25.obo')

'''
# We can generate subgraphs from our interest, but its more efficient to stick to generating embedding for the whole thing and using the output name and annot dict to get what we want
go_graph = go_graph.subgraph(go_label_space_of_interest)
print(go_graph.number_of_nodes(), go_graph.number_of_edges())'''

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
go_node2vec = Node2Vec(go_graph, dimensions= DIMENSIONS, walk_length=WALK_LENGTH, num_walks= NUM_WALKS, workers=WORKERS)

# Embed nodes
model = go_node2vec.fit(window=10, min_count=1, batch_words=4)

#model.wv.most_similar('GO:2001286')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

#edges embedding
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
