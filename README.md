# plant-go-predictor (WIP)

Plant Go Predictor (PGP) is a automated function predicting (AFP) model trained on specialized plant data. This model uses graph representations of experimental plant protein structure data and seqres records to predict fuctions as Gene Ontology (GO) terms. 


## To-dos

The model is predicting higher level GO terms, must do better class weight sampling to get more specific functions: Focal loss
    The problem is  more on the label encoding rather than the loss function/the class weights function

Make revisions to the label encoding method, current multihot encoding makes it loose heirachial information
- Get the entire label space, and the .obo file for the whole gene ontology (GO) up to date
- Convert GO into graph object
- Select an ontology and only obtain the sub graph that contains the label space we are interested in (the nodes)
- Use GOA2Vec or Node to Vec to generate embeddings for each node. Have all these as a dense vector (This will be similar to eye matrix in onehot encoding) - set a feature size
- For each protein get reteive the corresponding GOA2Vec embedding  for their annotated GO term
- Aggregrate them for each protein - we can do weighted aggregation ?
- Use it to train the model
- Decode it going reverse in steps above 

