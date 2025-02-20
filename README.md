# plant-go-predictor (WIP)

Plant Go Predictor (PGP) is a automated function predicting (AFP) model trained on specialized plant data. This model uses graph representations of experimental plant protein structure data and seqres records to predict fuctions as Gene Ontology (GO) terms. 


<<<<<<< HEAD
## To-dos

The model is predicting higher level GO terms, must do better class weight sampling to get more specific functions: Focal loss
    The problem is  more on the label encoding rather than the loss function/the class weights function


## Important integration to make soon 
1. Now i get the GO terms represented as a column matrix as an output. This corresponds to each protein
2. We can use this as a embedding and integrate a new model, that does graph traversal from root (Biological_process) to the leave nodes!
3. That way we can get new likely functions 
4. This might need for encoding the GO DAG graph and finding a way to use the output to traverse the GO graph? for each protein
=======
>>>>>>> 8e274c386a5e71bd203947a0d377d0f8d173fc23
