# deep-green (WIP)

Deep Green (DG) is a automated function predicting (AFP) model trained on specialized plant data. This is a  This model uses graph representations of experimental plant protein structure data and seqres records to predict fuctions as Gene Ontology (GO) terms. 


## About
The functional annotation of plant proteins lags far behind their animal counterparts, with <2% of _Viridiplantae_ sequences in UniProtKB supported by experimental evidence. Here, we present Deep Green (DG), a graph neural network trained exclusively on Viridiplantae structures that combines ProtBERT embeddings with GATv2 attention to achieve statistically significant improvements (ε < 0.1, ASO test) over state-of-the-art baselines. 
