#!/bin/bash

DATA_DIR="data"
ACCESS_DATE=$(date +"%Y-%m-%d")

mkdir $DATA_DIR
printf "\n\n  CREATING DATA DIRECTORY \n" $DATA_DIR

#printf "\n\n  DOWNLOADING SEQUENCES\n"
#wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz -O $DATA_DIR/uniref100_${ACCESS_DATE}.fasta.gz

#printf "\n\n  DOWNLOADING GO HIERARCHY...\n"
#wget https://purl.obolibrary.org/obo/go/go-basic.obo -O $DATA_DIR/go-basic_${ACCESS_DATE}.obo

#printf "\n\n DOWNLOADING SIFTs PDB-GO SUMMARY FILE... \n"
#wget "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_go.tsv.gz" -O $DATA_DIR/pdb_chain_go.tsv_${ACCESS_DATE}.gz

#wget https://cdn.rcsb.org/resources/sequence/clusters/bc-50.out -O $DATA_DIR/bc-50%.out

#wget "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-50.txt" -O $DATA_DIR/clustered_chains_50%.txt
#wget "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-60.txt" -O $DATA_DIR/clustered_chains_60%.txt
#wget "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-70.txt" -O $DATA_DIR/clustered_chains_70%.txt
#wget "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-80.txt" -O $DATA_DIR/clustered_chains_80%.txt
#wget "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-90.txt" -O $DATA_DIR/clustered_chains_90%.txt
#wget "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-100.txt" -O $DATA_DIR/clustered_chains_100%.txt


#There are better ways to do this xD
cd $DATA_DIR
mkdir "structure_files"
cd ../
chmod +x batch_download.sh
./batch_download.sh -f preprocessing/data/arabidopsis_pdb_ids_2024-06-25.txt -o $DATA_DIR/structure_files/ -c 
gunzip $DATA_DIR/*.gz

