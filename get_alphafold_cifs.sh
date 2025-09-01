#!/bin/bash

ACCESS_DATE=$(date +"%Y-%m-%d")

# Create a directory for downloading the alphafold structures
AF_DIR="af_cifs"
mkdir -p "./examples/"$AF_DIR
cd "./examples/"$AF_DIR

# Download using the FTP links from the AF download sites
printf "\n\n DOWNLOADING CIF FILES FROM ALPHAFOLD ... \n"
wget -O "af_structures_${ACCESS_DATE}.tar" https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000059680_39947_ORYSJ_v4.tar

tar -xvf "af_structures_"${ACCESS_DATE}.tar

# From the downloaded files remove the pdb.gz files and only keep the cif.gz files 
cd "./examples/af_structures_"${ACCESS_DATE}.tar
find . -name ".pdb.gz" -type f -delete

# gunzip the cif.gz files (I think the code just runs with cif.gz so this no need)
