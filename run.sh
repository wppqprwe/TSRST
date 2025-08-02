#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu" or "mps"
pixel_size=0.5  # desired pixel size for the whole analysis
n_genes=1000  # number of most variable genes to impute

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt

python rescale.py "data/MBC/" --image
python preprocess.py "data/MBC/" --image
python extract_features.py "data/MBC/" --device="cuda"
python impute_locs.py "data/MBC/" --epochs=400 --device="cuda" --n-states=1
python cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=10 "data/MBP/embeddings-gene.pickle" "data/MBP/clusters-gene-256/"

# predict super-resolution gene expression
# rescale coordinates and spot radius
python rescale.py ${prefix} --locs --radius

# train gene expression prediction model and predict at super-resolution
python impute.py "data/MB/" --epochs=400 --device="cuda" --n-states=2 --testing=1 --cluster=1  # train model from scratch

# visualize imputed gene expression
python plot_imputed.py "data/151673/"

# # segment image without tissue mask
python cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=6 "data/her2st/embeddings-gene.pickle" "data/her2st/clusters-gene/"


# visualize spot-level gene expression data
python plot_spots.py "data/151673/"