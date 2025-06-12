# PS-EVO: Adding evolutionary algos to PS-VAE
This entire repository is based on: [Molecule Generation by Principal Subgraph Mining and Assembling](https://github.com/THUNLP-MT/PS-VAE).\
I am just adding evolutionary algorithms to it and updating the framework.

## Installation/Environment setup

```bash
conda create --name PSEVO python=3.12
conda activate PSEVO
pip install matplotlib
pip install joblib
pip install pandas

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
#Only if conda fails, because with pip there is a bug in 2.5.1
#pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
conda install conda-forge::pytest
conda install conda-forge::rdkit
```