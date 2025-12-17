import opt
import torch
import numpy as np
from GAE import EGAE
from utils import setup_seed
from train import Pretrain_gae
from sklearn.decomposition import PCA
from load_data import LoadDataset, load_graph
import scanpy as sc
import os
from graph_func import graph_construction
import pandas as pd
setup_seed(1)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

opt.args.data_path = '../data/151507/151507.h5ad'.format(opt.args.name,opt.args.name)
opt.args.model_save_path = '../model/model_gae/{}_gae.pkl'.format(opt.args.name)

adata = sc.read(opt.args.data_path)
print(adata)
adata.var_names_make_unique()
 
adata.layers['count'] = adata.X
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)


x = adata.X
ground_truth = adata.obs["ground_truth"]
ground_truth = pd.Categorical(ground_truth).codes

dataset = LoadDataset(x)

adj = graph_construction(adata,opt.args.k,50,'KNN')
adj = adj['adj_norm'].to(device)
data = torch.Tensor(dataset.x).to(device)

print(data.is_sparse)
print(adj)

model_gae = EGAE(
    gae_n_enc_1=opt.args.gae_n_enc_1,
    gae_n_enc_2=opt.args.gae_n_enc_2,
    gae_n_enc_3=opt.args.gae_n_enc_3,
    gae_n_dec_1=opt.args.gae_n_dec_1,
    gae_n_dec_2=opt.args.gae_n_dec_2,
    gae_n_dec_3=opt.args.gae_n_dec_3,
    n_input=opt.args.n_components,
).to(device)

Pretrain_gae(model_gae, data, adj,  opt.args.gamma_value,ground_truth=ground_truth,)
