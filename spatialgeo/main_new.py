import opt
import torch
import numpy as np
from SpatialGEO_model import SpatialGEO
from utils import setup_seed
from sklearn.decomposition import PCA
from load_data import LoadDataset, load_graph, construct_knn_graph
from train_new import Train
import scanpy as sc
import pandas as pd
from graph_func import graph_construction, get_distance
setup_seed(opt.args.seed)

print("network setting…")

opt.args.n_clusters = 7
opt.args.n_input = 2000
opt.args.k = 10
node_input_dim = 20  # 节点输入特征的维度
edge_input_dim = 1   # 边输入特征的维度
hidden_dim = 20     # 隐藏层的维度
num_layers = 4      # GNN层的数量
dropout = 0.2       # Dropout率
augment_eps = 0.1   # 数据增强的epsilon值
task = 'your_task' 
### cuda
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
### root
opt.args.data_path = '../data/151507/151507.h5ad'.format(opt.args.name,opt.args.name)
opt.args.pre_model_save_path = '../model/model_pretrain/{}_pretrain.pkl'.format(opt.args.name)
opt.args.final_model_save_path = '../model/model_spatialgeo/{}_spatialgeo.pkl'.format(opt.args.name)

### data pre-processing

adata = sc.read(opt.args.data_path)

adata.var_names_make_unique()
        # preprocess

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

distance = get_distance(adata)
distance_tensor = torch.tensor(distance, dtype=torch.float).to(device)

###  model definition
model = SpatialGEO(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
             ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
             gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
             node_input_dim=node_input_dim,
             edge_input_dim=edge_input_dim,
             hidden_dim=hidden_dim,  # 隐藏层的维度
             num_layers = num_layers,  # GNN层的数量
             dropout = dropout,  # Dropout率
             augment_eps = augment_eps,  # 数据增强的epsilon值
             task = task,
             n_input=opt.args.n_input,
             n_z=opt.args.n_z,
             n_clusters=opt.args.n_clusters,
             v=opt.args.freedom_degree,
             n_node=data.size()[0],
             device=device).to(device)

### training
print("Training on {}…".format(opt.args.name))
print(opt.args.lr)
Train(opt.args.epoch, model, data, adj,  distance_tensor, opt.args.lr, opt.args.pre_model_save_path, opt.args.final_model_save_path,
      opt.args.n_clusters, opt.args.gamma_value, opt.args.lambda_value, device,ground_truth=ground_truth,)
