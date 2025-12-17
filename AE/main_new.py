import torch
from AE import AE
import numpy as np
from scipy.sparse import issparse
from opt import args
from sklearn.decomposition import PCA
from utils import setup_seed
from torch.utils.data import Dataset, DataLoader
from train import Pretrain_ae
import scanpy as sc
import pandas as pd
setup_seed(1)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# adata = sc.read_visium("./data/151507")
# adata.var_names_make_unique()
# # preprocess
# adata.layers['count'] = adata.X.toarray()

# ### add ground truth
# Ann_df = pd.read_csv("./data/151507/truth.txt", sep='\t', header=None, index_col=0)
# Ann_df.columns = ['Ground Truth']
# adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
# adata = adata[~pd.isnull(adata.obs['ground_truth'])]
# adata = adata[adata.obs['ground_truth'] != 'NAN']

# adata.write('./data/151507/151507.h5ad')
args.data_path = '../data/151507/151507.h5ad'.format(args.name,args.name)
args.model_save_path = '../model/model_ae/{}_ae.pkl'.format(args.name)

print("Data: {}".format(args.data_path))
print("Label: {}".format(args.label_path))


class LoadDataset(Dataset):
    def __init__(self, data):
        if issparse(data):
            self.x = data.toarray()
        else:
            self.x = np.array(data, dtype=np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))
    
adata = sc.read(args.data_path)
print(adata)

# preprocess
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
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

model = AE(
    ae_n_enc_1=args.ae_n_enc_1,
    ae_n_enc_2=args.ae_n_enc_2,
    ae_n_enc_3=args.ae_n_enc_3,
    ae_n_dec_1=args.ae_n_dec_1,
    ae_n_dec_2=args.ae_n_dec_2,
    ae_n_dec_3=args.ae_n_dec_3,
    n_input=args.n_input,
    n_z=args.n_z).to(device)

Pretrain_ae(model, dataset, train_loader, device,ground_truth=ground_truth)


