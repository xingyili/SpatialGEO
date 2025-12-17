import torch
import opt
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm  # 引入 tqdm

nmi_result = []
ari_result = []


def Pretrain_gae(model, data, adj, gamma_value, lr=opt.args.lr, ground_truth=None):
    best_ari = 0
    optimizer = Adam(model.parameters(), lr=lr)

    loop = tqdm(range(opt.args.epoch), desc="Pretraining GAE")
    
    for epoch in loop: 

        z_egae, z_hat, adj_hat = model(data, adj)

        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss = loss_w + gamma_value * loss_a

        postfix_info = {'loss': '{:.4f}'.format(loss.item())}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ground_truth is not None:
            kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z_egae.data.cpu().numpy())

            ari, nmi = eva(ground_truth, kmeans.labels_, epoch)
            nmi_result.append(nmi)
            ari_result.append(ari)
            
            postfix_info['ari'] = '{:.4f}'.format(ari)

            if ari > best_ari:
                best_ari = ari
                torch.save(model.state_dict(), opt.args.model_save_path)
                postfix_info['best_ari'] = '{:.4f}'.format(best_ari)
        else:
            torch.save(model.state_dict(), opt.args.model_save_path)
        loop.set_postfix(postfix_info)