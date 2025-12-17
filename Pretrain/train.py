import torch
from opt import args
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import opt
from tqdm import tqdm  # 引入 tqdm

nmi_result = []
ari_result = []

def Pretrain(model, data, adj, distance,ground_truth = None):
    best_ari = 0
    optimizer = Adam(model.parameters(), lr=args.lr)
    loop = tqdm(range(opt.args.epoch), desc="Pretraining")
    for epoch in loop:

        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(data, adj, distance)

        loss_1 = F.mse_loss(x_hat, data)
        loss_2 = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_3 = F.mse_loss(adj_hat, adj.to_dense())
        loss_4 = F.mse_loss(z_ae, z_igae)  # simple aligned

        loss = loss_1 + args.alpha * loss_2 + args.beta \
               * loss_3 + args.omega * loss_4  # you can tune all kinds of hyper-parameters to get better performance.
        
        postfix_info = {'loss': '{:.4f}'.format(loss.item())}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ground_truth is not None:
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

            ari,nmi = eva(ground_truth, kmeans.labels_, epoch)
            nmi_result.append(nmi)
            ari_result.append(ari)

            postfix_info['ari'] = '{:.4f}'.format(ari)
            
            if ari > best_ari:
                best_ari = ari
                torch.save(model.state_dict(), opt.args.pre_model_save_path)
                postfix_info['best_ari'] = '{:.4f}'.format(best_ari)
        else:
            torch.save(model.state_dict(), opt.args.model_save_path)
        loop.set_postfix(postfix_info)