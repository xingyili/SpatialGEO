import opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import adjust_learning_rate
from utils import eva, target_distribution
from sklearn import metrics
import numpy as np
from tqdm import tqdm  # 引入 tqdm

nmi_result = []
ari_result = []


def Train(epoch, model, data, adj, distance, lr, pre_model_save_path, final_model_save_path, n_clusters,
          gamma_value, lambda_value, device, ground_truth=None):
    best_ari = 0
    optimizer = Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(pre_model_save_path, map_location='cpu'))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_egae, q, q1, q2, z_tilde = model(data, adj, distance)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    if ground_truth is not None:
        eva(ground_truth, cluster_id, 'Initialization')
    loop = tqdm(range(opt.args.epoch), desc="SpatialGEO training:")
    for epoch in loop:
        
        x_hat, z_hat, adj_hat, z_ae, z_egae, q, q1, q2, z_tilde = model(data, adj, distance)

        tmp_q = q.data
        p = target_distribution(tmp_q)

        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_egae = loss_w + gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_egae + lambda_value * loss_kl
        # print('{} loss: {}'.format(epoch, loss))
        # print("{} loss_egae {} loss_ae {} loss_kl".format(loss_egae, loss_ae, loss_kl))
        postfix_info = {'loss': '{:.4f}'.format(loss.item())}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ground_truth is not None:
            kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
            ARI = metrics.adjusted_rand_score(ground_truth, kmeans.labels_)
            ari,nmi = eva(ground_truth, kmeans.labels_, epoch)
            nmi_result.append(nmi)
            ari_result.append(ari)
            postfix_info['ari'] = '{:.4f}'.format(ari)
            if ARI > best_ari:
                best_ari = ARI

                torch.save(model.state_dict(), opt.args.final_model_save_path)
                postfix_info['best_ari'] = '{:.4f}'.format(best_ari)
        else:
            torch.save(model.state_dict(), opt.args.model_save_path)
        loop.set_postfix(postfix_info)