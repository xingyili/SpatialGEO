import torch
from AE import AE
from GAE import IGAE
from opt import args
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from GeometricGraphLearning import GraphL

class Pre_model(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 node_input_dim, edge_input_dim, hidden_dim,
                 num_layers, dropout, augment_eps, task,
                 n_input, n_z, n_clusters, v=1.0, n_node=None, device=None):
        super(Pre_model, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.ae_model_save_path))

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

        self.gae.load_state_dict(torch.load(args.gae_model_save_path))

        self.graphl = GraphL(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,  # 隐藏层的维度
            num_layers=num_layers,  # GNN层的数量
            dropout=dropout,  # Dropout率
            augment_eps=augment_eps,  # 数据增强的epsilon值
            task=task)

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(device)
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj, distance):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        print(f"z_igae.shape:{z_igae.shape}")
        print(f"z_ae.shape:{z_igae.shape}")
        print(f"z_i.shape:{z_i.shape}")

        edge_index = adj.coalesce().indices()
        edge_features = distance[edge_index[0], edge_index[1]]
        edge_features = edge_features.view(-1, 1)

        z_g = self.graphl(z_i.float(), edge_index, edge_features)
        z_l = torch.spmm(adj, z_i)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj
        print(f"z_hat.shape:{z_hat.shape}")
        print(f"x_hat.shape:{x_hat.shape}")
        print(f"adj_hat.shape:{adj_hat.shape}")
        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde
