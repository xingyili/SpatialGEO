import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances as pair
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import os

def construct_knn_graph(adata, k_neighbors, file_path):
    """
    根据空间坐标使用k-近邻算法构造图，并将边的关系保存为txt文件。

    参数:
    - adata: AnnData对象，包含空间坐标在adata.obsm['spatial']。
    - k_neighbors: int, 每个点的邻居数量。
    - file_path: str, 保存邻接矩阵的文件路径。如果为None，则不保存文件。

    返回:
    - adj: numpy.ndarray, 表示图的邻接矩阵。
    """
    k_neighbors += 1
    # 获取空间坐标
    X_spatial = adata.obsm['spatial']

    # 计算距离矩阵
    distances = pairwise_distances(X_spatial, metric='euclidean')

    # 应用k-近邻算法
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X_spatial)
    knn_indices = neigh.kneighbors(X_spatial, return_distance=False)

    # 构建邻接矩阵
    num_points = X_spatial.shape[0]
    adj = np.zeros((num_points, num_points), dtype=np.int8)
    for i in range(num_points):
        # 只遍历实际的邻居数量，并确保不连接自己
        neighbors = knn_indices[i][knn_indices[i] != i]
        for neighbor in neighbors:
            adj[i, neighbor] = 1  # 直接赋值，不考虑是否已存在

    # 如果需要保存到文件
    if file_path is not None:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            print(1)
            os.makedirs(directory)

        with open(file_path, 'w') as f:
            for i in range(num_points):
                for j in range(num_points):
                    if adj[i, j] == 1:
                        f.write(f"{i + 1} {j + 1}\n")

    return adj

def load_graph(k, graph_k_save_path, graph_save_path, data_path):
    if k:
        path = graph_k_save_path
    else:
        path = graph_save_path

    print("Loading path:", path)

    data = np.loadtxt(data_path, dtype=float)
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges_unordered -= 1

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))
