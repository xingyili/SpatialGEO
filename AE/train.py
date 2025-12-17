import torch
import opt
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

nmi_result = []
ari_result = []


def Pretrain_ae(model, dataset, train_loader, device,lr = opt.args.lr,epoch = opt.args.epoch, ground_truth = None):
    best_ari = 0
    optimizer = Adam(model.parameters(), lr=lr)
    
    # 修改处1：使用 tqdm 包装 range，并赋值给一个变量以便更新后缀信息
    loop = tqdm(range(epoch), desc="Pretraining AE")
    
    for epoch in loop: # 循环对象改为 loop

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_hat, z_ae = model(x)
            loss = F.mse_loss(x_hat, x)


            postfix_info = {'loss': '{:.4f}'.format(loss.item())}

            if ground_truth is not None:
                kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z_ae.data.cpu().numpy())

                ari,nmi = eva(ground_truth, kmeans.labels_, epoch)
                nmi_result.append(nmi)
                ari_result.append(ari)

                if ari > best_ari:
                    best_ari = ari
                    # print('New best model found with ARI: {:.4f}'.format(best_ari))
                    # 保存模型
                    torch.save(model.state_dict(), opt.args.model_save_path)
                    # print("z_ae:", z_ae_np.shape)
                
                postfix_info['ari'] = '{:.4f}'.format(ari)
                postfix_info['best_ari'] = '{:.4f}'.format(best_ari)
            else:
                torch.save(model.state_dict(), opt.args.model_save_path)
            loop.set_postfix(postfix_info)