import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
from abc import ABC
from myutils import *


# 构建邻接矩阵类
class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cuda:0"):
        super(ConstructAdjMatrix, self).__init__()
        self.device = device
        self.adj = torch.where(original_adj_mat > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        self.adj = self.adj.float()

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1) + 1, -0.5))#得到每个节点的出度，一步归一化并构建归一化对角矩阵 d_x
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0) + 1, -0.5))#入度
        agg_drug_lp = torch.mm(torch.mm(d_x, self.adj), d_y)#邻接矩阵的聚合，得到药物邻接矩阵
        agg_side_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1) + 1, -1)#d_c 是一个向量，表示每一行的归一化值，用于自环矩阵的构造
        self_drug_lp = torch.diag(torch.add(d_c, 1))#药物自环矩阵
        d_d = torch.pow(torch.sum(self.adj, dim=0) + 1, -1)
        self_side_lp = torch.diag(torch.add(d_d, 1))
        return agg_drug_lp, agg_side_lp, self_drug_lp, self_side_lp

# 加载特征类
class LoadFeature(nn.Module, ABC):
    def __init__(self, drug_sim, side_sim, device="cpu"):
        super(LoadFeature, self).__init__()
        drug_sim = torch.from_numpy(drug_sim).to(device).float()  # 转换为 float32
        self.drug_feat = torch_z_normalized(drug_sim, dim=1).to(device).float()  # 转换为 float32
        self.side_feat = torch.from_numpy(side_sim).to(device).float()  # 转换为 float32

    def forward(self):
        drug_feat = self.drug_feat
        side_feat = self.side_feat
        return drug_feat, side_feat

# GCN 编码器类
class GEncoder(nn.Module, ABC):
    def __init__(self, agg_d_lp, agg_s_lp, self_d_lp, self_s_lp, drug_feat, side_feat, layer_size, alpha):
        super(GEncoder, self).__init__()
        self.agg_d_lp = agg_d_lp.float()  # 转换为 float32
        self.agg_s_lp = agg_s_lp.float()  # 转换为 float32
        self.self_d_lp = self_d_lp.float()  # 转换为 float32
        self.self_s_lp = self_s_lp.float()  # 转换为 float32

        self.layers = layer_size
        self.alpha = alpha
        self.drug_feat = drug_feat.float()  # 转换为 float32
        self.side_feat = side_feat.float()  # 转换为 float32

        self.fc_drug = nn.Linear(self.drug_feat.shape[1], layer_size[0], bias=True)
        self.fc_side = nn.Linear(self.side_feat.shape[1], layer_size[0], bias=True)
        self.ld = nn.BatchNorm1d(layer_size[0])
        self.ls = nn.BatchNorm1d(layer_size[0])
        self.lm_drug = nn.Linear(layer_size[0], layer_size[1], bias=True)
        self.lm_side = nn.Linear(layer_size[0], layer_size[1], bias=True)

    def forward(self):
        drug_fc = self.ld(self.fc_drug(self.drug_feat))#降维并进行归一化
        side_fc = self.ls(self.fc_side(self.side_feat))

        drug_gcn = torch.mm(self.self_d_lp, drug_fc) + torch.mm(self.agg_d_lp, side_fc)#SD+Ld*Hs
        side_gcn = torch.mm(self.self_s_lp, side_fc) + torch.mm(self.agg_s_lp, drug_fc)

        drug_ni = torch.mul(drug_gcn, drug_fc)
        side_ni = torch.mul(side_gcn, side_fc)

        drug_emb = fun.relu(self.lm_drug((1 - self.alpha) * drug_gcn + self.alpha * drug_ni))
        side_emb = fun.relu(self.lm_side((1 - self.alpha) * side_gcn + self.alpha * side_ni))

        return drug_emb, side_emb




# GCN 解码器类
class GDecoder(nn.Module, ABC):
    def __init__(self, gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, drug_emb, side_emb):
        Corr = torch_corr_x_y(drug_emb, side_emb)#计算两个矩阵之间的相关性
        output = scale_sigmoid(Corr, alpha=self.gamma)
        return output

# GCN 模型类
class drugFreq(nn.Module, ABC):
    def __init__(self, adj_mat, drug_sim, side_sim, layer_size, alpha, gamma, device="cuda:0"):
        super(drugFreq, self).__init__()
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(drug_sim, side_sim, device=device)
        agg_drug_lp, agg_side_lp, self_drug_lp, self_side_lp = construct_adj_matrix()
        drug_feat, side_feat = loadfeat()
        self.encoder = GEncoder(agg_drug_lp, agg_side_lp, self_drug_lp, self_side_lp,
                                drug_feat, side_feat, layer_size, alpha)
        self.decoder = GDecoder(gamma=gamma)

    def forward(self):
        drug_emb, side_emb = self.encoder()
        output = self.decoder(drug_emb, side_emb)
        return output

# 优化器类
class Optimizer(nn.Module, ABC):
    def __init__(self, model,adj, train_data, test_data, test_mask, train_mask, ap_fun, aupr_fun, rmse_fun, mae_fun, lr=0.0001, wd=1e-05, epochs=800, test_freq=1500, device="cpu", lam=0.01, eps=1e-8):
        super(Optimizer, self).__init__()
        self.adj= torch.tensor(adj).to(device).float()
        self.model = model.to(device)
        self.train_data = train_data.to(device).float()  # 确保为 float32
        self.test_data = test_data.to(device).float()  # 确保为 float32
        self.test_mask = test_mask.to(device).float().bool()  # 转换为 BoolTensor
        self.train_mask = train_mask.to(device).float().bool()  # 转换为 BoolTensor

        self.ap_fun = ap_fun#评估函数-平均准确率
        self.aupr_fun = aupr_fun
        self.rmse_fun = rmse_fun
        self.mae_fun = mae_fun
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq#测试频率
        self.lam = lam
        self.eps = eps
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        for epoch in torch.arange(self.epochs):
            true_data = torch.masked_select(self.adj, self.train_mask).long()#从总的交互矩阵中根据掩码获得训练数据
            true_data_label = torch.where(true_data > 0, torch.tensor(1, device=true_data.device), true_data)
            best_predict = 0
            best_auc = 0
            best_aupr = 0
            best_rmse = float('inf')
            best_mae = float('inf')
            # print(epoch.item())
            predict_data = self.model()

            #修改
            train_data_flat = self.train_data.reshape(-1)
            predict_data_flat = predict_data.reshape(-1)
            mask_flat = self.train_mask.reshape(-1)

            # 获取非零位置的布尔掩码
            non_zero_mask =  (train_data_flat > 0) & mask_flat.bool()
            # 应用掩码
            train_data_selected = train_data_flat[non_zero_mask]
            predict_data_selected = predict_data_flat[non_zero_mask]
            mask_selected = mask_flat[non_zero_mask]


            loss = mse_loss(train_data_selected, predict_data_selected, mask_selected)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            predict_data_masked = torch.masked_select(predict_data, self.train_mask)#获取掩码为True的预测结果（根据训练掩码筛选出预测数据中用于训练的数据）

            auc = self.ap_fun(true_data_label, predict_data_masked)

            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.test_mask)

            if epoch % self.test_freq == 0:
                print(f"epoch:{epoch.item():4d} loss:{loss.item():.6f}")
        with torch.no_grad():
            self.model.eval()
            predict_data = self.model()

            self.test_mask = self.test_mask.bool()
            indices = torch.nonzero(self.test_mask, as_tuple=True)#获取所有非零标签的索引

            # 根据索引提取元素
            row_indices, col_indices = indices  # 分别是行索引和列索引
            true_test_data = self.adj[row_indices, col_indices].long()
            test_mask_np = self.test_mask.cpu().numpy()

            # 将 true_test_data 转换为 NumPy 数组
            true_test_data_np = true_test_data.cpu().numpy()

            has_positive = (true_test_data > 0).sum().item() > 0
            num_true = self.test_mask.sum().item()
            print(num_true)

            true_data_label = torch.where(true_test_data > 0, torch.tensor(1, device=true_test_data.device), true_test_data)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            non_zero_indices = torch.nonzero(true_test_data, as_tuple=True)  # 获取所有非零标签的索引

            filtered_true_data = true_test_data[non_zero_indices]  # 只选择非零标签
            filtered_predict_data = predict_data_masked[non_zero_indices]  # 对应选择预测结果中的相同部分
            rmse = self.rmse_fun(filtered_true_data, filtered_predict_data)
            mae = self.mae_fun(filtered_true_data, filtered_predict_data)

            print(f"Final Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            best_rmse = rmse
            best_mae = mae
            best_predict = predict_data_masked

        print("Fit finished.")

        return true_test_data, best_predict, best_rmse, best_mae