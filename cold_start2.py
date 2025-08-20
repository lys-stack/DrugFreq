#分类
# coding: utf-8
import argparse
import numpy as np
import pandas as pd
import torch
from model2 import drugFreq, Optimizer
from myutils import *
from combine import *
import numpy as np



def generate_balanced_kfold_masks(total_rows=664, total_cols=994, n_splits=10):
    mask_matrices = []
    indices = np.arange(total_rows)
    np.random.shuffle(indices)  # 打乱索引

    # 分割成 n_splits 组
    fold_sizes = [total_rows // n_splits] * n_splits# 计算每一折的大小
    for i in range(total_rows % n_splits):
        fold_sizes[i] += 1

    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size

        # 初始化一个全训练集掩码
        mask = np.ones((total_rows, total_cols), dtype=bool)
        test_indices = indices[start:end]# 取出当前折的测试集索引
        mask[test_indices, :] = False# 将这些行设置为 False，表示测试集
        mask_matrices.append(mask)# 加入掩码矩阵
        start = end

    return mask_matrices





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drugFreq Cold Start Experiment")
    parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="the weight decay for l2 normalization")
    parser.add_argument('--layer_size', nargs='*', type=int, default=[1024,3072], help='Output sizes of every layer')
    parser.add_argument('--alpha', type=float, default=0.4, help="the scale for balance gcn and ni")
    parser.add_argument('--gamma', type=float, default=8, help="the scale for sigmoid")
    parser.add_argument('--epochs', type=int, default=3000, help="the epochs for model")
    args = parser.parse_args()


    ideal_kernel_df, ideal_kernel_values = read_csv_with_labels('./Data/Frequency_664/Drug-Side_Effect_Frequency664.csv')

    drug_similarity_matrices = [
        read_csv_with_labels("./Data/drug/664_drug_drug_scores.csv")[1],
        read_csv_with_labels("./Data/drug/664_drug_fingerprint_jaccard_similarity_matrix_new.csv")[1],
        read_csv_with_labels("./Data/drug/664_drug_disease_jaccard_matrix.csv")[1]

    ]

    # 读取副作用相似性矩阵
    side_similarity_matrices = [
        read_csv_with_labels("./Data/side/kvplm_Side_Effect_Similarity_Matrix.csv")[1],
        read_csv_with_labels("./Data/side/semantic.csv")[1],
        read_csv_with_labels("./Data/side/word_new.csv")[1]
    ]

    # 生成掩码矩阵
    mask_matrices = generate_balanced_kfold_masks(total_rows=664, total_cols=994, n_splits=10)

    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()
    AUC = 0
    AUPR = 0

    # 进行冷启动的十折交叉验证
    for fold_idx, mask in enumerate(mask_matrices):
        print(f"Fold {fold_idx + 1}/10")

        # 掩码理想核矩阵，生成当前折的训练集和验证集
        ideal_kernel_values_masked = ideal_kernel_values.copy()  # 复制原始理想核矩阵
        ideal_kernel_values_masked[mask == 0] = 0  # 应用掩码，确保模型在训练时无法访问测试集的数据

        # 计算药物和副作用的理想核矩阵
        ideal_kernel_drugs = np.dot(ideal_kernel_values_masked, ideal_kernel_values_masked.T)
        ideal_kernel_drugs = kernel_normalized(ideal_kernel_drugs)

        ideal_kernel_sides = np.dot(ideal_kernel_values_masked.T, ideal_kernel_values_masked)
        ideal_kernel_sides = kernel_normalized(ideal_kernel_sides)



        # 特征融合
        fused_drug_sim = perform_feature_fusion(
            np.arange(ideal_kernel_values_masked.shape[0]), drug_similarity_matrices, ideal_kernel_drugs, lambd=0.8,
            matrix_type='DSE_drug'
        )

        fused_side_sim = perform_feature_fusion(
            np.arange(ideal_kernel_values_masked.shape[1]), side_similarity_matrices, ideal_kernel_sides, lambd=0.8,
            matrix_type='side'
        )

        print("mask", mask)

        train_mask = mask
        test_mask = ~mask

        print("test_mask", test_mask)  # [750,994]

        # 转换为 PyTorch 张量并移动到设备
        train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool).to(args.device)
        test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool).to(args.device)
        num_true_train_mask = torch.sum(train_mask_tensor).item()
        num_true_test_mask = torch.sum(test_mask_tensor).item()

        print(f"train_mask_tensor 中 True 的数量: {num_true_train_mask}")
        print(f"test_mask_tensor 中 True 的数量: {num_true_test_mask}")

        # 转换为 PyTorch 张量，并保持数据的形状
        train_data_tensor = torch.from_numpy(ideal_kernel_values_masked).float().to(args.device) * torch.from_numpy(mask).float().to(args.device)
        # 将 train_data_tensor 中大于0的元素换成1，其他元素为0
        train_data_tensor = torch.where(train_data_tensor > 0, 1, 0)

        test_data_tensor = torch.from_numpy(ideal_kernel_values_masked).float().to(args.device) * torch.from_numpy(test_mask).float().to(args.device)


        # 初始化模型并训练
        model = drugFreq(adj_mat=train_data_tensor, drug_sim=fused_drug_sim, side_sim=fused_side_sim,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device).to(args.device)

        # 创建优化器实例，使用处理过的训练集和测试集张量
        opt = Optimizer(model, ideal_kernel_values,train_data_tensor, test_data_tensor,
                        train_mask=torch.from_numpy(train_mask).bool().to(args.device),
                        test_mask=torch.from_numpy(test_mask).bool().to(args.device),
                        ap_fun=roc_auc, aupr_fun=aupr, rmse_fun=rmse, mae_fun=mae,
                        lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)

        true_data, predict_data, auc_data, aupr_data = opt()
        # true_data, predict_data, auc_data, aupr_data, rmse_data, mae_data = opt()


        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))

        # 记录性能指标
        AUC += auc_data
        AUPR += aupr_data

    print("Best AUC: %.4f" % (AUC / 10), "Best AUPR: %.4f" % (AUPR / 10))
