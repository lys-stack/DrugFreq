# 回归
import argparse

from sklearn.model_selection import KFold

from model3 import drugFreq, Optimizer
from myutils import *
from combine import *
import numpy as np
import random



def generate_balanced_kfold_masks(DAL, n_splits=10):
    # 获取所有正样本和负样本位置
    positive_samples = np.array([(i, j) for i in range(DAL.shape[0]) for j in range(DAL.shape[1]) if DAL[i, j] != 0])
    negative_samples = np.array([(i, j) for i in range(DAL.shape[0]) for j in range(DAL.shape[1]) if DAL[i, j] == 0])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_masks, test_masks = [], []

    for train_idx, test_idx in kf.split(positive_samples):
        train_mask = np.zeros_like(DAL, dtype=bool)
        test_mask = np.zeros_like(DAL, dtype=bool)

        # 正样本划分
        train_pos = positive_samples[train_idx]  # 九折正样本
        test_pos = positive_samples[test_idx]    # 剩余一折正样本

        # --- 训练集掩码 ---
        # 1. 添加九折正样本
        for i, j in train_pos:
            train_mask[i, j] = True

        # 2. 添加所有原始零条目
        for i, j in negative_samples:
            train_mask[i, j] = True

        # --- 测试集掩码 ---
        # 1. 添加剩余一折正样本
        for i, j in test_pos:
            test_mask[i, j] = True

        # 2. 添加原始零条目
        for i, j in negative_samples:
            test_mask[i, j] = True

        train_masks.append(train_mask)
        test_masks.append(test_mask)

    return train_masks, test_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drugFreq Hot Start Experiment")
    parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="the weight decay for l2 normalization")
    parser.add_argument('--layer_size', nargs='*', type=int, default=[1024, 3072], help='Output sizes of every layer')
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
    train_masks, test_masks = generate_balanced_kfold_masks(ideal_kernel_values, n_splits=10)


    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()
    RMSE = 0
    MAE = 0
    n_splits=10


    for i in range(n_splits):
        print(f"Fold {i + 1}/10")
        train_mask_i = train_masks[i]
        test_mask_i = test_masks[i]

        # 掩码理想核矩阵，生成当前折的训练集和验证集
        ideal_kernel_values_masked = ideal_kernel_values.copy()  # 复制原始理想核矩阵
        ideal_kernel_values_masked[test_mask_i == 1] = 0  # 应用掩码，使得训练过程中不会使用测试集数据

        # 计算药物和副作用的理想核矩阵
        ideal_kernel_drugs = np.dot(ideal_kernel_values_masked, ideal_kernel_values_masked.T)
        ideal_kernel_drugs = kernel_normalized(ideal_kernel_drugs)

        ideal_kernel_sides = np.dot(ideal_kernel_values_masked.T, ideal_kernel_values_masked)
        ideal_kernel_sides = kernel_normalized(ideal_kernel_sides)



        # 特征融合
        fused_drug_sim = perform_feature_fusion(
            np.arange(ideal_kernel_values_masked.shape[0]), drug_similarity_matrices, ideal_kernel_drugs, lambd=0.8,
            matrix_type='drug'
        )

        fused_side_sim = perform_feature_fusion(
            np.arange(ideal_kernel_values_masked.shape[1]), side_similarity_matrices, ideal_kernel_sides, lambd=0.8,
            matrix_type='side'
        )

        # 转换为 PyTorch 张量并移动到设备
        train_mask_tensor = torch.tensor(train_mask_i, dtype=torch.bool).to(args.device)
        test_mask_tensor = torch.tensor(test_mask_i, dtype=torch.bool).to(args.device)
        num_true_train_mask = torch.sum(train_mask_tensor).item()
        num_true_test_mask = torch.sum(test_mask_tensor).item()

        print(f"train_mask_tensor 中 True 的数量: {num_true_train_mask}")
        print(f"test_mask_tensor 中 True 的数量: {num_true_test_mask}")

        # 转换为 PyTorch 张量，并保持数据的形状
        train_data_tensor = torch.from_numpy(ideal_kernel_values).float().to(args.device) * torch.from_numpy(train_mask_i).float().to(args.device)
        test_data_tensor = torch.from_numpy(ideal_kernel_values).float().to(args.device) * torch.from_numpy(test_mask_i).float().to(args.device)

        model = drugFreq(adj_mat=train_data_tensor, drug_sim=fused_drug_sim, side_sim=fused_side_sim,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device).to(args.device)

        # 创建优化器实例，使用处理过的训练集和测试集张量
        opt = Optimizer(model, ideal_kernel_values,train_data_tensor, test_data_tensor,
                        train_mask=torch.from_numpy(train_mask_i).bool().to(args.device),
                        test_mask=torch.from_numpy(test_mask_i).bool().to(args.device),
                        ap_fun=roc_auc, aupr_fun=aupr, rmse_fun=rmse, mae_fun=mae,
                        lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)

        true_data, predict_data, rmse_data, mae_data = opt()
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))

        # 记录性能指标
        RMSE += rmse_data
        MAE += mae_data


    print("Best RMSE: %.4f" % (RMSE / 10),"Best MAE: %.4f" % (MAE / 10))
