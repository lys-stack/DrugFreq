import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import pubchempy as pcp
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,mean_absolute_error,mean_squared_error
import itertools as it
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from scipy.stats import spearmanr


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distribute_compute(lr_list, wd_list, scale_list, layer_size, sigma_list, beta_list, workers: int, id: int):
    all_list = []
    for lr, wd, sc, la, sg, bt in it.product(lr_list, wd_list, scale_list, layer_size, sigma_list, beta_list):
        all_list.append([lr, wd, sc, la, sg, bt])
    list = np.array_split(all_list, workers)
    return list[id]


def get_fingerprint(x):
    """
    :param x: DSE_drug cid
    :return: fingerprint
    """
    drug_data = pcp.Compound.from_cid(x)
    fingerprint = ""
    for x in drug_data.fingerprint:
        fingerprint += "{:04b}".format(int(x, 16))
    finger = np.array([int(x) for x in fingerprint])
    return finger


def save_fingerprint(cid_list, last_cid, fpath):
    """
    :param cid_list: cid list
    :param last_cid: last cid, -1 begining
    :param fpath: save path
    :return: file save to path
    """
    if last_cid > 0:
        index = np.where(np.array(cid_list) == last_cid)[0].tolist()[0] + 1
    else:
        index = last_cid + 1
    length = len(cid_list)
    if index < length:
        for i in range(index, length):
            cid = cid_list[i]
            fing = get_fingerprint(cid)
            path = fpath + str(cid)
            np.save(path, fing)
            print(cid, "OK.")
            time.sleep(1)
    else:
        print("All compound has finished!")
    return None


def read_fingerprint_cid(path: str):
    """
    :param path: file path
    :return: fingerprint and cid
    """
    file_list = os.listdir(path)
    fingerprint = np.array([], dtype=np.int)
    cid = []
    for name in file_list:
        if name.endswith("npy"):
            fpath = path + name
            fing = np.load(fpath)
            fingerprint = np.hstack((fingerprint, fing))
            cid.append(int(name.split(".")[0]))
    fingerprint = fingerprint.reshape((-1, 920))
    return fingerprint, cid


def common_data_index(data_for_index: np.ndarray, data_for_cmp: np.ndarray):
    """
    :param data_for_index: data for index, numpy array
    :param data_for_cmp: data for compare, numpy array
    :return: index of common data in data for index
    """
    index = np.array([np.where(x in data_for_cmp, 1, 0) for x in data_for_index])
    index = np.where(index == 1)[0]
    return index

'''将输入的矩阵（无论是否是稀疏矩阵）转换为一种标准的稀疏数据格式'''
def to_coo_matrix(adj_mat: np.ndarray or sp.coo.coo_matrix):
    """
    :param adj_mat: adj matrix, numpy array
    :return: sparse matrix, sp.coo.coo_matrix
    """
    if not sp.isspmatrix_coo(adj_mat):
        adj_mat = sp.coo_matrix(adj_mat)
    return adj_mat
#
def mse_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor):
    """
        :param true_data: true data
        :param predict_data: predict data
        :param masked: data mask
        :return: mean square error
        """
    # true_data = torch.mul(true_data, masked)
    # predict_data = torch.mul(predict_data, masked)
    # loss_fun = nn.MSELoss()
    # loss = loss_fun(predict_data, true_data)
    # return loss
    true_data = torch.mul(true_data, masked)
    predict_data = torch.mul(predict_data, masked)
    loss_fun = nn.MSELoss()
    loss = loss_fun(predict_data, true_data)

    return loss


# def new_loss(true_data: torch.Tensor, predict_data: torch.Tensor,lam=0.03,eps=0.5):
#     """
#         :param true_data: true data
#         :param predict_data: predict data
#         :param masked: data mask
#         :return: mean square error
#         """
#     # true_data = torch.mul(true_data, masked)
#     # predict_data = torch.mul(predict_data, masked)
#     x0 = torch.where(true_data == 0)
#     x1 = torch.where(true_data != 0)
#     loss = (torch.sum((predict_data[x1] - true_data[x1]) ** 2) + lam * torch.sum((predict_data[x0] - eps) ** 2))
#     return loss
def new_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor,lam=0.03,eps=0.5):
    """
        :param true_data: true data
        :param predict_data: predict data
        :param masked: data mask
        :return: mean square error
        """
    true_data = torch.mul(true_data, masked)
    predict_data = torch.mul(predict_data, masked)
    x0 = torch.where(true_data == 0)
    x1 = torch.where(true_data != 0)
    loss = (torch.sum((predict_data[x1] - true_data[x1]) ** 2) + lam * torch.sum((predict_data[x0] - eps) ** 2))
    return loss

"""改为关联所用的损失函数"""
def binary_cross_entropy_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor, lam=1, eps=0.5):
    """
        :param true_data: true data
        :param predict_data: predict data (after sigmoid)
        :param masked: data mask
        :param lam: regularization parameter
        :param eps: for handling zero values in the true data
        :return: binary cross entropy loss with regularization
    """
    # Apply mask to true data and predicted data
    true_data = torch.mul(true_data, masked)
    predict_data = torch.mul(predict_data, masked)

    # Apply BCE Loss for binary classification (expecting probabilities after sigmoid)
    criterion = nn.BCELoss(reduction='sum')

    # Calculate binary cross entropy loss (for binary classification)
    loss = criterion(predict_data, true_data)

    # Regularization term for the zero values in true_data (to handle missing entries)
    x0 = torch.where(true_data == 0)
    loss += lam * torch.sum((predict_data[x0] - eps) ** 2)

    return loss


'''可视化多分类损失函数'''
def cross_entropy_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor):
    """
    :param true_data: true data (labels), shape: [batch_size]
    :param predict_data: predict data (logits), shape: [batch_size, num_classes]
    :param masked: data mask, shape: [batch_size]
    :return: cross entropy loss
    """
    # Apply mask to filter valid data points
    valid_indices = torch.nonzero(masked).squeeze()
    true_data = true_data[valid_indices]
    predict_data = predict_data[valid_indices]

    # Create CrossEntropyLoss instance
    criterion = nn.CrossEntropyLoss()

    # Calculate cross entropy loss
    loss = criterion(predict_data, true_data.long())

    return loss


# def combined_loss(true_data, predict_data, mask, alpha=0.5):
#     """
#     自定义损失函数，结合 MSE 和 MAE
#     :param true_data: 真实标签
#     :param predict_data: 模型预测的标签
#     :param mask: 用于选择有效数据的掩码
#     :param alpha: 平衡 MSE 和 MAE 的权重系数
#     :return: 计算出的组合损失
#     """
#     true_data_masked = torch.masked_select(true_data, mask)
#     predict_data_masked = torch.masked_select(predict_data, mask)
#
#     mse = torch.mean((true_data_masked - predict_data_masked) ** 2)
#     mae = torch.mean(torch.abs(true_data_masked - predict_data_masked))
#
#     # 组合 MSE 和 MAE
#     loss = alpha * mse + (1 - alpha) * mae
#     return loss


def cross_entropy_loss(true_data: torch.Tensor, predict_data: torch.Tensor, masked: torch.Tensor):
    """
    :param true_data: true data
    :param predict_data: predict data
    :param masked: data mask
    :return: cross entropy loss
    """
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.BCELoss()
    loss = loss_fun(predict_data, true_data)
    return loss


def mask(positive: sp.coo.coo_matrix, negative: sp.coo.coo_matrix, dtype=int):
    """
    :param positive: positive data
    :param negative: negative data
    :param dtype: return data type
    :return: data mask
    """
    row = np.hstack((positive.row, negative.row))
    col = np.hstack((positive.col, negative.col))
    data = [1] * row.shape[0]
    masked = sp.coo_matrix((data, (row, col)), shape=positive.shape).toarray().astype(dtype)
    masked = torch.from_numpy(masked)
    return masked


def to_tensor(positive, identity=False):
    """
    :param positive: positive sample
    :param identity: if add identity
    :return: tensor
    """
    if identity:
        data = positive + sp.identity(positive.shape[0])
    else:
        data = positive
    data = torch.from_numpy(data.toarray()).float()
    return data


def evaluate_all(true_data: torch.Tensor, predict_data: torch.Tensor):
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    auc = roc_auc(true_data, predict_data)
    ap = ap_score(true_data, predict_data)
    f1, thresholds = f1_score_binary(true_data, predict_data)
    acc = accuracy_binary(true_data, predict_data, thresholds)
    precision = precision_binary(true_data, predict_data, thresholds)
    recall = recall_binary(true_data, predict_data, thresholds)
    mcc = mcc_binary(true_data, predict_data, thresholds)
    return auc, ap, acc, f1, mcc


def evaluate_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    auc = roc_auc(true_data, predict_data)
    ap = ap_score(true_data, predict_data)
    return auc, ap


def ap_score(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    area under the precision-recall curve
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: ap
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    area = average_precision_score(y_true=true_data.detach().cpu().numpy(), y_score=predict_data.detach().cpu().numpy())
    return area

# '''斯皮尔曼等级相关系数（SCC）'''
# def scc(true_data: torch.Tensor, predict_data: torch.Tensor):
#     """
#     :param true_data: true values, torch tensor 1D
#     :param predict_data: predicted values, torch tensor 1D
#     :return: Spearman correlation coefficient
#     """
#     # 确保输入是 1D 张量
#     assert true_data.dim() == 1 and predict_data.dim() == 1, "Input tensors must be 1D!"
#
#     # 将张量转换为 NumPy 数组
#     true_data_np = true_data.detach().cpu().numpy()
#     predict_data_np = predict_data.detach().cpu().numpy()
#
#     # 计算 Spearman 相关系数
#     correlation, _ = spearmanr(true_data_np, predict_data_np)
#
#     return correlation


'''计算Roc曲线下面积（AUC）'''
def roc_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: AUC score
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"#断言语句，确保 true_data 中所有元素都在 0 和 1 的范围内。
    aucs = roc_auc_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    return aucs

'''受试者工作特征曲线下面积（AUROC）'''
def auroc(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    计算 AUROC（Area Under the Receiver Operating Characteristic Curve）

    :param true_data: 真实标签，torch tensor 1D，值应为 0 或 1
    :param predict_data: 预测值，torch tensor 1D
    :return: AUROC score
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"

    # 将 tensor 转换为 numpy 数组
    true_data_np = true_data.detach().cpu().numpy()
    predict_data_np = predict_data.detach().cpu().numpy()

    # 计算假阳性率 (FPR) 和真正率 (TPR)
    fpr, tpr, _ = roc_curve(true_data_np, predict_data_np)

    # 计算 AUC（即 AUROC）
    auroc_score = auc(fpr, tpr)

    return auroc_score

'''PR 曲线下的面积（AUPR）,用于评估二分类器的预测能力'''
def aupr(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true labels, torch tensor 1D
    :param predict_data: predicted scores, torch tensor 1D
    :return: AUPR score
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    aupr_score = average_precision_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    return aupr_score

'''计算平均绝对误差（MAE）'''
def mae(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true labels, torch tensor 1D
    :param predict_data: predicted scores, torch tensor 1D
    :return: MAE score
    """
    mae_score = mean_absolute_error(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    return mae_score

'''计算均方根误差（RMSE）'''
def rmse(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true labels, torch tensor 1D
    :param predict_data: predicted scores, torch tensor 1D
    :return: RMSE score
    """
    mse_score = mean_squared_error(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    rmse_score = np.sqrt(mse_score)
    return rmse_score



'''计算F1分数'''
def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true data,torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: max F1 score and threshold
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    with torch.no_grad():
        thresholds = torch.unique(predict_data)
    size = torch.tensor([thresholds.size()[0], true_data.size()[0]], dtype=torch.int32, device=true_data.device)
    ones = torch.ones([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.view([1, -1]).ge(thresholds.view([-1, 1])), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data.view([1, -1])), ones, zeros), dim=1)
    tp = torch.sum(torch.mul(predict_value, true_data.view([1, -1])), dim=1)
    two = torch.tensor(2, dtype=torch.float32, device=true_data.device)
    n = torch.tensor(size[1].item(), dtype=torch.float32, device=true_data.device)
    scores = torch.div(torch.mul(two, tp), torch.add(n, torch.sub(torch.mul(two, tp), tpn)))
    max_f1_score = torch.max(scores)
    threshold = thresholds[torch.argmax(scores)]
    return max_f1_score, threshold

'''计算准确率'''
def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: acc
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    n = true_data.size()[0]
    ones = torch.ones(n, dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(n, dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data), ones, zeros))
    score = torch.div(tpn, n)
    return score

'''计算精确率'''
def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    true_neg = torch.sub(ones, true_data)
    tf = torch.sum(torch.mul(true_neg, predict_value))
    score = torch.div(tp, torch.add(tp, tf))
    return score

'''计算召回率'''
def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    predict_neg = torch.sub(ones, predict_value)
    fn = torch.sum(torch.mul(predict_neg, true_data))
    score = torch.div(tp, torch.add(tp, fn))
    return score

'''计算二分类问题中的 Matthews相关系数 (MCC)'''
def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    predict_neg = torch.sub(ones, predict_value)
    true_neg = torch.sub(ones, true_data)
    tp = torch.sum(torch.mul(true_data, predict_value))
    tn = torch.sum(torch.mul(true_neg, predict_neg))
    fp = torch.sum(torch.mul(true_neg, predict_value))
    fn = torch.sum(torch.mul(true_data, predict_neg))
    delta = torch.tensor(0.00001, dtype=torch.float32, device=true_data.device)
    score = torch.div((tp * tn - fp * fn), torch.add(delta, torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    return score

'''计算二维Torch张量的相关系数矩阵'''
def torch_corr(tensor: torch.Tensor, dim=0):
    """
    :param tensor: an 2D torch tensor
    :param dim:
        0 : Calculate row correlation
        1 : Calculate col correlation
    :return: correlation coefficient
    """
    mean = torch.mean(tensor, dim=1 - dim)
    if dim:
        tensor_mean = torch.sub(tensor, mean)
        tensor_cov = torch.mm(torch.t(tensor_mean), tensor_mean)
    else:
        mean = mean.view([mean.size()[0], -1])
        tensor_mean = torch.sub(tensor, mean)
        tensor_cov = torch.mm(tensor_mean, torch.t(tensor_mean))
    diag = torch.diag(tensor_cov)
    diag = torch.sqrt(diag)
    diag = torch.mm(diag.view([-1, 1]), diag.view([1, -1]))
    tensor_corr = torch.div(tensor_cov, diag)
    return tensor_corr

'''计算两个矩阵的相关系数'''
def torch_corr_x_y(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    :param tensor1: a matrix, torch Tensor
    :param tensor2: a matrix, torch Tensor
    :return: corr(tensor1, tensor2)
    """
    assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
    tensor2 = torch.t(tensor2)
    mean1 = torch.mean(tensor1, dim=1).view([-1, 1])#计算 tensor1 的每一行的均值，并将结果变形为列向量（n × 1张量）
    mean2 = torch.mean(tensor2, dim=0).view([1, -1])#计算 tensor2 的每一列的均值，并将结果变形为行向量（1 × n张量）
    lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))#计算 tensor1 和 tensor2 的协方差矩阵，完成中心化操作
    lxx = torch.diag(torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1))))#计算 tensor1 的每一行的方差
    lyy = torch.diag(torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2)))#计算 tensor2 的每一列的方差
    std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))#计算 tensor1 和 tensor2 的标准差
    corr_x_y = torch.div(lxy, std_x_y)#计算 tensor1 和 tensor2 的相关系数
    return corr_x_y


def torch_euclidean_dist(tensor: torch.Tensor, dim=0):
    """
    :param tensor: a 2D torch tensor
    :param dim:
        0 : represent row
        1 : represent col
    :return: return euclidean distance
    """
    if dim:
        tensor_mul_tensor = torch.mm(torch.t(tensor), tensor)
    else:
        tensor_mul_tensor = torch.mm(tensor, torch.t(tensor))
    diag = torch.diag(tensor_mul_tensor)
    n_diag = diag.size()[0]
    tensor_diag = diag.repeat([n_diag, 1])
    diag = diag.view([n_diag, -1])
    dist = torch.sub(torch.add(tensor_diag, diag), torch.mul(tensor_mul_tensor, 2))
    dist = torch.sqrt(dist)
    return dist


def torch_dist(tensor: torch.Tensor, p=0 or int):
    """
    :param tensor: an 2D tensor
    :param p: pow
    :return: distance between rows
    """
    size = tensor.size()
    tensor_flatten = torch.flatten(tensor)
    tensor_mat = tensor.repeat([1, 1, size[0]])
    tensor_flatten = tensor_flatten.repeat([1, size[0], 1])
    tensor_sub = torch.sub(tensor_mat, tensor_flatten)
    tensor_sub = tensor_sub.view([size[0], size[0], size[1]])
    tensor_sub = torch.abs(tensor_sub)
    if p == 0:
        tensor_sub = torch.pow(tensor_sub, p)
        dist = torch.sum(tensor_sub, dim=2)
        diag = torch.diag(dist)
        dist = torch.sub(dist, torch.diag(diag))
    elif p == 1:
        dist = torch.sum(tensor_sub, dim=2)
    else:
        tensor_sub = torch.pow(tensor_sub, p)
        dist = torch.sum(tensor_sub, dim=2)
        dist = torch.pow(dist, 1 / p)
    return dist


def torch_z_normalized(tensor: torch.Tensor, dim=0):
    """
    :param tensor: an 2D torch tensor
    :param dim:
        0 : normalize row data
        1 : normalize col data
    :return: Gaussian normalized tensor
    """
    mean = torch.mean(tensor, dim=1 - dim)
    std = torch.std(tensor, dim=1 - dim)
    if dim:
        tensor_sub_mean = torch.sub(tensor, mean)
        tensor_normalized = torch.div(tensor_sub_mean, std)
    else:
        size = mean.size()[0]
        tensor_sub_mean = torch.sub(tensor, mean.view([size, -1]))
        tensor_normalized = torch.div(tensor_sub_mean, std.view([size, -1]))
    return tensor_normalized


def exp_similarity(tensor: torch.Tensor, sigma: torch.Tensor, normalize=True):
    """
    :param tensor: an torch tensor
    :param sigma: scale parameter
    :param normalize: normalize or not
    :return: exponential similarity
    """
    if normalize:
        tensor = torch_z_normalized(tensor, dim=1)
    tensor_dist = torch_euclidean_dist(tensor, dim=0)
    exp_dist = torch.exp(-tensor_dist / (2 * torch.pow(sigma, 2)))
    return exp_dist


def jaccard_coef(tensor: torch.Tensor):
    """
    :param tensor: an torch tensor, 2D
    :return: jaccard coefficient
    """
    assert torch.all(tensor.le(1)) and torch.all(tensor.ge(0)), "Value must be 0 or 1"
    size = tensor.size()
    tensor_3d = torch.flatten(tensor).repeat([size[0]]).view([size[0], size[0], size[1]])
    ones = torch.ones(tensor_3d.size(), dtype=torch.float32, device=tensor.device)
    zeros = torch.zeros(tensor_3d.size(), dtype=torch.float32, device=tensor.device)
    tensor_3d = torch.add(tensor_3d, tensor.view([size[0], 1, size[1]]))
    intersection = torch.where(tensor_3d.eq(2), ones, zeros)
    union = torch.where(tensor_3d.eq(2), ones, tensor_3d)
    intersection = torch.sum(intersection, dim=2)
    union = torch.sum(union, dim=2)
    union = torch.where(union.eq(0), torch.add(union, 0.1), union)
    eye = torch.eye(union.size()[0], dtype=torch.float32, device=tensor.device)
    jaccard = torch.div(intersection, union)
    jaccard = torch.where(jaccard.eq(0), eye, jaccard)
    return jaccard


def full_kernel(exp_dist: torch.Tensor):
    """
    :param exp_dist: exponential similarity
    :return: full kernel
    """
    n = exp_dist.shape[0]
    ones = torch.ones(n, n, dtype=torch.float, device=exp_dist.device)
    one = torch.diag(torch.diag(ones))
    mask_diag = torch.mul(torch.sub(ones, one), exp_dist)
    mask_diag_sum = torch.sum(mask_diag, dim=1).view([n, -1])
    mask_diag = torch.div(mask_diag, 2 * mask_diag_sum)
    mask_diag = torch.add(mask_diag, 0.5 * one)
    return mask_diag


def sparse_kernel(exp_dist: torch.Tensor, k: int):
    """
    :param exp_dist: exponential similarity
    :param k: knn k
    :return: sparse kernel
    """
    n = exp_dist.shape[0]
    maxk = torch.topk(exp_dist, k, dim=1)
    mink_1 = torch.topk(exp_dist, n - k, dim=1, largest=False)
    index = torch.arange(n, device=exp_dist.device).view([n, -1])
    exp_dist[index, mink_1.indices] = 0
    knn_sum = torch.sum(maxk.values, dim=1).view([n, -1])
    exp_dist = torch.div(exp_dist, knn_sum)
    return exp_dist
def scale_sigmoid(x, alpha=1.0):
    """
    自定义的 scale_sigmoid 函数，用于将输出值范围控制在 [0, 5]
    :param x: 输入值
    :param alpha: 控制 sigmoid 变化速率的参数
    :return: 缩放后的输出值，范围在 [0, 5] 之间
    """
    scaled_sigmoid = torch.sigmoid(alpha * x) * 5
    return scaled_sigmoid


def scale_sigmoid2(x, alpha=1.0):
    """
    自定义的 scale_sigmoid 函数，用于将输出值范围控制在 [0, 5]
    :param x: 输入值
    :param alpha: 控制 sigmoid 变化速率的参数
    :return: 缩放后的输出值，范围在 [0, 1] 之间
    """
    scaled_sigmoid = torch.sigmoid(alpha * x)
    return scaled_sigmoid


def scale_softplus(x, gamma=1.0):
    """
    自定义的 scale_softplus 函数，用于将输出值范围控制在 [0, 5]
    :param x: 输入值
    :param gamma: 控制 Softplus 变化速率的参数
    :return: 缩放后的输出值，范围在 [0, 5] 之间
    """
    # 计算 Softplus 激活函数
    softplus_output = torch.log(1 + torch.exp(gamma * x))

    # 将 Softplus 输出缩放到 [0, 5] 范围
    max_softplus = torch.max(softplus_output)  # 获取 Softplus 输出的最大值
    scaled_output = 5 * (softplus_output / max_softplus)  # 缩放到 [0, 5]

    return scaled_output


def corr_scale_sigmoid(tensor: torch.Tensor, alpha: int or float):
    """
    :param tensor: a torch tensor, range is [-1, 1]
    :param alpha: an scale parameter to sigmod
    :return: mapping tensor to [0, 1]
    """
    alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
    output = torch.sigmoid(torch.mul(alpha, tensor))
    return output


def np_delete_value(arr: np.ndarray, obj: np.ndarray):
    """
    :param arr: 1-D vector, narray
    :param obj: 1-D vector, value that would be removed , narray
    :return: after removed vector
    """
    index = [np.where(x == arr)[0][0] for x in obj if x in arr]
    arr = np.delete(arr, index)
    return arr


def translate_result(tensor: torch.Tensor or np.ndarray):
    """
    :param tensor: torch tensor or np.ndarray
    :return: pd.DataFrame
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    arr = tensor.reshape((1, -1))
    arr = pd.DataFrame(arr)
    return arr


def calculate_train_test_index(response: np.ndarray, pos_train_index: np.ndarray, pos_test_index: np.ndarray):
    """
    :param response: response vector, np.ndarray
    :param pos_train_index: positive train index in response
    :param pos_test_index: positive test index in reaponse
    :return: train index and test index
    """
    neg_response_index = np.where(response == 0)[0]
    neg_test_index = np.random.choice(neg_response_index, pos_test_index.shape[0])
    neg_train_index = np_delete_value(neg_response_index, neg_test_index)
    test_index = np.hstack((pos_test_index, neg_test_index))
    train_index = np.hstack((pos_train_index, neg_train_index))
    return train_index, test_index


def dir_path(k=1):
    """
    :param k: 当前路径后退级数
    :return: 后退k级后的目录
    """
    fpath = os.path.realpath(__file__)
    dir_name = os.path.dirname(fpath)
    dir_name = dir_name.replace("\\", "/")
    p = len(dir_name) - 1
    while p > 0:
        if dir_name[p] == "/":
            k -= 1
            if k == 0:
                break
        p -= 1
    p += 1
    dir_name = dir_name[0: p]
    return dir_name


def extract_row_data(data: pd.DataFrame, row: int):
    """
    :param data: DataFrame
    :param row: row index, int
    :return: not nan data
    """
    data = np.array(data, dtype=np.float32)
    target = data[row, :]
    target = target[np.where(~np.isnan(target))[0]]
    return target


def transfer_data(data: pd.DataFrame, label: str):
    lenght = data.shape[0]
    target_label = np.array([label] * lenght)
    data["label"] = target_label
    return data


def link_data_frame(*data):
    """
    :param data: link DataFrame data
    :return: linked data
    """
    temp = pd.DataFrame()
    for i in data:
        temp = temp.append(i)
    return temp


def calculate_limit(*data, key: str or int):
    temp = pd.DataFrame()
    for value in data:
        temp = temp.append(value)
    max_value = temp[key].max() + 0.1
    min_value = temp[key].min() - 0.1
    return min_value, max_value


def delete_all_sub_str(string: str, sub: str, join_str=""):
    """
    :param string: long string, str
    :param sub: sub-string of string, str
    :param join_str: join string, str or None
    :return: after delete all sub string
    """
    string = string.split(sep=sub)
    string = np.array(string)
    string = join_str.join(np.delete(string, np.where(string == "")[0]))
    return string


def get_best_index(fname: str):
    """
    :param fname: file path and name
    :return: best index
    """
    file = open(fname, "r")
    string = file.readlines()
    string = "".join(string)
    string = string.replace("\n", "")
    temp = string.split(sep="accs")[0]
    temp = temp.split(sep=":")[1]
    aucs = delete_all_sub_str(string=temp, sub=" ", join_str=",")
    aucs = aucs.replace(",]", "]")
    aucs = np.array(eval(aucs))
    string = string.split(sep="accs")[1]
    avg_auc = string.split(sep="avg_aucs")[1]
    avg_auc = avg_auc.split(sep=" ")[0]
    avg_auc = float(avg_auc.split(sep=":")[1])
    abs_auc = np.abs(aucs - avg_auc)
    index = np.argmin(abs_auc)
    return index


def gather_color_code(*string):
    """
    :param string: colors, "blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "yellow", "cyan"
    :return: colors code, list
    """
    color_str = ["blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "yellow", "cyan"]
    palette = sns.color_palette()
    color_map = dict(zip(color_str, palette))
    colors = [color_map[color] for color in string]
    return colors

# import os
# import time
# import torch
# import numpy as np
# import pandas as pd
# import torch.nn as nn
# import seaborn as sns
# import pubchempy as pcp
# import scipy.sparse as sp
# from sklearn.metrics import roc_auc_score, average_precision_score
# import itertools as it
# from itertools import combinations
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
#
# import itertools
# from functools import partial
# import warnings
#
#
#
# def multiclass_auc_ovo(y_true, y_score):
#     n_classes = len(y_score[0])
#     class_pairs = list(combinations(range(n_classes), 2))
#     auc_scores = []
#     for pair in class_pairs:
#         y_true_pair, y_score_pair = [], []
#         for true_label, score in zip(y_true, y_score):
#             if true_label in pair:
#                 y_true_pair.append(1 if true_label == pair[0] else 0)
#                 y_score_pair.append(score[pair[0]] - score[pair[1]])
#         auc_scores.append(roc_auc_score(y_true_pair, y_score_pair))
#     return auc_scores
#
#
#
# def ap_score(true_data: torch.Tensor, predict_data: torch.Tensor):
#     """
#     area under the precision-recall curve for multiclass classification
#     :param true_data: true data, torch tensor 2D
#     :param predict_data: predict data, torch tensor 2D
#     :return: ap scores for each class
#     """
#     assert true_data.size() == predict_data.size(), "Size mismatch between true and predicted data!"
#     num_classes = true_data.size(1)
#     ap_scores = []
#     for i in range(num_classes):
#         class_true_data = true_data[:, i].detach().cpu().numpy()
#         class_predict_data = predict_data[:, i].detach().cpu().numpy()
#         ap = average_precision_score(y_true=class_true_data, y_score=class_predict_data)
#         ap_scores.append(ap)
#     return ap_scores
#
#
#
# def init_seeds(seed=0):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     #os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# def distribute_compute(lr_list,wd_list,scale_list,layer_size,sigma_list,beta_list,workers:int,id:int):
#     all_list = []
#     for lr,wd,sc,la,sg,bt in it.product(lr_list,wd_list,scale_list,layer_size,sigma_list,beta_list):
#         all_list.append([lr,wd,sc,la,sg,bt])
#     list = np.array_split(all_list,workers)
#     return list[id]
#
# def get_fingerprint(x):
#     """
#     :param x: DSE_drug cid
#     :return: fingerprint
#     """
#     drug_data = pcp.Compound.from_cid(x)
#     fingerprint = ""
#     for x in drug_data.fingerprint:
#         fingerprint += "{:04b}".format(int(x, 16))
#     finger = np.array([int(x) for x in fingerprint])
#     return finger
#
#
# def save_fingerprint(cid_list, last_cid, fpath):
#     """
#     :param cid_list: cid list
#     :param last_cid: last cid, -1 begining
#     :param fpath: save path
#     :return: file save to path
#     """
#     if last_cid > 0:
#         index = np.where(np.array(cid_list) == last_cid)[0].tolist()[0] + 1
#     else:
#         index = last_cid + 1
#     length = len(cid_list)
#     if index < length:
#         for i in range(index, length):
#             cid = cid_list[i]
#             fing = get_fingerprint(cid)
#             path = fpath + str(cid)
#             np.save(path, fing)
#             print(cid, "OK.")
#             time.sleep(1)
#     else:
#         print("All compound has finished!")
#     return None
#
#
# def read_fingerprint_cid(path: str):
#     """
#     :param path: file path
#     :return: fingerprint and cid
#     """
#     file_list = os.listdir(path)
#     fingerprint = np.array([], dtype=np.int)
#     cid = []
#     for name in file_list:
#         if name.endswith("npy"):
#             fpath = path + name
#             fing = np.load(fpath)
#             fingerprint = np.hstack((fingerprint, fing))
#             cid.append(int(name.split(".")[0]))
#     fingerprint = fingerprint.reshape((-1, 920))
#     return fingerprint, cid
#
#
# def common_data_index(data_for_index: np.ndarray, data_for_cmp: np.ndarray):
#     """
#     :param data_for_index: data for index, numpy array
#     :param data_for_cmp: data for compare, numpy array
#     :return: index of common data in data for index
#     """
#     index = np.array([np.where(x in data_for_cmp, 1, 0) for x in data_for_index])
#     index = np.where(index == 1)[0]
#     return index
#
#
#
#
# def to_coo_matrix(adj_mat: np.ndarray or sp.coo.coo_matrix):
#     """
#     :param adj_mat: adj matrix, numpy array
#     :return: sparse matrix, sp.coo.coo_matrix
#     """
#     if not sp.isspmatrix_coo(adj_mat):
#         adj_mat = sp.coo_matrix(adj_mat)
#     return adj_mat
#
#
# def mse_loss(true_data: torch.Tensor, predict_data:  torch.Tensor, masked: torch.Tensor):
#     """
#     :param true_data: true data
#     :param predict_data: predict data
#     :param masked: data mask
#     :return: mean square error
#     """
#     true_data = torch.mul(true_data, masked)
#     predict_data = torch.mul(predict_data, masked)
#     loss_fun = nn.MSELoss()
#     loss = loss_fun(predict_data, true_data)
#     return loss
#
#
# def cross_entropy_loss(true_data: torch.Tensor, predict_data:  torch.Tensor, masked: torch.Tensor):
#     """
#     :param true_data: true data
#     :param predict_data: predict data
#     :param masked: data mask
#     :return: cross entropy loss
#     """
#     masked = masked.to(torch.bool)
#     true_data = torch.masked_select(true_data, masked)
#     predict_data = torch.masked_select(predict_data, masked)
#     loss_fun = nn.BCELoss()
#     loss = loss_fun(predict_data, true_data)
#     return loss
#
#
# def mask(positive: sp.coo.coo_matrix, negative: sp.coo.coo_matrix, dtype=int):
#     """
#     :param positive: positive data
#     :param negative: negative data
#     :param dtype: return data type
#     :return: data mask
#     """
#     row = np.hstack((positive.row, negative.row))
#     col = np.hstack((positive.col, negative.col))
#     data = [1] * row.shape[0]
#     masked = sp.coo_matrix((data, (row, col)), shape=positive.shape).toarray().astype(dtype)
#     masked = torch.from_numpy(masked)
#     return masked
#
#
# def to_tensor(positive, identity=False):
#     """
#     :param positive: positive sample
#     :param identity: if add identity
#     :return: tensor
#     """
#     if identity:
#         data = positive + sp.identity(positive.shape[0])
#     else:
#         data = positive
#     data = torch.from_numpy(data.toarray()).float()
#     return data
#
#
# def evaluate_all(true_data: torch.Tensor, predict_data: torch.Tensor):
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     auc = roc_auc(true_data, predict_data)
#     ap = ap_score(true_data, predict_data)
#     f1, thresholds = f1_score_binary(true_data, predict_data)
#     acc = accuracy_binary(true_data, predict_data, thresholds)
#     precision = precision_binary(true_data, predict_data, thresholds)
#     recall = recall_binary(true_data, predict_data, thresholds)
#     mcc = mcc_binary(true_data, predict_data, thresholds)
#     return auc, ap, acc, f1, mcc
#
# def evaluate_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     auc = roc_auc(true_data, predict_data)
#     ap = ap_score(true_data, predict_data)
#     return auc, ap
#
#
#
# def rmse_score(true_data: torch.Tensor, predict_data: torch.Tensor):
#     """
#     Root Mean Square Error
#     :param true_data: true data, torch tensor 1D
#     :param predict_data: predicted data, torch tensor 1D
#     :return: rmse
#     """
#     assert true_data.shape == predict_data.shape, "Shape mismatch between true and predicted data!"
#     rmse = torch.sqrt(torch.mean((true_data - predict_data) ** 2))
#     return rmse
# def mae_score(true_data: torch.Tensor, predict_data: torch.Tensor):
#     """
#     Mean Absolute Error
#     :param true_data: true data, torch tensor 1D
#     :param predict_data: predicted data, torch tensor 1D
#     :return: mae
#     """
#     assert true_data.shape == predict_data.shape, "Shape mismatch between true and predicted data!"
#     mae = torch.mean(torch.abs(true_data - predict_data))
#     return mae
#
# def compute_aupr(true_data, predict_data_masked):
#     """
#     Compute Average Precision-Recall (AUPR) score.
#
#     Parameters:
#     - true_data: torch.Tensor, true labels (1D tensor)
#     - predict_data_masked: torch.Tensor, predicted scores (1D tensor)
#
#     Returns:
#     - aupr: float, AUPR score
#     """
#     true_data_np = true_data.cpu().numpy()
#     predict_data_masked_np = predict_data_masked.cpu().detach().numpy()
#
#     # Compute AUPR for each class
#     aupr_scores = []
#     for class_label in np.unique(true_data_np):
#         class_true = (true_data_np == class_label).astype(int)
#         class_predict = predict_data_masked_np if len(predict_data_masked_np.shape) == 1 else predict_data_masked_np[:,
#                                                                                               class_label]
#         aupr_class = average_precision_score(class_true, class_predict)
#         aupr_scores.append(aupr_class)
#
#     # Weighted average AUPR
#     class_counts = np.bincount(true_data_np.astype(int))
#     class_weights = class_counts / len(true_data_np)
#     weighted_aupr = np.average(aupr_scores, weights=class_weights)
#
#     return weighted_aupr
#
#
# # def ap_score(true_data: torch.Tensor, predict_data: torch.Tensor):
# #     """
# #     area under the precision-recall curve
# #     :param true_data: train data, torch tensor 1D
# #     :param predict_data: predict data, torch tensor 1D
# #     :return: ap
# #     """
# #     # assert torch.all(true_data.ge(0)) and torch.all(true_data.le(5)), "Out of range!"
# #     # area = ap_score_multiclass(y_true=true_data, y_score=predict_data)
# #     # area = ap_score_multiclass(y_true=true_data.detach().cpu().numpy(), y_score=predict_data.detach().cpu().numpy())
# #     # 计算每个类别的 AUPR 值
# #     aupr_scores = []
# #     for i in range(true_data.shape[0]):
# #         aupr_scores.append(average_precision_score(true_data[:, i], predict_data[:, i]))
# #
# #     def mean_auc(auc_scores):
# #         return "Mean AUPR: {:.4f}".format(np.mean(auc_scores))
#
#     # return area
#
#
#
# from sklearn.metrics import roc_auc_score
# import torch
# import numpy as np
#
# from sklearn.metrics import roc_auc_score
# import torch
# import numpy as np
#
# import numpy as np
# from sklearn.metrics import roc_auc_score
# import torch
#
# from sklearn.metrics import roc_auc_score
# import torch
# import numpy as np
#
# # def roc_auc(true_data, predict_data_masked):
# #     # 将 true_data 和 predict_data_masked 转换为 numpy 数组
# #     true_data_np = true_data.detach().cpu().numpy()
# #     predict_data_masked_np = predict_data_masked.detach().cpu().numpy()
# #
# #     # 打印以确保数据已正确转换
# #     # print("true_data_np:", true_data_np)
# #     # print("predict_data_masked_np:", predict_data_masked_np)
# #
# #     # 计算每个类别的 AUC 分数
# #     auc_scores = []
# #     unique_classes = np.unique(true_data_np)
# #     for cls in unique_classes:
# #         true_binary = (true_data_np == cls).astype(int)
# #         auc = roc_auc_score(true_binary, predict_data_masked_np)
# #         auc_scores.append(auc)
# #
# #     # 计算平均 AUC 分数
# #     mean_auc = np.mean(auc_scores)
# #     return mean_auc
#
# from sklearn.metrics import roc_auc_score
#
# def roc_auc(true_data, predict_data_masked):
#     # 将 true_data 和 predict_data_masked 转换为 numpy 数组
#     true_data_np = true_data.detach().cpu().numpy()
#     predict_data_masked_np = predict_data_masked.detach().cpu().numpy()
#
#     # 计算每个类别的 AUC 分数
#     auc_scores = []
#     unique_classes = np.unique(true_data_np)
#     for cls in unique_classes:
#         true_binary = (true_data_np == cls).astype(int)
#         # 提取预测数据的方式需要根据具体情况调整
#         if len(predict_data_masked_np.shape) == 2:
#             predict_data_cls = predict_data_masked_np[:, cls]  # 假设 predict_data_masked_np 的第二维表示类别
#         else:
#             predict_data_cls = predict_data_masked_np  # 如果 predict_data_masked_np 是一维的，直接使用
#
#         auc = roc_auc_score(true_binary, predict_data_cls)
#         auc_scores.append(auc)
#
#     # 计算平均 AUC 分数
#     mean_auc = np.mean(auc_scores)
#     return mean_auc
#
#
#
# # def roc_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
# #     """
# #     :param true_data: train data, torch tensor 1D
# #     :param predict_data: predict data, torch tensor 1D
# #     :return: AUC score
# #     """
# #     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(5)), "Out of range!"
# #     # aucs = multiclass_auc_ovo(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
# #     # aucs = multiclass_auc_ovo(true_data, predict_data)
# #     auc_scores = []
# #     for i in range(true_data.shape[0]):
# #         auc_scores.append(roc_auc_score(true_data[:, i], predict_data[:, i]))
# #
# #     # 输出每个类别的 AUC 值
# #     # for i, auc in enumerate(auc_scores):
# #     #     print("Class {} AUC: {:.4f}".format(i, auc))
# #
# #     # 输出平均 AUC 值
# #     def mean_auc(auc_scores):
# #         return "Mean AUC: {:.4f}".format(np.mean(auc_scores))
# #
# #     # return aucs
#
#
#
#
# def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
#     """
#     :param true_data: true data,torch tensor 1D
#     :param predict_data: predict data, torch tensor 1D
#     :return: max F1 score and threshold
#     """
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     with torch.no_grad():
#         thresholds = torch.unique(predict_data)
#     size = torch.tensor([thresholds.size()[0], true_data.size()[0]], dtype=torch.int32, device=true_data.device)
#     ones = torch.ones([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
#     zeros = torch.zeros([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
#     predict_value = torch.where(predict_data.view([1, -1]).ge(thresholds.view([-1, 1])), ones, zeros)
#     tpn = torch.sum(torch.where(predict_value.eq(true_data.view([1, -1])), ones, zeros), dim=1)
#     tp = torch.sum(torch.mul(predict_value, true_data.view([1, -1])), dim=1)
#     two = torch.tensor(2, dtype=torch.float32, device=true_data.device)
#     n = torch.tensor(size[1].item(), dtype=torch.float32, device=true_data.device)
#     scores = torch.div(torch.mul(two, tp), torch.add(n, torch.sub(torch.mul(two, tp), tpn)))
#     max_f1_score = torch.max(scores)
#     threshold = thresholds[torch.argmax(scores)]
#     return max_f1_score, threshold
#
#
# def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
#     """
#     :param true_data: true data, 1D torch Tensor
#     :param predict_data: predict data , 1D torch Tensor
#     :param threshold: threshold, float or torch Tensor
#     :return: acc
#     """
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     n = true_data.size()[0]
#     ones = torch.ones(n, dtype=torch.float32, device=true_data.device)
#     zeros = torch.zeros(n, dtype=torch.float32, device=true_data.device)
#     predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
#     tpn = torch.sum(torch.where(predict_value.eq(true_data), ones, zeros))
#     score = torch.div(tpn, n)
#     return score
#
#
# def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
#     """
#     :param true_data: true data, 1D torch Tensor
#     :param predict_data: predict data , 1D torch Tensor
#     :param threshold: threshold, float or torch Tensor
#     :return: precision
#     """
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
#     zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
#     predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
#     tp = torch.sum(torch.mul(true_data, predict_value))
#     true_neg = torch.sub(ones, true_data)
#     tf = torch.sum(torch.mul(true_neg, predict_value))
#     score = torch.div(tp, torch.add(tp, tf))
#     return score
#
#
# def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
#     """
#     :param true_data: true data, 1D torch Tensor
#     :param predict_data: predict data , 1D torch Tensor
#     :param threshold: threshold, float or torch Tensor
#     :return: precision
#     """
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
#     zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
#     predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
#     tp = torch.sum(torch.mul(true_data, predict_value))
#     predict_neg = torch.sub(ones, predict_value)
#     fn = torch.sum(torch.mul(predict_neg, true_data))
#     score = torch.div(tp, torch.add(tp, fn))
#     return score
#
#
# def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
#     """
#     :param true_data: true data, 1D torch Tensor
#     :param predict_data: predict data , 1D torch Tensor
#     :param threshold: threshold, float or torch Tensor
#     :return: precision
#     """
#     assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
#     ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
#     zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
#     predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
#     predict_neg = torch.sub(ones, predict_value)
#     true_neg = torch.sub(ones, true_data)
#     tp = torch.sum(torch.mul(true_data, predict_value))
#     tn = torch.sum(torch.mul(true_neg, predict_neg))
#     fp = torch.sum(torch.mul(true_neg, predict_value))
#     fn = torch.sum(torch.mul(true_data, predict_neg))
#     delta = torch.tensor(0.00001, dtype=torch.float32, device=true_data.device)
#     score = torch.div((tp * tn - fp * fn), torch.add(delta, torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
#     return score
#
# def torch_corr(tensor: torch.Tensor, dim=0):
#     """
#     :param tensor: an 2D torch tensor
#     :param dim:
#         0 : Calculate row correlation
#         1 : Calculate col correlation
#     :return: correlation coefficient
#     """
#     mean = torch.mean(tensor, dim=1-dim)
#     if dim:
#         tensor_mean = torch.sub(tensor, mean)
#         tensor_cov = torch.mm(torch.t(tensor_mean), tensor_mean)
#     else:
#         mean = mean.view([mean.size()[0], -1])
#         tensor_mean = torch.sub(tensor, mean)
#         tensor_cov = torch.mm(tensor_mean, torch.t(tensor_mean))
#     diag = torch.diag(tensor_cov)
#     diag = torch.sqrt(diag)
#     diag = torch.mm(diag.view([-1, 1]), diag.view([1, -1]))
#     tensor_corr = torch.div(tensor_cov, diag)
#     return tensor_corr
#
#
# def torch_corr_x_y(tensor1: torch.Tensor, tensor2: torch.Tensor):
#     """
#     :param tensor1: a matrix, torch Tensor
#     :param tensor2: a matrix, torch Tensor
#     :return: corr(tensor1, tensor2)
#     """
#     assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
#     tensor2 = torch.t(tensor2)
#     mean1 = torch.mean(tensor1, dim=1).view([-1, 1])
#     mean2 = torch.mean(tensor2, dim=0).view([1, -1])
#     lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))
#     lxx = torch.diag(torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1))))
#     lyy = torch.diag(torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2)))
#     std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
#     corr_x_y = torch.div(lxy, std_x_y)
#     return corr_x_y
#
#
# def torch_euclidean_dist(tensor: torch.Tensor, dim=0):
#     """
#     :param tensor: a 2D torch tensor
#     :param dim:
#         0 : represent row
#         1 : represent col
#     :return: return euclidean distance
#     """
#     if dim:
#         tensor_mul_tensor = torch.mm(torch.t(tensor), tensor)
#     else:
#         tensor_mul_tensor = torch.mm(tensor, torch.t(tensor))
#     diag = torch.diag(tensor_mul_tensor)
#     n_diag = diag.size()[0]
#     tensor_diag = diag.repeat([n_diag, 1])
#     diag = diag.view([n_diag, -1])
#     dist = torch.sub(torch.add(tensor_diag, diag), torch.mul(tensor_mul_tensor, 2))
#     dist = torch.sqrt(dist)
#     return dist
#
#
# def torch_dist(tensor: torch.Tensor, p=0 or int):
#     """
#     :param tensor: an 2D tensor
#     :param p: pow
#     :return: distance between rows
#     """
#     size = tensor.size()
#     tensor_flatten = torch.flatten(tensor)
#     tensor_mat = tensor.repeat([1, 1, size[0]])
#     tensor_flatten = tensor_flatten.repeat([1, size[0], 1])
#     tensor_sub = torch.sub(tensor_mat, tensor_flatten)
#     tensor_sub = tensor_sub.view([size[0], size[0], size[1]])
#     tensor_sub = torch.abs(tensor_sub)
#     if p == 0:
#         tensor_sub = torch.pow(tensor_sub, p)
#         dist = torch.sum(tensor_sub, dim=2)
#         diag = torch.diag(dist)
#         dist = torch.sub(dist, torch.diag(diag))
#     elif p == 1:
#         dist = torch.sum(tensor_sub, dim=2)
#     else:
#         tensor_sub = torch.pow(tensor_sub, p)
#         dist = torch.sum(tensor_sub, dim=2)
#         dist = torch.pow(dist, 1/p)
#     return dist
#
#
# def torch_z_normalized(tensor: torch.Tensor, dim=0):
#     """
#     :param tensor: an 2D torch tensor
#     :param dim:
#         0 : normalize row data
#         1 : normalize col data
#     :return: Gaussian normalized tensor
#     """
#     mean = torch.mean(tensor, dim=1-dim)
#     std = torch.std(tensor, dim=1-dim)
#     if dim:
#         tensor_sub_mean = torch.sub(tensor, mean)
#         tensor_normalized = torch.div(tensor_sub_mean, std)
#     else:
#         size = mean.size()[0]
#         tensor_sub_mean = torch.sub(tensor, mean.view([size, -1]))
#         tensor_normalized = torch.div(tensor_sub_mean, std.view([size, -1]))
#     return tensor_normalized
#
#
# def exp_similarity(tensor: torch.Tensor, sigma: torch.Tensor, normalize=True):
#     """
#     :param tensor: an torch tensor
#     :param sigma: scale parameter
#     :param normalize: normalize or not
#     :return: exponential similarity
#     """
#     if normalize:
#         tensor = torch_z_normalized(tensor, dim=1)
#     tensor_dist = torch_euclidean_dist(tensor, dim=0)
#     exp_dist = torch.exp(-tensor_dist/(2*torch.pow(sigma, 2)))
#     return exp_dist
#
#
# def jaccard_coef(tensor: torch.Tensor):
#     """
#     :param tensor: an torch tensor, 2D
#     :return: jaccard coefficient
#     """
#     assert torch.all(tensor.le(1)) and torch.all(tensor.ge(0)), "Value must be 0 or 1"
#     size = tensor.size()
#     tensor_3d = torch.flatten(tensor).repeat([size[0]]).view([size[0], size[0], size[1]])
#     ones = torch.ones(tensor_3d.size(), dtype=torch.float32, device=tensor.device)
#     zeros = torch.zeros(tensor_3d.size(), dtype=torch.float32, device=tensor.device)
#     tensor_3d = torch.add(tensor_3d, tensor.view([size[0], 1, size[1]]))
#     intersection = torch.where(tensor_3d.eq(2), ones, zeros)
#     union = torch.where(tensor_3d.eq(2), ones, tensor_3d)
#     intersection = torch.sum(intersection, dim=2)
#     union = torch.sum(union, dim=2)
#     union = torch.where(union.eq(0), torch.add(union, 0.1), union)
#     eye = torch.eye(union.size()[0], dtype=torch.float32, device=tensor.device)
#     jaccard = torch.div(intersection, union)
#     jaccard = torch.where(jaccard.eq(0), eye, jaccard)
#     return jaccard
#
#
# def full_kernel(exp_dist: torch.Tensor):
#     """
#     :param exp_dist: exponential similarity
#     :return: full kernel
#     """
#     n = exp_dist.shape[0]
#     ones = torch.ones(n, n, dtype=torch.float, device=exp_dist.device)
#     one = torch.diag(torch.diag(ones))
#     mask_diag = torch.mul(torch.sub(ones, one), exp_dist)
#     mask_diag_sum = torch.sum(mask_diag, dim=1).view([n, -1])
#     mask_diag = torch.div(mask_diag, 2*mask_diag_sum)
#     mask_diag = torch.add(mask_diag, 0.5*one)
#     return mask_diag
#
#
# def sparse_kernel(exp_dist: torch.Tensor, k: int):
#     """
#     :param exp_dist: exponential similarity
#     :param k: knn k
#     :return: sparse kernel
#     """
#     n = exp_dist.shape[0]
#     maxk = torch.topk(exp_dist, k, dim=1)
#     mink_1 = torch.topk(exp_dist, n-k, dim=1, largest=False)
#     index = torch.arange(n, device=exp_dist.device).view([n, -1])
#     exp_dist[index, mink_1.indices] = 0
#     knn_sum = torch.sum(maxk.values, dim=1).view([n, -1])
#     exp_dist = torch.div(exp_dist, knn_sum)
#     return exp_dist
#
#
# # def scale_sigmoid(tensor: torch.Tensor, alpha: int or float):
# #     """
# #     :param tensor: a torch tensor, range is [-1, 1]
# #     :param alpha: an scale parameter to sigmod
# #     :return: mapping tensor to [0, 1]
# #     """
# #     alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
# #     output = torch.relu(torch.mul(alpha, tensor))
# #     return output
# # def scale_sigmoid(tensor: torch.Tensor, alpha: int or float):
# #     """
# #     :param tensor: a torch tensor, range is [-1, 1]
# #     :param alpha: a scale parameter to sigmoid
# #     :return: mapping tensor to [0, 5]
# #     """
# #     alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
# #     output = torch.sigmoid(torch.mul(alpha, tensor))
# #     scaled_output = output * 5  # Scale the output to the range [0, 5]
# #     return scaled_output
# def scale_sigmoid(tensor: torch.Tensor, alpha: int or float):
#     """
#     :param tensor: a torch tensor, range is [-1, 1]
#     :param alpha: an scale parameter to sigmod
#     :return: mapping tensor to [0, 1]
#     """
#     alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
#     output = torch.sigmoid(torch.mul(alpha, tensor))
#     return output
# def np_delete_value(arr: np.ndarray, obj: np.ndarray):
#     """
#     :param arr: 1-D vector, narray
#     :param obj: 1-D vector, value that would be removed , narray
#     :return: after removed vector
#     """
#     index = [np.where(x == arr)[0][0] for x in obj if x in arr]
#     arr = np.delete(arr, index)
#     return arr
#
#
# # def translate_result(tensor: torch.Tensor or np.ndarray):
# #     """
# #     :param tensor: torch tensor or np.ndarray
# #     :return: pd.DataFrame
# #     """
# #     if isinstance(tensor, torch.Tensor):
# #         tensor = tensor.detach().cpu().numpy()
# #     arr = tensor.reshape((1, -1))
# #     arr = pd.DataFrame(arr)
# #     return arr
# def translate_result(tensor: torch.Tensor or np.ndarray):
#     """
#     :param tensor: torch tensor or np.ndarray
#     :return: pd.DataFrame
#     """
#     # 如果输入是 PyTorch 张量，则将其转换为 NumPy 数组
#     if isinstance(tensor, torch.Tensor):
#         tensor = tensor.detach().cpu().numpy()
#
#     # 将数组重塑为形状 (1, -1)
#     arr = tensor.reshape((1, -1))
#
#     # 将重塑后的数组转换为 Pandas DataFrame
#     arr = pd.DataFrame(arr)
#
#     return arr
#
#
# def calculate_train_test_index(response: np.ndarray, pos_train_index: np.ndarray, pos_test_index: np.ndarray):
#     """
#     :param response: response vector, np.ndarray
#     :param pos_train_index: positive train index in response
#     :param pos_test_index: positive test index in reaponse
#     :return: train index and test index
#     """
#     neg_response_index = np.where(response == 0)[0]
#     neg_test_index = np.random.choice(neg_response_index, pos_test_index.shape[0])
#     neg_train_index = np_delete_value(neg_response_index, neg_test_index)
#     test_index = np.hstack((pos_test_index, neg_test_index))
#     train_index = np.hstack((pos_train_index, neg_train_index))
#     return train_index, test_index
#
#
# def dir_path(k=1):
#     """
#     :param k: 当前路径后退级数
#     :return: 后退k级后的目录
#     """
#     fpath = os.path.realpath(__file__)
#     dir_name = os.path.dirname(fpath)
#     dir_name = dir_name.replace("\\", "/")
#     p = len(dir_name) - 1
#     while p > 0:
#         if dir_name[p] == "/":
#             k -= 1
#             if k == 0:
#                 break
#         p -= 1
#     p += 1
#     dir_name = dir_name[0: p]
#     return dir_name
#
#
# def extract_row_data(data: pd.DataFrame, row: int):
#     """
#     :param data: DataFrame
#     :param row: row index, int
#     :return: not nan data
#     """
#     data = np.array(data, dtype=np.float32)
#     target = data[row, :]
#     target = target[np.where(~np.isnan(target))[0]]
#     return target
#
#
# def transfer_data(data: pd.DataFrame, label: str):
#     lenght = data.shape[0]
#     target_label = np.array([label]*lenght)
#     data["label"] = target_label
#     return data
#
#
# def link_data_frame(*data):
#     """
#     :param data: link DataFrame data
#     :return: linked data
#     """
#     temp = pd.DataFrame()
#     for i in data:
#         temp = temp.append(i)
#     return temp
#
#
# def calculate_limit(*data, key: str or int):
#     temp = pd.DataFrame()
#     for value in data:
#         temp = temp.append(value)
#     max_value = temp[key].max() + 0.1
#     min_value = temp[key].min() - 0.1
#     return min_value, max_value
#
#
# def delete_all_sub_str(string: str, sub: str, join_str=""):
#     """
#     :param string: long string, str
#     :param sub: sub-string of string, str
#     :param join_str: join string, str or None
#     :return: after delete all sub string
#     """
#     string = string.split(sep=sub)
#     string = np.array(string)
#     string = join_str.join(np.delete(string, np.where(string == "")[0]))
#     return string
#
#
# def get_best_index(fname: str):
#     """
#     :param fname: file path and name
#     :return: best index
#     """
#     file = open(fname, "r")
#     string = file.readlines()
#     string = "".join(string)
#     string = string.replace("\n", "")
#     temp = string.split(sep="accs")[0]
#     temp = temp.split(sep=":")[1]
#     aucs = delete_all_sub_str(string=temp, sub=" ", join_str=",")
#     aucs = aucs.replace(",]", "]")
#     aucs = np.array(eval(aucs))
#     string = string.split(sep="accs")[1]
#     avg_auc = string.split(sep="avg_aucs")[1]
#     avg_auc = avg_auc.split(sep=" ")[0]
#     avg_auc = float(avg_auc.split(sep=":")[1])
#     abs_auc = np.abs(aucs - avg_auc)
#     index = np.argmin(abs_auc)
#     return index
#
#
# def gather_color_code(*string):
#     """
#     :param string: colors, "blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "yellow", "cyan"
#     :return: colors code, list
#     """
#     color_str = ["blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "yellow", "cyan"]
#     palette = sns.color_palette()
#     color_map = dict(zip(color_str, palette))
#     colors = [color_map[color] for color in string]
#     return colors
#
# import numpy as np
# from scipy.sparse import issparse
# from sklearn.preprocessing import normalize
# from sklearn.utils.extmath import safe_sparse_dot
# from sklearn.metrics.pairwise import check_pairwise_arrays
#
# def cosine_similarity(X, Y=None, dense_output=True):
#     """Compute cosine similarity between samples in X and Y.
#
#     Cosine similarity, or the cosine kernel, computes similarity as the
#     normalized dot product of X and Y:
#
#         K(X, Y) = <X, Y> / (||X||*||Y||)
#
#     On L2-normalized data, this function is equivalent to linear_kernel.
#
#     Parameters
#     ----------
#     X : ndarray or sparse array, shape: (n_samples_X, n_features)
#         Input data.
#
#     Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
#         Input data. If ``None``, the output will be the pairwise
#         similarities between all samples in ``X``.
#
#     dense_output : boolean (optional), default True
#         Whether to return dense output even when the input is sparse. If
#         ``False``, the output is sparse if both input arrays are sparse.
#
#         .. versionadded:: 0.17
#            parameter ``dense_output`` for dense output.
#
#     Returns
#     -------
#     kernel matrix : array
#         An array with shape (n_samples_X, n_samples_Y).
#     """
#     X, Y = check_pairwise_arrays(X, Y)
#
#     X_normalized = normalize(X, copy=True)
#     if X is Y:
#         Y_normalized = X_normalized
#     else:
#         Y_normalized = normalize(Y, copy=True)
#
#     K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)
#
#     return K
#
#
#
