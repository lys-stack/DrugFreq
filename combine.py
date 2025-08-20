# combine.py

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import scipy.sparse as sp

def kernel_normalized(k):#对核矩阵进行标准化
    n = len(k)
    k = np.abs(k)
    index_nozeros = k.nonzero()#获取非零元素的索引
    min_value = min(k[index_nozeros])
    k[k == 0] = min_value#将所有值为0的元素替换为最小值
    diag = np.resize(np.diagonal(k), [n, 1]) ** 0.5#获取对角线元素的平方根
    k_nor = k / (np.dot(diag, diag.T))#对角线元素的平方根作为每个元素的标准差，除以对角线元素的平方根
    return k_nor#返回标准化后的核矩阵

# def get_CKA_Wi(P, q, G, h, A, b):#使用二次规划求解器计算特定权重
#     """
#     :param P:二次项系数矩阵
#     :param q: 一次性项系数向量
#     :param G: 不等式约束矩阵
#     :param h: 不等式约束的右侧值
#     :param A: 等式约束矩阵
#     :param b: 等式约束的右侧值
#     :return:
#     """
#     P = matrix(P)
#     G = matrix(G)
#     h = matrix(h)
#     A = matrix(A)
#     sol = solvers.qp(P, q, G, h, A, b)#求解线性规划问题
#     return np.array(sol['x']).flatten()#将求解结果中的x转换为numpy数组展平并返回

def get_CKA_Wi(P, q, G, h, A, b):
    # 如果是 numpy 数组则转换；否则保留为 cvxopt.matrix
    P = matrix(P) if not isinstance(P, matrix) else P
    q = matrix(q) if not isinstance(q, matrix) else q
    G = matrix(G) if not isinstance(G, matrix) else G
    h = matrix(h) if not isinstance(h, matrix) else h
    A = matrix(A) if not isinstance(A, matrix) else A
    b = matrix(b) if not isinstance(b, matrix) else b

    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol['x']).flatten()


def get_trace(a, b):#计算两个矩阵乘积的迹
    return np.trace(np.dot(a.T, b))

def get_P(train_x_k_list):#据训练相似度矩阵列表计算矩阵 P
    l = len(train_x_k_list)
    n = len(train_x_k_list[0])
    In = np.identity(n)#创建n阶单位矩阵
    ln = np.ones([n, 1])#创建n*1阶全1列向量
    Un = In - (1 / n) * np.dot(ln, ln.T)#计算归一化矩阵的中心化矩阵
    P = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            P[i, j] = get_trace(np.dot(np.dot(Un, train_x_k_list[i]), Un), np.dot(np.dot(Un, train_x_k_list[j]), Un.T))
    return P

'''用于计算多核学习（MKL）中的目标函数部分，它帮助量化每个相似度矩阵与理想核矩阵之间的差异'''
def get_q(train_x_k_list, ideal_kernel):
    l = len(train_x_k_list)
    n = len(train_x_k_list[0])
    In = np.identity(n)
    ln = np.ones([n, 1])
    Un = In - (1 / n) * np.dot(ln, ln.T) #计算归一化矩阵的中心化矩阵
    Ki = ideal_kernel #获取理想核矩阵
    a = np.zeros([l, 1])
    for i in range(l):
        a[i, 0] = get_trace(np.dot(np.dot(Un, train_x_k_list[i]), Un), Ki) #计算每个相似度矩阵与理想核矩阵乘积的迹，表示两个矩阵的对齐程度
    return a

'''归一化操作'''
def get_Med(array):
    l = len(array)#获取矩阵的维度大小
    re = np.zeros([l, l])#初始化一个l*l的矩阵
    for i in range(l):
        for j in range(l):
            re[i][j] = array[i][j] / ((array[i][i] ** 0.5) * (array[j][j] ** 0.5))
    return re

'''对输入的矩阵进行归一化处理'''
def get_mu(array):
    s = np.sum(array ** 2)#计算 array 的元素平方和
    return array / np.sqrt(s)#用矩阵除以它自身二次范数（三角归一化处理）

'''余弦相似度计算'''
def get_WW(t1, t2):
    fenzi = np.trace(np.dot(t1, t2))#计算两个矩阵乘积的迹
    fenmu = np.sqrt(np.trace(np.dot(t1, t1)) * np.trace(np.dot(t2, t2)))#算矩阵 t1 、t2 的点积自身，相当于计算其 Frobenius 范数的平方。np.trace() 计算这个结果的迹。并开根号
    return round(fenzi / fenmu, 4)#返回两个矩阵的余弦相似度

def get_n_weight(k_train_list, ideal_kernel, lambd=0.8):
    n = len(k_train_list)
    Wij = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            Wij[i][j] = get_WW(k_train_list[i], k_train_list[j])#计算两个相似度矩阵的余弦相似度
            Wij[j][i] = Wij[i][j]
    D = Wij.sum(axis=0)#计算Wij矩阵的每一列的和，得到度矩阵
    Dii = np.diag(D)#创建一个对角矩阵 Dii，其对角线元素为 D
    L = Dii - Wij#计算拉普拉斯矩阵
    L = np.abs(L)
    M = get_Med(get_P(k_train_list))
    P = M + lambd * L
    a = get_mu(get_q(k_train_list, ideal_kernel))
    q = -1 * a
    G = -1 * np.identity(n)
    h = np.zeros([n, 1])
    A = np.ones([1, n])
    b = matrix([[1.0]])  # 确保 'b' 是一个 (1, 1) 的 cvxopt 矩阵

    return get_CKA_Wi(P, q, G, h, A, b)

def read_csv_with_labels(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    values = df.apply(pd.to_numeric, errors='coerce').fillna(0).values
    return df, values



# def perform_feature_fusion(train_indices, similarity_matrices, ideal_kernel_values, lambd=0.8, matrix_type='DSE_drug'):
#     """
#     Perform feature fusion (CKA-MKL) for a given set of similarity matrices based on the training indices.
#
#     Parameters:
#     - train_indices (list of int): Indices of the training samples (药物或副作用索引，已减去标签偏移)。
#     - similarity_matrices (list of numpy arrays): List of similarity matrices to be fused。
#     - ideal_kernel_values (numpy array): Ideal kernel matrix for fusion。
#     - lambd (float): Regularization parameter。
#     - matrix_type (str): "DSE_drug" 表示药物相似度矩阵，"side" 表示副作用相似度矩阵。
#
#     Returns:
#     - fused_sim (numpy array): Fused similarity matrix。
#     """
#     print(f"Performing feature fusion for {len(train_indices)} indices")
#     print(f"Ideal kernel shape: {ideal_kernel_values.shape}")
#     print(f"Train indices: {train_indices[:5]} ...")  # 打印前5个索引
#
#     # 检查并过滤掉超过矩阵大小的非法索引
#     if matrix_type == 'DSE_drug':
#         max_size = ideal_kernel_values.shape[0]  # 药物维度是750
#     else:
#         max_size = ideal_kernel_values.shape[1]  # 副作用维度是994
#
#     valid_train_indices = [i for i in train_indices if i < max_size]
#
#     if len(valid_train_indices) != len(train_indices):
#         print(f"Warning: {len(train_indices) - len(valid_train_indices)} indices were out of bounds and removed.")
#
#     # 使用合法的训练集索引计算理想核
#     train_ideal_kernel = ideal_kernel_values[np.ix_(valid_train_indices, valid_train_indices)]
#     print(f"train_ideal_kernel shape: {train_ideal_kernel.shape}")  # 应为合法大小
#     train_ideal_kernel = np.dot(train_ideal_kernel, train_ideal_kernel.T)
#     train_ideal_kernel = kernel_normalized(train_ideal_kernel)
#     print(f"Normalized train_ideal_kernel shape: {train_ideal_kernel.shape}")
#
#     # 提取合法的训练集对应的相似度矩阵
#     train_similarity = [m[np.ix_(valid_train_indices, valid_train_indices)] for m in similarity_matrices]
#     print(f"Number of similarity matrices: {len(train_similarity)}")
#     for idx, sim in enumerate(train_similarity):
#         print(f"Similarity matrix {idx} shape: {sim.shape}")
#
#     # 计算核权重
#     weights = get_n_weight(train_similarity, train_ideal_kernel, lambd=lambd)
#     print(f"Kernel weights: {weights}")
#
#     # 合并相似度矩阵
#     fused_sim = np.zeros_like(similarity_matrices[0])
#     for i in range(len(similarity_matrices)):
#         fused_sim += weights[i] * similarity_matrices[i]
#     print(f"Fused similarity matrix shape: {fused_sim.shape}")
#
#     return fused_sim
def perform_feature_fusion(train_indices, similarity_matrices, ideal_kernel_values, lambd=0.8, matrix_type='DSE_drug'):
    """
    Perform feature fusion (CKA-MKL) for a given set of similarity matrices based on the training indices.

    Parameters:
    - train_indices (list of int): Indices of the training samples (药物或副作用索引，已减去标签偏移)。
    - similarity_matrices (list of numpy arrays): List of similarity matrices to be fused。
    - ideal_kernel_values (numpy array): Ideal kernel matrix for fusion。
    - lambd (float): Regularization parameter。
    - matrix_type (str): "DSE_drug" 表示药物相似度矩阵，"side" 表示副作用相似度矩阵。

    Returns:
    - fused_sim (numpy array): Fused similarity matrix with the same shape as the input matrices。
    """
    print(f"Performing feature fusion for {len(train_indices)} indices")
    print(f"Ideal kernel shape: {ideal_kernel_values.shape}")
    print(f"Train indices: {train_indices[:5]} ...")  # 打印前5个索引

    # 检查并过滤掉超过矩阵大小的非法索引
    if matrix_type == 'DSE_drug':
        max_size = ideal_kernel_values.shape[0]  # 药物维度是750
    else:
        max_size = ideal_kernel_values.shape[1]  # 副作用维度是994

    valid_train_indices = [i for i in train_indices if i < max_size]#过滤掉超过矩阵大小的非法索引

    if len(valid_train_indices) != len(train_indices):
        print(f"Warning: {len(train_indices) - len(valid_train_indices)} indices were out of bounds and removed.")

    # 使用合法的训练集索引提取理想核矩阵的训练集部分
    train_ideal_kernel = ideal_kernel_values[np.ix_(valid_train_indices, valid_train_indices)]#从理想核矩阵 ideal_kernel_values 中提取出一个只包含valid_train_indices这些样本的子矩阵 train_ideal_kernel
    print(f"train_ideal_kernel shape: {train_ideal_kernel.shape}")  # 应为合法大小
    train_ideal_kernel = np.dot(train_ideal_kernel, train_ideal_kernel.T)#计算理想自协方差矩阵，反映了药物或副作用之间的相似度关系的自协方差
    train_ideal_kernel = kernel_normalized(train_ideal_kernel)
    print(f"Normalized train_ideal_kernel shape: {train_ideal_kernel.shape}")

    # 提取合法的训练集对应的相似度矩阵
    train_similarity = [m[np.ix_(valid_train_indices, valid_train_indices)] for m in similarity_matrices]#提取出一个只包含valid_train_indices这些样本的子矩阵，并将其存储在列表 train_similarity 中
    print(f"Number of similarity matrices: {len(train_similarity)}")
    for idx, sim in enumerate(train_similarity):#enumerate(train_similarity) 用于生成一个索引-值对的迭代器
        print(f"Similarity matrix {idx} shape: {sim.shape}")

    # 计算核权重，只使用训练集信息
    weights = get_n_weight(train_similarity, train_ideal_kernel, lambd=lambd)
    print(f"Kernel weights: {weights}")

    # 使用核权重合并整个相似度矩阵（训练集和测试集）
    fused_sim = np.zeros_like(similarity_matrices[0])
    for i in range(len(similarity_matrices)):
        # 对每个相似度矩阵，使用计算的权重进行加权融合
        fused_sim += weights[i] * similarity_matrices[i]

    # 确保融合后的矩阵形状与原始矩阵一致
    print(f"Fused similarity matrix shape: {fused_sim.shape}")

    return fused_sim
