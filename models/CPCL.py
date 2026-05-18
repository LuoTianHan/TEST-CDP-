"""
CP-CL模块: 因果感知对比学习 (Causal Perception Contrastive Learning)
实现变量依赖图构建、物理感知样本对构建、判别式嵌入空间优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.mlp import GraphAttentionLayer, GroupAttentionAggregation

# 延迟导入可选依赖（仅在运行时用到）
_imported_optional = False
_euclidean = None
_fastdtw = None
_mutual_info_regression = None
_grangercausalitytests = None

def _ensure_optional_imports():
    """确保可选依赖已导入"""
    global _imported_optional, _euclidean, _fastdtw, _mutual_info_regression, _grangercausalitytests
    if not _imported_optional:
        try:
            from scipy.spatial.distance import euclidean as _euclidean_impl
            _euclidean = _euclidean_impl
        except ImportError:
            pass
        try:
            from fastdtw import fastdtw as _fastdtw_impl
            _fastdtw = _fastdtw_impl
        except ImportError:
            pass
        try:
            from sklearn.feature_selection import mutual_info_regression as _mi_impl
            _mutual_info_regression = _mi_impl
        except ImportError:
            pass
        try:
            from statsmodels.tsa.stattools import grangercausalitytests as _gc_impl
            _grangercausalitytests = _gc_impl
        except ImportError:
            pass
        _imported_optional = True


class VariableDependencyGraph:
    """
    变量依赖图构建器
    基于多维统计度量（Pearson相关、DTW、互信息、Granger因果）构建复合依赖图
    """
    def __init__(self, tau_pos=0.5, tau_neg=0.1, th_dtw=10.0, th_mi=0.1, max_lag=5, significance=0.05):
        """
        :param tau_pos: 正样本Pearson相关系数阈值
        :param tau_neg: 负样本Pearson相关系数阈值
        :param th_dtw: DTW距离阈值
        :param th_mi: 互信息阈值
        :param max_lag: Granger因果检验的最大滞后阶数
        :param significance: Granger因果检验显著性水平
        """
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.th_dtw = th_dtw
        self.th_mi = th_mi
        self.max_lag = max_lag
        self.significance = significance
    
    def pearson_correlation(self, x, y):
        """计算Pearson相关系数"""
        x_mean, y_mean = x.mean(), y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = torch.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
        return numerator / (denominator + 1e-8)
    
    def dtw_distance(self, x, y):
        """计算DTW动态时间弯曲距离"""
        _ensure_optional_imports()
        x_np = x.cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        if _fastdtw is not None and _euclidean is not None:
            distance, _ = _fastdtw(x_np, y_np, dist=_euclidean)
        else:
            # 降级为欧氏距离
            distance = np.sqrt(np.sum((x_np - y_np) ** 2))
        return distance
    
    def mutual_information(self, x, y):
        """计算互信息"""
        _ensure_optional_imports()
        x_np = x.cpu().numpy().reshape(-1, 1)
        y_np = y.cpu().numpy().flatten()
        try:
            if _mutual_info_regression is not None:
                mi = _mutual_info_regression(x_np, y_np, random_state=42)[0]
            else:
                # 降级为相关系数近似
                mi = abs(np.corrcoef(x_np.flatten(), y_np)[0, 1])
        except:
            mi = 0.0
        return mi
    
    def granger_causality(self, x, y):
        """
        计算Granger因果系数
        返回: (最大Granger系数, 最优滞后阶数)
        """
        _ensure_optional_imports()
        try:
            x_np = x.cpu().numpy().flatten()
            y_np = y.cpu().numpy().flatten()
            
            # 构建二维数组 [y, x] 用于Granger检验
            data = np.column_stack([y_np, x_np])
            
            # Granger因果检验
            if _grangercausalitytests is not None:
                gc_res = _grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
                
                # 提取各滞后阶数的p值和F统计量
                max_gc = -1
                best_lag = 0
                for lag in range(1, self.max_lag + 1):
                    f_stat = gc_res[lag][0]['ssr_ftest'][0]
                    p_value = gc_res[lag][0]['ssr_ftest'][1]
                    if p_value < self.significance and f_stat > max_gc:
                        max_gc = f_stat
                        best_lag = lag
            else:
                # 降级：使用互相关作为Granger因果的近似
                best_lag = 0
                max_corr = -1
                for lag in range(1, min(self.max_lag + 1, len(x_np) // 2)):
                    if lag >= len(x_np) or lag >= len(y_np):
                        break
                    corr = abs(np.corrcoef(x_np[lag:], y_np[:-lag])[0, 1])
                    if corr > max_corr:
                        max_corr = corr
                        best_lag = lag
                max_gc = max_corr if max_corr > 0 else 0
            
            # 归一化Granger系数 (Cov(x_t,i, x_{t-tau,j}) / (sigma_i * sigma_j))
            if max_gc > 0 and best_lag > 0 and best_lag < len(x_np):
                cov = np.cov(x_np[best_lag:], y_np[:-best_lag])[0, 1]
                std_i, std_j = np.std(x_np), np.std(y_np)
                gc_coeff = abs(cov) / (std_i * std_j + 1e-8)
            else:
                gc_coeff = 0.0
                best_lag = 0
            
            return gc_coeff, best_lag
        except Exception as e:
            return 0.0, 0
    
    def build_dependency_graph(self, time_series):
        """
        构建复合依赖图
        :param time_series: [T, C] 多变量时间序列
        :return: 依赖矩阵 A [C, C], Granger滞后矩阵 lag_matrix [C, C]
        """
        T, C = time_series.shape
        A = torch.zeros(C, C)
        lag_matrix = torch.zeros(C, C, dtype=torch.long)
        
        for i in range(C):
            for j in range(C):
                if i == j:
                    A[i, j] = 1.0
                    continue
                
                x_i = time_series[:, i]
                x_j = time_series[:, j]
                
                # Pearson相关系数
                rho = self.pearson_correlation(x_i, x_j)
                
                # DTW距离
                dtw = self.dtw_distance(x_i, x_j)
                
                # 互信息
                mi = self.mutual_information(x_i, x_j)
                
                # Granger因果系数
                gc, lag = self.granger_causality(x_j, x_i)  # j对i的Granger因果影响
                
                # 复合依赖强度
                dep_strength = (abs(rho) + (1.0 / (1.0 + dtw / 10.0)) + mi + gc) / 4.0
                A[i, j] = dep_strength
                lag_matrix[i, j] = lag
        
        return A, lag_matrix
    
    def find_dependency_groups(self, A, K=None):
        """
        基于依赖图连通性将变量划分为K个依赖组
        :param A: 依赖矩阵 [C, C]
        :param K: 组数（若None则自动确定）
        :return: 组分配列表
        """
        C = A.size(0)
        
        # 使用阈值二值化依赖图
        threshold = A.mean()
        adj_binary = (A > threshold).float().numpy()
        
        # 使用并查集找到连通分量
        parent = list(range(C))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(C):
            for j in range(i + 1, C):
                if adj_binary[i, j] > 0:
                    union(i, j)
        
        groups = {}
        for i in range(C):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        return list(groups.values())


class PhysicalInformedSamplePairs:
    """
    物理感知样本对构建器
    基于依赖图构建正样本对和负样本对
    """
    def __init__(self, tau_pos=0.5, tau_neg=0.1, th_dtw=10.0, th_mi=0.1):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.th_dtw = th_dtw
        self.th_mi = th_mi
    
    def construct_positive_pairs(self, embeddings, A, lag_matrix, time_series):
        """
        构建正样本对
        :param embeddings: [C, d] 变量嵌入
        :param A: 依赖矩阵 [C, C]
        :param lag_matrix: 滞后矩阵 [C, C]
        :param time_series: [T, C] 原始时间序列
        :return: 正样本对列表 [(anchor_idx, positive_idx, type, lag)]
        """
        C = embeddings.size(0)
        positives = []
        
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                
                rho = A[i, j]
                lag = lag_matrix[i, j].item()
                
                # 强同步正样本：高Pearson相关 + 低DTW距离 + 高互信息
                if rho > self.tau_pos:
                    positives.append((i, j, 'strong_sync', 0))
                
                # 时滞驱动正样本：基于Granger因果的最优滞后
                if lag > 0:
                    positives.append((i, j, 'time_lag', lag))
        
        return positives
    
    def construct_negative_pairs(self, embeddings, A):
        """
        构建负样本对（困难负样本）
        :param embeddings: [C, d] 变量嵌入
        :param A: 依赖矩阵 [C, C]
        :return: 负样本对列表 [(anchor_idx, negative_idx, type)]
        """
        C = embeddings.size(0)
        negatives = []
        
        for i in range(C):
            # 弱相关样本
            for j in range(C):
                if i != j and A[i, j] < self.tau_neg:
                    negatives.append((i, j, 'weak_corr'))
            
            # 随机干扰样本
            random_idx = np.random.choice([j for j in range(C) if j != i])
            negatives.append((i, random_idx, 'random'))
            
            # 反物理样本（通过低依赖强度识别）
            anti_candidates = [j for j in range(C) if j != i and A[i, j] < 0.05]
            if anti_candidates:
                anti_idx = np.random.choice(anti_candidates)
                negatives.append((i, anti_idx, 'anti_physical'))
        
        return negatives


class CPCLModule(nn.Module):
    """
    因果感知对比学习模块 (Causal Perception Contrastive Learning)
    完整实现依赖图构建、样本对构建、对比学习和组注意力聚合
    """
    def __init__(self, hidden_dim, tau_pos=0.5, tau_neg=0.1, th_dtw=10.0, th_mi=0.1, 
                 temperature=0.07, num_gat_layers=2, num_heads=4, dropout=0.1):
        super(CPCLModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # 依赖图构建器
        self.dep_graph_builder = VariableDependencyGraph(tau_pos, tau_neg, th_dtw, th_mi)
        
        # 样本对构建器
        self.sample_builder = PhysicalInformedSamplePairs(tau_pos, tau_neg, th_dtw, th_mi)
        
        # 组内注意力聚合
        self.group_attention = GroupAttentionAggregation(hidden_dim, num_heads)
        
        # 图注意力网络层（跨组信息传播）
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout, concat=(l < num_gat_layers - 1))
            for l in range(num_gat_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 可学习的变量查询向量（用于组内注意力）
        self.variable_queries = nn.Parameter(torch.randn(1, hidden_dim))
    
    def contrastive_loss(self, h, positives, negatives):
        """
        对比损失函数 (InfoNCE)
        :param h: [C, d] 物理感知表示
        :param positives: 正样本对列表
        :param negatives: 负样本对列表
        :return: 对比损失值
        """
        total_loss = 0.0
        count = 0
        
        for anchor_idx in range(h.size(0)):
            # 获取该锚点的正样本
            pos_indices = [p[1] for p in positives if p[0] == anchor_idx]
            neg_indices = [n[1] for n in negatives if n[0] == anchor_idx]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            anchor = h[anchor_idx]  # [d]
            pos_samples = h[pos_indices]  # [num_pos, d]
            neg_samples = h[neg_indices]  # [num_neg, d]
            
            # 计算相似度
            pos_sim = F.cosine_similarity(anchor.unsqueeze(0), pos_samples, dim=1) / self.temperature  # [num_pos]
            neg_sim = F.cosine_similarity(anchor.unsqueeze(0), neg_samples, dim=1) / self.temperature  # [num_neg]
            
            # InfoNCE损失
            numerator = torch.exp(pos_sim).sum()
            denominator = numerator + torch.exp(neg_sim).sum()
            
            loss = -torch.log(numerator / (denominator + 1e-8))
            total_loss += loss
            count += 1
        
        return total_loss / max(count, 1)
    
    def forward(self, embeddings, time_series):
        """
        CP-CL前向传播
        :param embeddings: [C, d] 初始变量嵌入
        :param time_series: [T, C] 原始时间序列
        :return: 优化后的物理感知表示 [C, d]
        """
        C, d = embeddings.shape
        
        # 构建变量依赖图
        A, lag_matrix = self.dep_graph_builder.build_dependency_graph(time_series)
        A = A.to(embeddings.device)
        lag_matrix = lag_matrix.to(embeddings.device)
        
        # 划分依赖组
        groups = self.dep_graph_builder.find_dependency_groups(A)
        
        # 构建样本对
        positives = self.sample_builder.construct_positive_pairs(embeddings, A, lag_matrix, time_series)
        negatives = self.sample_builder.construct_negative_pairs(embeddings, A)
        
        # 优化对比损失（通过可学习的变换使物理相关变量聚合）
        h = embeddings.clone()
        
        # 组内注意力聚合
        group_representations = []
        for group in groups:
            if len(group) == 0:
                continue
            group_embeds = h[group]  # [group_size, d]
            agg_rep = self.group_attention(group_embeds)  # [d]
            group_representations.append(agg_rep)
        
        # 跨组信息传播（GAT）
        if len(groups) > 1:
            h_group = torch.stack(group_representations)  # [num_groups, d]
            
            # 构建组间邻接矩阵
            num_groups = len(groups)
            adj_group = torch.zeros(num_groups, num_groups, device=h.device)
            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    # 计算组间连接强度
                    cross_strength = 0.0
                    for vi in groups[i]:
                        for vj in groups[j]:
                            cross_strength += A[vi, vj].item()
                    cross_strength /= max(len(groups[i]) * len(groups[j]), 1)
                    adj_group[i, j] = cross_strength
                    adj_group[j, i] = cross_strength
            
            # 应用GAT层
            for gat_layer in self.gat_layers:
                h_group = gat_layer(h_group, adj_group)
            
            # 将组表示映射回变量级别
            h_new = torch.zeros_like(h)
            for g_idx, group in enumerate(groups):
                for v_idx in group:
                    h_new[v_idx] = h[v_idx] + h_group[g_idx]  # 残差连接
            
            h = self.layer_norm(h_new)
        
        # 计算对比损失（仅在训练时）
        if self.training:
            loss_cl = self.contrastive_loss(h, positives, negatives)
            return h, loss_cl, A
        
        return h, torch.tensor(0.0, device=h.device), A
