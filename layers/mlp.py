"""
TEST-CDP 基础层模块
包含多层感知机(MLP)、图注意力网络(GAT)等基础组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """
    多层感知机模块
    用于将时间序列token映射到LLM的嵌入维度
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1, activation='tanh'):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数选择
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT)
    用于跨组信息传播，捕获变量间的全局依赖结构
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 可学习的线性变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力系数参数
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, h, adj):
        """
        前向传播
        :param h: 节点特征 [N, in_features]
        :param adj: 邻接矩阵 [N, N]
        :return: 更新后的节点特征 [N, out_features]
        """
        Wh = torch.mm(h, self.W)  # [N, out_features]
        
        # 计算注意力系数
        e = self._prepare_attentional_mechanism_input(Wh)  # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(e, self.a).squeeze(2))  # [N, N]
        
        # 掩码：只在邻接矩阵中有连接的节点间计算注意力
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # 聚合邻居信息
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """
        准备注意力机制输入：为每对节点拼接特征
        """
        N = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # [N*N, out_features]
        Wh_repeated_alternating = Wh.repeat(N, 1)  # [N*N, out_features]
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GroupAttentionAggregation(nn.Module):
    """
    组内注意力聚合模块
    在变量依赖图的每个连通组内进行注意力聚合
    """
    def __init__(self, hidden_dim, num_heads=4):
        super(GroupAttentionAggregation, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # 组查询向量
        self.group_queries = nn.Parameter(torch.randn(1, num_heads, self.head_dim))
        
        # 线性投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, group_embeddings):
        """
        组内注意力聚合
        :param group_embeddings: [group_size, hidden_dim]
        :return: 聚合后的组表示 [hidden_dim]
        """
        if group_embeddings.size(0) == 0:
            return torch.zeros(self.hidden_dim, device=group_embeddings.device)
        
        # 线性投影
        Q = self.group_queries  # [1, num_heads, head_dim]
        K = self.k_proj(group_embeddings).view(group_embeddings.size(0), self.num_heads, self.head_dim)
        V = self.v_proj(group_embeddings).view(group_embeddings.size(0), self.num_heads, self.head_dim)
        
        # 注意力计算
        scores = torch.einsum('h d, n h d -> h n', Q.squeeze(0), K) / self.scale  # [num_heads, group_size]
        attn = F.softmax(scores, dim=-1)  # [num_heads, group_size]
        
        # 加权聚合
        out = torch.einsum('h n, n h d -> h d', attn, V)  # [num_heads, head_dim]
        out = out.view(1, -1)  # [1, hidden_dim]
        out = self.out_proj(out)
        
        return out.squeeze(0)


class TemporalEncoder(nn.Module):
    """
    时间序列编码器
    使用1D卷积提取时间特征
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3, kernel_size=3, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 后续层
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        :param x: [batch, seq_len, input_dim]
        :return: [batch, seq_len, hidden_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        return x


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    为时间序列提供时间位置信息
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        :param x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class LayerNorm(nn.Module):
    """
    层归一化
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
