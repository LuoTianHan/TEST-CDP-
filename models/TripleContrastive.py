"""
三重对比表示层 (Triple Contrastive Representation Layer)
基于TEST论文实现：实例级(instance-wise)、特征级(feature-wise)和文本原型对齐(text-prototype-aligned)对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mlp import MLP, TemporalEncoder


class TripleContrastiveEncoder(nn.Module):
    """
    三重对比编码器
    融合实例级、特征级和文本原型对齐的对比学习
    """
    def __init__(self, input_dim, hidden_dim, llm_embed_dim, num_text_prototypes=10,
                 instance_temp=0.5, feature_temp=0.5, text_temp=0.5, dropout=0.1):
        super(TripleContrastiveEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.llm_embed_dim = llm_embed_dim
        self.num_text_prototypes = num_text_prototypes
        self.instance_temp = instance_temp
        self.feature_temp = feature_temp
        self.text_temp = text_temp
        
        # 时间序列编码器（使用TCN结构）
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim, num_layers=3, dropout=dropout)
        
        # 投影头（用于实例级对比）
        self.instance_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 文本原型（使用可学习原型替代实际文本嵌入）
        self.text_prototypes = nn.Parameter(torch.randn(num_text_prototypes, hidden_dim) * 0.02)
        
        # 特征级对比的参数
        self.feature_align_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 最终映射到LLM嵌入维度
        self.to_llm_embed = nn.Linear(hidden_dim, llm_embed_dim)
        
        # 自编码器解码器（用于重建验证）
        self.decoder = nn.Sequential(
            nn.Linear(llm_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def instance_wise_contrast(self, embeddings, aug_embeddings):
        """
        实例级对比学习
        :param embeddings: [batch, hidden_dim] 原始嵌入
        :param aug_embeddings: [batch, hidden_dim] 增强视图嵌入
        :return: 实例级对比损失
        """
        batch_size = embeddings.size(0)
        
        # 投影
        z = self.instance_projector(embeddings)  # [batch, hidden_dim]
        z_aug = self.instance_projector(aug_embeddings)  # [batch, hidden_dim]
        
        # 归一化
        z = F.normalize(z, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)
        
        # 正样本对：原始与增强视图
        pos_sim = torch.sum(z * z_aug, dim=-1) / self.instance_temp  # [batch]
        
        # 负样本对：batch内其他样本
        neg_sim = torch.mm(z, z_aug.t()) / self.instance_temp  # [batch, batch]
        
        # 对角线是正样本，排除
        mask = torch.eye(batch_size, device=z.device).bool()
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # InfoNCE损失
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sim), dim=-1)
        
        loss = -torch.log(numerator / (denominator + 1e-8)).mean()
        return loss
    
    def feature_wise_contrast(self, embeddings, aug_embeddings):
        """
        特征级对比学习
        :param embeddings: [batch, hidden_dim] 原始嵌入
        :param aug_embeddings: [batch, hidden_dim] 增强视图嵌入
        :return: 特征级对比损失
        """
        batch_size, feat_dim = embeddings.shape
        
        # 构建特征矩阵 [batch, hidden_dim]
        m = embeddings  # 锚点特征矩阵
        m_pos = aug_embeddings  # 正样本特征矩阵
        
        # 随机采样负样本
        neg_indices = torch.randperm(batch_size)
        m_neg = embeddings[neg_indices]  # 负样本特征矩阵
        
        # 对齐项：对齐同一特征列
        alignment_loss = 0.0
        for i in range(feat_dim):
            m_i = m[:, i]  # [batch]
            m_pos_i = m_pos[:, i]  # [batch]
            
            # 计算对齐相似度
            align_sim = F.cosine_similarity(m_i.unsqueeze(0), m_pos_i.unsqueeze(0), dim=1)
            alignment_loss += align_sim.mean()
        
        # 差异项：不同特征列之间保持差异
        diff_loss = 0.0
        for i in range(feat_dim):
            m_i = m[:, i]  # [batch]
            
            # 与其他特征列的对比
            pos_sims = []
            neg_sims = []
            for j in range(feat_dim):
                if i == j:
                    continue
                m_pos_j = m_pos[:, j]
                m_neg_j = m_neg[:, j]
                
                pos_sim = F.cosine_similarity(m_i.unsqueeze(0), m_pos_j.unsqueeze(0), dim=1)
                neg_sim = F.cosine_similarity(m_i.unsqueeze(0), m_neg_j.unsqueeze(0), dim=1)
                pos_sims.append(pos_sim)
                neg_sims.append(neg_sim)
            
            if len(pos_sims) > 0:
                pos_sims = torch.stack(pos_sims, dim=0)  # [num_features-1, batch]
                neg_sims = torch.stack(neg_sims, dim=0)
                
                numerator = torch.exp(pos_sims / self.feature_temp).sum(dim=0)
                denominator = numerator + torch.exp(neg_sims / self.feature_temp).sum(dim=0)
                diff_loss += -torch.log(numerator / (denominator + 1e-8)).mean()
        
        # 综合损失
        loss = -alignment_loss / feat_dim + diff_loss / feat_dim
        return loss
    
    def text_prototype_contrast(self, embeddings, aug_embeddings):
        """
        文本原型对齐对比学习
        将时间序列嵌入映射到文本原型坐标系
        :param embeddings: [batch, hidden_dim]
        :param aug_embeddings: [batch, hidden_dim]
        :return: 文本原型对齐损失
        """
        batch_size = embeddings.size(0)
        
        # 文本原型
        prototypes = self.text_prototypes  # [num_text_prototypes, hidden_dim]
        
        # 对齐项：保证TS嵌入和文本原型的空间范围一致
        align_loss = 0.0
        for i in range(self.num_text_prototypes):
            proto = prototypes[i]  # [hidden_dim]
            # 计算与所有样本的相似度
            sims = F.cosine_similarity(proto.unsqueeze(0), embeddings, dim=1)
            align_loss += sims.mean()
        
        # 对比项：使用文本原型作为坐标轴映射TS嵌入
        # 通过原型映射构造特征矩阵
        m = embeddings @ prototypes.t()  # [batch, num_text_prototypes]
        m_pos = aug_embeddings @ prototypes.t()
        
        # 在原型空间进行特征级对比
        contrast_loss = 0.0
        for i in range(self.num_text_prototypes):
            m_i = m[:, i]
            m_pos_i = m_pos[:, i]
            
            pos_sim = (m_i * m_pos_i).sum() / (torch.norm(m_i) * torch.norm(m_pos_i) + 1e-8)
            
            # 负样本
            neg_indices = torch.randperm(batch_size)
            m_neg = m[neg_indices]
            m_neg_i = m_neg[:, i]
            neg_sim = (m_i * m_neg_i).sum() / (torch.norm(m_i) * torch.norm(m_neg_i) + 1e-8)
            
            contrast_loss += -torch.log(
                torch.exp(pos_sim / self.text_temp) / 
                (torch.exp(pos_sim / self.text_temp) + torch.exp(neg_sim / self.text_temp) + 1e-8)
            )
        
        loss = -align_loss / self.num_text_prototypes + contrast_loss / self.num_text_prototypes
        return loss
    
    def data_augmentation(self, x):
        """
        数据增强：抖动和缩放策略
        :param x: [batch, seq_len, input_dim]
        :return: 增强后的时间序列
        """
        # 抖动：添加随机噪声
        noise = torch.randn_like(x) * 0.1
        x_jitter = x + noise
        
        # 缩放：幅度缩放
        scale = torch.randn(x.size(0), 1, 1, device=x.device) * 0.2 + 1.0
        x_scale = x * scale
        
        # 组合增强
        x_aug = x_jitter * 0.5 + x_scale * 0.5
        return x_aug
    
    def forward(self, x, return_all_losses=False):
        """
        三重对比表示层前向传播
        :param x: [batch, seq_len, input_dim] 时间序列输入
        :param return_all_losses: 是否返回所有损失分量
        :return: LLM兼容的嵌入, 对比损失, (可选)重建损失
        """
        # 编码原始输入
        h = self.temporal_encoder(x)  # [batch, seq_len, hidden_dim]
        
        # 全局池化得到实例级表示
        embeddings = h.mean(dim=1)  # [batch, hidden_dim]
        
        # 数据增强
        x_aug = self.data_augmentation(x)
        h_aug = self.temporal_encoder(x_aug)
        aug_embeddings = h_aug.mean(dim=1)
        
        # 三重对比损失
        loss_ins = self.instance_wise_contrast(embeddings, aug_embeddings)
        loss_feat = self.feature_wise_contrast(embeddings, aug_embeddings)
        loss_text = self.text_prototype_contrast(embeddings, aug_embeddings)
        
        # 总对比损失
        loss_cl = loss_ins + loss_feat + loss_text
        
        # 映射到LLM嵌入空间
        llm_embeddings = self.to_llm_embed(embeddings)  # [batch, llm_embed_dim]
        
        # 自编码重建（可选）
        recon = self.decoder(llm_embeddings)
        loss_ae = F.mse_loss(recon, x[:, -1, :] if x.size(1) > 1 else x.mean(dim=1))
        
        if return_all_losses:
            return llm_embeddings, {
                'total': loss_cl + 0.1 * loss_ae,
                'instance': loss_ins,
                'feature': loss_feat,
                'text': loss_text,
                'autoencode': loss_ae
            }
        
        return llm_embeddings, loss_cl + 0.1 * loss_ae
    
    def encode(self, x):
        """
        仅编码，不计算损失（用于推理）
        :param x: [batch, seq_len, input_dim]
        :return: [batch, llm_embed_dim]
        """
        h = self.temporal_encoder(x)
        embeddings = h.mean(dim=1)
        llm_embeddings = self.to_llm_embed(embeddings)
        return llm_embeddings
