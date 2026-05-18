"""
DPG模块: 动态提示生成器 (Dynamic Prompt Generator)
实现双银行架构、三维上下文感知和自适应提示生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class SemanticBank(nn.Module):
    """
    语义银行
    定义6类可学习的提示嵌入向量，对应不同的时间序列模式
    """
    def __init__(self, prompt_dim, num_classes=6):
        super(SemanticBank, self).__init__()
        self.num_classes = num_classes
        self.prompt_dim = prompt_dim
        
        # 6类语义模式的可学习嵌入
        # P1: Trend - 趋势演化模式
        # P2: Season - 季节周期性模式
        # P3: Spike - 尖峰/异常波动模式
        # P4: Corr - 变量间相关强度模式
        # P5: Volat - 波动率/方差特征模式
        # P6: Cycle - 循环模式
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_dim) * 0.02) for _ in range(num_classes)
        ])
        
        # 模式名称（用于可视化解释）
        self.prompt_names = ['Trend', 'Season', 'Spike', 'Corr', 'Volat', 'Cycle']
    
    def get_prompts(self):
        """
        获取所有语义提示
        :return: [num_classes, prompt_dim]
        """
        return torch.cat([p for p in self.prompts], dim=0)
    
    def get_prompt_name(self, idx):
        """获取提示名称"""
        return self.prompt_names[idx]


class HistoricalExperienceBank:
    """
    历史经验银行
    存储历史最优提示配置 H = {(c_j, p_j, e_error)}
    """
    def __init__(self, max_size=1000, condition_dim=64):
        self.max_size = max_size
        self.condition_dim = condition_dim
        
        # 存储历史记录: (condition_feature, prompt_embedding, error)
        self.history = deque(maxlen=max_size)
    
    def add(self, condition, prompt, error):
        """
        添加历史记录
        :param condition: 条件特征向量 [condition_dim]
        :param prompt: 最优提示嵌入 [prompt_dim]
        :param error: 验证误差标量
        """
        self.history.append({
            'condition': condition.detach().cpu(),
            'prompt': prompt.detach().cpu(),
            'error': error
        })
    
    def retrieve(self, current_condition, top_k=5, similarity_threshold=0.5):
        """
        检索历史经验
        :param current_condition: 当前条件特征 [condition_dim]
        :param top_k: 返回最相似的k条记录
        :param similarity_threshold: 相似度阈值
        :return: 匹配的历史记录列表
        """
        if len(self.history) == 0:
            return []
        
        current_condition = current_condition.detach().cpu()
        
        # 计算余弦相似度
        similarities = []
        for record in self.history:
            cond = record['condition']
            sim = F.cosine_similarity(current_condition.unsqueeze(0), cond.unsqueeze(0))
            similarities.append(sim.item())
        
        # 筛选相似度高于阈值的记录
        candidate_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        
        # 按相似度排序，取top-k
        candidate_indices = sorted(candidate_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        
        candidates = []
        for idx in candidate_indices:
            record = self.history[idx]
            candidates.append({
                'condition': record['condition'],
                'prompt': record['prompt'],
                'error': record['error'],
                'similarity': similarities[idx]
            })
        
        return candidates
    
    def size(self):
        return len(self.history)


class ContextExtractor(nn.Module):
    """
    三维上下文提取器
    提取全局上下文G、任务上下文T和变量上下文D
    """
    def __init__(self, hidden_dim, max_seq_len=5000):
        super(ContextExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 全局上下文：统计特征提取（均值、方差、趋势强度、FFT频率特征）
        self.global_context_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 8维统计特征
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 任务上下文：预测任务元信息
        self.task_context_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 预测长度、时间粒度、领域标识等
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 变量上下文：物理感知表示的聚合
        self.variable_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def extract_global_context(self, time_series):
        """
        提取全局统计上下文
        :param time_series: [batch, seq_len, n_vars] 或 [seq_len, n_vars]
        :return: 全局上下文向量 [hidden_dim]
        """
        if time_series.dim() == 3:
            time_series = time_series.mean(dim=0)  # [seq_len, n_vars]
        
        seq_len, n_vars = time_series.shape
        
        # 计算统计特征
        features = []
        
        for v in range(n_vars):
            var_series = time_series[:, v]
            
            # 均值
            mean_val = var_series.mean().item()
            # 方差
            var_val = var_series.var().item()
            # 趋势强度（线性回归斜率）
            t = torch.arange(len(var_series), dtype=torch.float32, device=var_series.device)
            slope = ((t - t.mean()) * (var_series - var_series.mean())).sum() / ((t - t.mean()) ** 2).sum()
            trend_strength = slope.item()
            # 峰度
            kurtosis = ((var_series - mean_val) ** 4).mean().item() / (var_val ** 2 + 1e-8)
            
            features.extend([mean_val, var_val, abs(trend_strength), kurtosis])
        
        # 取所有变量的平均特征
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=time_series.device)
        feat_tensor = feat_tensor.view(n_vars, 4).mean(dim=0)  # [4]
        
        # FFT频率特征
        fft_vals = torch.fft.rfft(time_series[:, 0])
        fft_magnitude = torch.abs(fft_vals)
        dominant_freq = torch.argmax(fft_magnitude).item() / len(fft_magnitude)
        freq_energy = fft_magnitude.sum().item()
        
        # 组合全局特征 [8]
        global_feat = torch.cat([
            feat_tensor,  # [4]
            torch.tensor([dominant_freq, freq_energy, seq_len / 100.0, n_vars / 10.0], 
                        dtype=torch.float32, device=time_series.device)  # [4]
        ])
        
        return self.global_context_proj(global_feat)
    
    def extract_task_context(self, pred_len, time_granularity=1.0, domain_id=0):
        """
        提取任务上下文
        :param pred_len: 预测长度
        :param time_granularity: 时间粒度
        :param domain_id: 领域标识
        :return: 任务上下文向量 [hidden_dim]
        """
        task_feat = torch.tensor([pred_len / 100.0, time_granularity, domain_id / 10.0, 0.0], 
                                 dtype=torch.float32)
        return self.task_context_proj(task_feat)
    
    def extract_variable_context(self, h_vars):
        """
        提取变量上下文：聚合物理感知表示
        :param h_vars: [n_vars, hidden_dim] CP-CL优化后的变量表示
        :return: 变量上下文向量 [hidden_dim]
        """
        D = h_vars.mean(dim=0)  # [hidden_dim]
        return self.variable_context_proj(D)


class DPGModule(nn.Module):
    """
    动态提示生成器 (Dynamic Prompt Generator)
    基于双银行架构和三维上下文自适应生成任务特定提示
    """
    def __init__(self, prompt_dim, hidden_dim, condition_dim=64, 
                 beta=0.3, top_k=5, max_history=1000, dropout=0.1):
        super(DPGModule, self).__init__()
        self.prompt_dim = prompt_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.beta = beta  # 历史经验融合系数
        self.top_k = top_k
        
        # 语义银行
        self.semantic_bank = SemanticBank(prompt_dim)
        
        # 历史经验银行（非参数化，运行时维护）
        self.historical_bank = HistoricalExperienceBank(max_history, condition_dim)
        
        # 上下文提取器
        self.context_extractor = ContextExtractor(hidden_dim)
        
        # 自适应加权MLP：根据三维上下文计算语义模式权重
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # [D || T || G]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # 6类语义模式权重
        )
        
        # 条件特征投影（用于历史经验匹配）
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim)
        )
    
    def compute_semantic_weights(self, D, T, G):
        """
        计算语义模式自适应权重
        :param D: 变量上下文 [hidden_dim]
        :param T: 任务上下文 [hidden_dim]
        :param G: 全局上下文 [hidden_dim]
        :return: 语义权重 alpha [6]
        """
        context = torch.cat([D, T, G])  # [hidden_dim * 3]
        logits = self.adaptive_mlp(context)  # [6]
        alpha = F.softmax(logits, dim=-1)
        return alpha
    
    def generate_dynamic_prompt(self, h_vars, time_series, pred_len=96, 
                                 time_granularity=1.0, domain_id=0, 
                                 return_weights=False):
        """
        生成动态提示
        :param h_vars: [n_vars, hidden_dim] CP-CL优化后的变量表示
        :param time_series: [seq_len, n_vars] 原始时间序列
        :param pred_len: 预测长度
        :param time_granularity: 时间粒度
        :param domain_id: 领域标识
        :param return_weights: 是否返回权重用于可视化
        :return: 动态提示 [prompt_dim], (可选)权重信息
        """
        # Step 1: 三维上下文提取
        D = self.context_extractor.extract_variable_context(h_vars)  # [hidden_dim]
        T = self.context_extractor.extract_task_context(pred_len, time_granularity, domain_id)  # [hidden_dim]
        G = self.context_extractor.extract_global_context(time_series)  # [hidden_dim]
        
        # 确保所有上下文在同一设备
        device = h_vars.device
        D, T, G = D.to(device), T.to(device), G.to(device)
        
        # Step 2: 自适应加权（语义银行）
        alpha = self.compute_semantic_weights(D, T, G)  # [6]
        semantic_prompts = self.semantic_bank.get_prompts().to(device)  # [6, prompt_dim]
        
        P_semantic = torch.sum(alpha.unsqueeze(1) * semantic_prompts, dim=0)  # [prompt_dim]
        
        # Step 3: 历史经验检索与融合
        condition = self.condition_proj(D)  # [condition_dim]
        
        candidates = self.historical_bank.retrieve(condition, top_k=self.top_k)
        
        P_historical = torch.zeros(self.prompt_dim, device=device)
        if len(candidates) > 0:
            total_sim = sum(c['similarity'] for c in candidates)
            for cand in candidates:
                weight = cand['similarity'] / (total_sim + 1e-8)
                P_historical += weight * cand['prompt'].to(device)
        
        # Step 4: 动态组合
        P_dyn = P_semantic + self.beta * P_historical  # [prompt_dim]
        
        # 存储当前配置到历史银行（仅在训练时）
        if self.training and len(candidates) > 0:
            # 在实际训练中应该在验证后调用add
            pass
        
        if return_weights:
            weights_info = {
                'semantic_weights': alpha.detach().cpu().numpy(),
                'semantic_names': self.semantic_bank.prompt_names,
                'historical_similarities': [c['similarity'] for c in candidates],
                'num_candidates': len(candidates)
            }
            return P_dyn, weights_info
        
        return P_dyn
    
    def update_history(self, condition, prompt, error):
        """
        更新历史经验银行
        :param condition: 条件特征 [condition_dim]
        :param prompt: 使用的提示 [prompt_dim]
        :param error: 验证误差
        """
        self.historical_bank.add(condition, prompt, error)
    
    def get_semantic_prompt_weights(self):
        """
        获取语义提示的当前权重（用于可视化）
        """
        return {name: p.norm().item() for name, p in zip(self.semantic_bank.prompt_names, self.semantic_bank.prompts)}
