"""
TEST-CDP主模型: 集成三重对比表示层、CP-CL模块和DPG模块的完整预测框架
使用冻结的LLaMA-2-7B作为骨干LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from models.TripleContrastive import TripleContrastiveEncoder
from models.CPCL import CPCLModule
from models.DPG import DPGModule


class TESTCDPModel(nn.Module):
    """
    TEST-CDP: Text prototype aligned embedding with Causal Perception and Dynamic Prompt
    整体框架:
    1. 三重对比表示层: 生成初始变量嵌入
    2. CP-CL模块: 基于因果感知的对比学习优化嵌入
    3. DPG模块: 动态生成任务特定提示
    4. 冻结LLM: 使用LLaMA-2-7B进行自回归预测
    """
    def __init__(self, configs):
        super(TESTCDPModel, self).__init__()
        
        self.configs = configs
        self.device = configs.device
        
        # 模型超参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.token_len = configs.token_len
        self.n_vars = configs.enc_in
        self.hidden_dim = configs.hidden_dim
        self.llm_embed_dim = configs.llm_embed_dim
        
        # ==================== 1. 冻结的LLM骨干 (LLaMA-2-7B) ====================
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            device_map="auto" if configs.use_gpu else None,
            low_cpu_mem_usage=True
        )
        
        # 冻结LLM所有参数
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
        
        self.llama_hidden_dim = configs.llm_embed_dim  # LLaMA-2-7B: 4096
        
        # ==================== 2. 三重对比表示层 ====================
        self.triple_encoder = TripleContrastiveEncoder(
            input_dim=configs.token_len,
            hidden_dim=configs.hidden_dim,
            llm_embed_dim=configs.llm_embed_dim,
            num_text_prototypes=configs.num_text_prototypes,
            instance_temp=configs.instance_temp,
            feature_temp=configs.feature_temp,
            text_temp=configs.text_temp,
            dropout=configs.dropout
        )
        
        # ==================== 3. CP-CL模块 ====================
        self.cpcl = CPCLModule(
            hidden_dim=configs.llm_embed_dim,
            tau_pos=configs.tau_pos,
            tau_neg=configs.tau_neg,
            th_dtw=configs.th_dtw,
            th_mi=configs.th_mi,
            temperature=configs.cpcl_temp,
            num_gat_layers=configs.num_gat_layers,
            num_heads=configs.num_heads,
            dropout=configs.dropout
        )
        
        # ==================== 4. DPG模块 ====================
        self.dpg = DPGModule(
            prompt_dim=configs.llm_embed_dim,
            hidden_dim=configs.llm_embed_dim,
            condition_dim=configs.condition_dim,
            beta=configs.dpg_beta,
            top_k=configs.dpg_top_k,
            dropout=configs.dropout
        )
        
        # ==================== 5. 可学习的软提示 (Soft Prompt) ====================
        self.soft_prompt_length = configs.prompt_length
        self.soft_prompt = nn.Parameter(torch.randn(configs.prompt_length, configs.llm_embed_dim) * 0.02)
        
        # ==================== 6. 预测头 ====================
        self.prediction_head = nn.Linear(configs.llm_embed_dim, configs.token_len)
        
        # ==================== 7. 归一化参数 ====================
        self.use_norm = configs.use_norm
        if self.use_norm:
            self.norm_mean = nn.Parameter(torch.zeros(configs.enc_in), requires_grad=False)
            self.norm_std = nn.Parameter(torch.ones(configs.enc_in), requires_grad=False)
        
        # 变量级投影（将每个变量的时间序列映射到嵌入空间）
        self.var_projector = nn.Linear(configs.seq_len, configs.llm_embed_dim)
        
        print(f"[TEST-CDP] 模型初始化完成")
        print(f"[TEST-CDP] LLM: LLaMA-2-7B (冻结)")
        print(f"[TEST-CDP] 序列长度: {configs.seq_len}, 预测长度: {configs.pred_len}")
        print(f"[TEST-CDP] 变量数: {configs.enc_in}, 嵌入维度: {configs.llm_embed_dim}")
    
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        长期预测前向传播
        :param x_enc: [batch, seq_len, n_vars] 历史观测
        :param x_mark_enc: [batch, seq_len, n_mark] 时间标记（可选）
        :param x_dec: [batch, label_len+pred_len, n_vars] 解码器输入（可选）
        :param x_mark_dec: 解码器时间标记（可选）
        :return: [batch, pred_len, n_vars] 预测结果
        """
        batch_size, seq_len, n_vars = x_enc.shape
        
        # Step 1: 数据归一化
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        
        # Step 2: 将多变量序列切分为token
        # x_enc: [batch, seq_len, n_vars] -> [batch * n_vars, token_num, token_len]
        x_enc = x_enc.permute(0, 2, 1)  # [batch, n_vars, seq_len]
        x_enc = x_enc.reshape(batch_size * n_vars, -1)  # [batch * n_vars, seq_len]
        
        # 滑动窗口切分token
        if seq_len % self.token_len == 0:
            token_num = seq_len // self.token_len
            x_tokens = x_enc.reshape(batch_size * n_vars, token_num, self.token_len)
        else:
            # 填充或截断到token_len的倍数
            pad_len = (self.token_len - seq_len % self.token_len) % self.token_len
            if pad_len > 0:
                x_padded = F.pad(x_enc, (0, pad_len), mode='replicate')
            else:
                x_padded = x_enc
            token_num = x_padded.size(1) // self.token_len
            x_tokens = x_padded.reshape(batch_size * n_vars, token_num, self.token_len)
        
        # Step 3: 三重对比表示层生成初始嵌入
        # 对每个变量独立编码
        var_embeddings = []
        for i in range(batch_size * n_vars):
            var_tokens = x_tokens[i:i+1]  # [1, token_num, token_len]
            llm_embed = self.triple_encoder.encode(var_tokens)  # [1, llm_embed_dim]
            var_embeddings.append(llm_embed)
        
        var_embeddings = torch.cat(var_embeddings, dim=0)  # [batch * n_vars, llm_embed_dim]
        var_embeddings = var_embeddings.view(batch_size, n_vars, self.llm_embed_dim)  # [batch, n_vars, llm_embed_dim]
        
        # Step 4: CP-CL模块优化变量嵌入（基于物理依赖）
        optimized_embeddings = []
        cpcl_losses = []
        dependency_graphs = []
        
        for b in range(batch_size):
            var_emb = var_embeddings[b]  # [n_vars, llm_embed_dim]
            ts_data = x_enc[b * n_vars:(b+1) * n_vars]  # [n_vars, seq_len]
            ts_data = ts_data.transpose(0, 1)  # [seq_len, n_vars]
            
            if self.training:
                h_opt, loss_cl, A = self.cpcl(var_emb, ts_data)
                cpcl_losses.append(loss_cl)
            else:
                h_opt, _, A = self.cpcl(var_emb, ts_data)
            
            optimized_embeddings.append(h_opt)
            dependency_graphs.append(A)
        
        optimized_embeddings = torch.stack(optimized_embeddings, dim=0)  # [batch, n_vars, llm_embed_dim]
        
        # Step 5: DPG模块生成动态提示
        dynamic_prompts = []
        prompt_weights = []
        
        for b in range(batch_size):
            h_vars = optimized_embeddings[b]  # [n_vars, llm_embed_dim]
            ts_data = x_enc[b * n_vars:(b+1) * n_vars].transpose(0, 1)  # [seq_len, n_vars]
            
            if self.training:
                P_dyn = self.dpg.generate_dynamic_prompt(
                    h_vars, ts_data, pred_len=self.pred_len
                )
            else:
                P_dyn, weights_info = self.dpg.generate_dynamic_prompt(
                    h_vars, ts_data, pred_len=self.pred_len, return_weights=True
                )
                prompt_weights.append(weights_info)
            
            dynamic_prompts.append(P_dyn)
        
        dynamic_prompts = torch.stack(dynamic_prompts, dim=0)  # [batch, llm_embed_dim]
        
        # Step 6: 组合输入序列（软提示 + 动态提示 + 优化后的变量嵌入）
        # 软提示 [batch, prompt_length, llm_embed_dim]
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 动态提示 [batch, 1, llm_embed_dim]
        dyn_prompt_batch = dynamic_prompts.unsqueeze(1)
        
        # 变量嵌入 [batch, n_vars, llm_embed_dim]
        var_embed_batch = optimized_embeddings
        
        # 拼接所有输入: [batch, prompt_length + 1 + n_vars, llm_embed_dim]
        llm_input = torch.cat([soft_prompt_batch, dyn_prompt_batch, var_embed_batch], dim=1)
        
        # Step 7: 输入到冻结的LLM
        # LLM期望输入是嵌入向量
        llm_output = self.llama.model(inputs_embeds=llm_input).last_hidden_state  # [batch, seq_len, llm_embed_dim]
        
        # Step 8: 从LLM输出中提取预测token并映射回变量空间
        # 取最后pred_len对应的输出位置进行预测
        pred_output = llm_output[:, -n_vars:, :]  # [batch, n_vars, llm_embed_dim]
        
        # 预测头: 映射到token_len维度
        pred_tokens = self.prediction_head(pred_output)  # [batch, n_vars, token_len]
        
        # 重组为预测序列 [batch, pred_len, n_vars]
        # 假设pred_len是token_len的倍数
        pred_vars = pred_tokens.transpose(1, 2)  # [batch, token_len, n_vars]
        
        # 扩展/截断到目标预测长度
        if self.pred_len <= self.token_len:
            preds = pred_vars[:, :self.pred_len, :]
        else:
            # 重复预测或插值
            repeat_times = self.pred_len // self.token_len + 1
            pred_expanded = pred_vars.repeat(1, repeat_times, 1)
            preds = pred_expanded[:, :self.pred_len, :]
        
        # Step 9: 反归一化
        if self.use_norm:
            preds = preds * stdev + means
        
        # Step 10: 收集辅助损失
        aux_loss = torch.tensor(0.0, device=preds.device)
        if self.training and len(cpcl_losses) > 0:
            cpcl_loss = torch.stack(cpcl_losses).mean()
            aux_loss = aux_loss + self.configs.cpcl_weight * cpcl_loss
        
        if self.training:
            return preds, aux_loss
        
        return preds
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播入口
        """
        if self.training:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)


class Model(TESTCDPModel):
    """
    兼容AutoTimes接口的模型包装类
    """
    pass
