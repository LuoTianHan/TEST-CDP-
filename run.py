"""
TEST-CDP: 因果感知对比学习与检索增强动态提示的时间序列预测
=============================================================
基于大型语言模型的多变量时间序列预测框架

使用方法:
    python run.py --task_name long_term_forecast --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --pred_len 96
"""

import argparse
import os
import random
import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import print_args


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TEST-CDP: Time Series Forecasting with Causal Perception Contrastive Learning and Retrieval-Enhanced Dynamic Prompt')
    
    # ==================== 基本配置 ====================
    parser.add_argument('--task_name', type=str, required=True,
                        default='long_term_forecast',
                        help='任务名称: [long_term_forecast, short_term_forecast, zero_shot_forecasting]')
    parser.add_argument('--is_training', type=int, required=True, default=1,
                        help='是否训练: 1=训练, 0=测试')
    parser.add_argument('--model_id', type=str, required=True, default='TEST_CDP_ETTh1_96',
                        help='模型ID，用于标识实验')
    parser.add_argument('--model', type=str, required=True, default='TEST_CDP_Llama',
                        help='模型名称: TEST_CDP_Llama')
    
    # ==================== 数据加载配置 ====================
    parser.add_argument('--data', type=str, required=True, default='ETTh1',
                        help='数据集类型: [ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic, ILI]')
    parser.add_argument('--root_path', type=str, default='./data/ETT/',
                        help='数据文件根目录')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测类型: M=多变量预测多变量, S=单变量预测单变量, MS=多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT',
                        help='目标变量名（S或MS任务时使用）')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间频率: [s, t, h, d, b, w, m]')
    parser.add_argument('--scale', type=bool, default=True,
                        help='是否对数据进行标准化')
    parser.add_argument('--timeenc', type=int, default=0,
                        help='时间编码方式: 0=使用原始时间特征, 1=使用正弦位置编码')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='模型检查点保存路径')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    
    # ==================== 预测任务配置 ====================
    parser.add_argument('--seq_len', type=int, default=672,
                        help='输入序列长度（回看窗口大小）')
    parser.add_argument('--label_len', type=int, default=576,
                        help='标签长度（解码器输入长度）')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='预测长度（预测未来多少个时间步）')
    parser.add_argument('--token_len', type=int, default=96,
                        help='token长度（时间序列切分长度）')
    
    # ==================== 模型结构配置 ====================
    # LLM配置
    parser.add_argument('--llm_ckp_dir', type=str, default='./llama2-7b/',
                        help='LLaMA-2-7B预训练模型路径')
    parser.add_argument('--llm_embed_dim', type=int, default=4096,
                        help='LLM嵌入维度（LLaMA-2-7B为4096）')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度（编码器和中间层）')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='输入变量数（特征维度）')
    parser.add_argument('--dec_in', type=int, default=7,
                        help='解码器输入变量数')
    parser.add_argument('--c_out', type=int, default=7,
                        help='输出变量数')
    
    # 软提示配置
    parser.add_argument('--prompt_length', type=int, default=16,
                        help='软提示长度')
    
    # 三重对比表示层配置
    parser.add_argument('--num_text_prototypes', type=int, default=10,
                        help='文本原型数量')
    parser.add_argument('--instance_temp', type=float, default=0.5,
                        help='实例级对比学习温度系数')
    parser.add_argument('--feature_temp', type=float, default=0.5,
                        help='特征级对比学习温度系数')
    parser.add_argument('--text_temp', type=float, default=0.5,
                        help='文本原型对比学习温度系数')
    
    # CP-CL模块配置
    parser.add_argument('--tau_pos', type=float, default=0.5,
                        help='正样本Pearson相关系数阈值')
    parser.add_argument('--tau_neg', type=float, default=0.1,
                        help='负样本Pearson相关系数阈值')
    parser.add_argument('--th_dtw', type=float, default=10.0,
                        help='DTW距离阈值')
    parser.add_argument('--th_mi', type=float, default=0.1,
                        help='互信息阈值')
    parser.add_argument('--cpcl_temp', type=float, default=0.07,
                        help='CP-CL对比学习温度系数')
    parser.add_argument('--cpcl_weight', type=float, default=0.1,
                        help='CP-CL损失权重')
    parser.add_argument('--num_gat_layers', type=int, default=2,
                        help='GAT层数')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='注意力头数')
    
    # DPG模块配置
    parser.add_argument('--condition_dim', type=int, default=64,
                        help='条件特征维度')
    parser.add_argument('--dpg_beta', type=float, default=0.3,
                        help='历史经验融合系数')
    parser.add_argument('--dpg_top_k', type=int, default=5,
                        help='历史经验检索top-k')
    
    # 通用配置
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比率')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='是否使用数据归一化')
    
    # ==================== 优化配置 ====================
    parser.add_argument('--train_epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值（验证损失不改善的轮数）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减（L2正则化）')
    parser.add_argument('--lradj', type=str, default='cosine',
                        help='学习率调整策略: [type1, type2, type3, cosine, constant]')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='是否使用混合精度训练')
    
    # ==================== GPU配置 ====================
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的GPU编号')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='是否使用多GPU训练')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='分布式训练的本地进程编号')
    
    # ==================== 其他配置 ====================
    parser.add_argument('--itr', type=int, default=1,
                        help='实验重复次数')
    parser.add_argument('--des', type=str, default='TEST_CDP',
                        help='实验描述')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否保存可视化结果')
    parser.add_argument('--seed', type=int, default=2021,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置设备
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    
    # 打印配置
    print_args(args)
    
    # 运行实验
    print("\n" + "="*60)
    print("初始化 TEST-CDP 实验")
    print("="*60 + "\n")
    
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        raise ValueError(f"不支持的任务类型: {args.task_name}")
    
    # 多次实验取平均
    all_results = []
    for ii in range(args.itr):
        setting = f"{args.model_id}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.hidden_dim}_nh{args.num_heads}_el{args.num_gat_layers}_dl{args.dropout}_lr{args.learning_rate}_wd{args.weight_decay}_bs{args.batch_size}_{args.des}_{ii}"
        
        print(f"\n[实验 {ii+1}/{args.itr}] 设置: {setting}")
        
        exp = Exp(args)
        
        if args.is_training:
            print("\n[阶段1/2] 开始训练...")
            exp.train(setting)
            print("\n[阶段2/2] 开始测试...")
            results = exp.test(setting, test=1)
        else:
            print("\n[测试模式] 加载预训练模型进行测试...")
            results = exp.test(setting, test=1)
        
        all_results.append(results)
        
        # 清理GPU缓存
        if args.use_gpu:
            torch.cuda.empty_cache()
    
    # 打印平均结果
    if len(all_results) > 1:
        print("\n" + "="*60)
        print(f"[平均结果] {args.itr}次实验平均")
        print("="*60)
        for key in ['MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE']:
            values = [r[key] for r in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.6f} ± {std_val:.6f}")
        print("="*60 + "\n")
    
    print("[完成] TEST-CDP实验结束")


if __name__ == '__main__':
    main()
