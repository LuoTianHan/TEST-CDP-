# TEST-CDP: Text prototype aligned embedding with Causal Perception and Dynamic Prompt

基于大型语言模型的多变量时间序列预测框架，集成了因果感知对比学习（CP-CL）和检索增强动态提示生成（DPG）。

## 论文信息

**论文标题**: Time Series Forecasting with Causal Perception Contrastive Learning and Retrieval-Enhanced Dynamic Prompt via Large Language Models

## 核心创新

### 1. 因果感知对比学习模块 (CP-CL)

- **变量依赖图构建**: 基于多维统计度量（Pearson相关、DTW距离、互信息、Granger因果检验）构建复合依赖图
- **物理感知样本对构建**: 
  - 正样本: 强同步正样本、时滞驱动正样本
  - 负样本: 弱相关样本、随机干扰样本、反物理样本
- **判别式嵌入空间优化**: 通过InfoNCE对比损失使物理相关变量聚合
- **组内注意力聚合 + GAT跨组传播**: 捕获全局依赖结构

### 2. 检索增强动态提示生成器 (DPG)

- **双银行架构**: 
  - 语义银行: 6类可学习提示嵌入（Trend/Season/Spike/Corr/Volat/Cycle）
  - 历史经验银行: 存储历史最优提示配置
- **三维上下文感知**: 全局上下文G、任务上下文T、变量上下文D
- **自适应提示生成**: 相似度匹配 → 自适应加权 → 动态组合

### 3. 三重对比表示层

- 实例级对比学习 (Instance-wise Contrast)
- 特征级对比学习 (Feature-wise Contrast)
- 文本原型对齐对比学习 (Text-Prototype-Aligned Contrast)

## 项目结构

```
TEST-CDP/
├── run.py                          # 主入口脚本
├── requirements.txt                # 依赖包列表
├── README.md                       # 项目说明
├── .gitignore                      # Git忽略文件
│
├── models/                         # 模型定义
│   ├── __init__.py
│   ├── TEST_CDP_Llama.py          # TEST-CDP主模型
│   ├── TripleContrastive.py       # 三重对比表示层
│   ├── CPCL.py                    # 因果感知对比学习模块
│   └── DPG.py                     # 动态提示生成器模块
│
├── layers/                         # 基础层组件
│   ├── __init__.py
│   └── mlp.py                     # MLP、GAT、注意力等基础层
│
├── data_provider/                  # 数据加载模块
│   ├── __init__.py
│   └── data_factory.py            # 数据集工厂函数
│
├── exp/                            # 实验逻辑
│   ├── __init__.py
│   ├── exp_basic.py               # 实验基类
│   └── exp_long_term_forecasting.py # 长期预测实验
│
├── utils/                          # 工具函数
│   ├── __init__.py
│   ├── tools.py                   # 学习率调整、早停、可视化等
│   └── metrics.py                 # 评估指标(MSE/MAE/RMSE等)
│
└── scripts/                        # 运行脚本
    ├── run_ettm1.sh               # ETTm1数据集实验
    ├── run_all_datasets.sh        # 全数据集实验
    └── run_zero_shot.sh           # 零样本跨域实验
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.35.0 (用于加载LLaMA-2-7B)
- numpy, pandas, scikit-learn, scipy
- statsmodels (Granger因果检验)
- fastdtw (DTW距离计算)

## 数据准备

### 数据集下载

| 数据集 | 描述 | 变量数 | 频率 |
|--------|------|--------|------|
| ETTh1/ETTh2 | 电力变压器数据 | 7 | 1小时 |
| ETTm1/ETTm2 | 电力变压器数据 | 7 | 15分钟 |
| Weather | 气象数据 | 21 | 10分钟 |
| ECL | 电力消耗数据 | 321 | 1小时 |
| Traffic | 道路占用率数据 | 862 | 1小时 |
| ILI | 流感样疾病监测数据 | 7 | 1周 |

将数据集放置在 `./data/` 目录下：

```bash
mkdir -p data/ETT-small data/weather data/electricity data/traffic data/illness
```

### LLM模型准备

下载 LLaMA-2-7B 模型并放置在指定目录：

```bash
mkdir -p ./llama2-7b/
# 将下载的模型文件放入该目录
```

## 使用说明

### 基本训练命令

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_672_96 \
    --model TEST_CDP_Llama \
    --data ETTh1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 672 \
    --pred_len 96 \
    --enc_in 7 \
    --train_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --llm_ckp_dir ./llama2-7b/
```

### 测试模式

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --model_id ETTh1_672_96 \
    --model TEST_CDP_Llama \
    --data ETTh1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTh1.csv \
    --pred_len 96 \
    --enc_in 7
```

### 批量运行实验

```bash
# ETTm1数据集全部预测长度实验
bash scripts/run_ettm1.sh

# 全部8个数据集实验
bash scripts/run_all_datasets.sh

# 零样本跨域迁移实验
bash scripts/run_zero_shot.sh
```

## 关键超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| seq_len | 672 | 输入序列长度（回看窗口） |
| pred_len | 96 | 预测长度 |
| hidden_dim | 512 | 编码器隐藏维度 |
| llm_embed_dim | 4096 | LLM嵌入维度（LLaMA-2-7B） |
| prompt_length | 16 | 软提示长度 |
| num_gat_layers | 2 | GAT层数 |
| num_heads | 4 | 注意力头数 |
| tau_pos | 0.5 | 正样本相关阈值 |
| tau_neg | 0.1 | 负样本相关阈值 |
| cpcl_weight | 0.1 | CP-CL损失权重 |
| dpg_beta | 0.3 | 历史经验融合系数 |
| learning_rate | 1e-4 | 学习率 |
| weight_decay | 0.01 | 权重衰减 |
| batch_size | 32 | 批次大小 |

## 实验结果

TEST-CDP在8个真实数据集上取得了优异的性能：

| 数据集 | MSE | MAE |
|--------|-----|-----|
| ETTh2 | 0.345 | 0.385 |
| Weather | 0.221 | 0.263 |
| ILI | 1.903 | 2.063 |

## 引用

如果本工作对您的研究有帮助，请引用以下论文：

```bibtex
@article{wang2025testcdp,
  title={Time Series Forecasting with Causal Perception Contrastive Learning and Retrieval-Enhanced Dynamic Prompt via Large Language Models},
  author={Wang, Haibin and Yu, Shenyang and Liu, Wenjie},
  journal={},
  year={2025}
}
```

## 许可证

本项目基于 MIT 许可证开源。
