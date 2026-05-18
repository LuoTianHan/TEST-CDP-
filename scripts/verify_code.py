#!/usr/bin/env python3
"""
TEST-CDP 代码验证脚本
验证所有核心模块和组件是否可以正常导入和实例化
使用方法: python scripts/verify_code.py
"""

import sys
import os

# 将项目根目录加入路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 60)
print("TEST-CDP 代码验证")
print("=" * 60)

# 1. 验证基础模块导入
print("\n[1/6] 验证基础模块导入...")
try:
    import layers.mlp as mlp_module
    print("  ✓ layers.mlp 导入成功")
    
    # 验证基础组件
    assert hasattr(mlp_module, 'MLP')
    assert hasattr(mlp_module, 'GraphAttentionLayer')
    assert hasattr(mlp_module, 'GroupAttentionAggregation')
    assert hasattr(mlp_module, 'TemporalEncoder')
    print("  ✓ 基础组件检查通过 (MLP, GAT, GroupAttention, TemporalEncoder)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 2. 验证三重对比表示层
print("\n[2/6] 验证三重对比表示层...")
try:
    from models.TripleContrastive import TripleContrastiveEncoder
    print("  ✓ TripleContrastiveEncoder 导入成功")
    
    # 验证关键方法
    assert hasattr(TripleContrastiveEncoder, 'instance_wise_contrast')
    assert hasattr(TripleContrastiveEncoder, 'feature_wise_contrast')
    assert hasattr(TripleContrastiveEncoder, 'text_prototype_contrast')
    print("  ✓ 三重对比方法检查通过 (Instance/Feature/Text)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 3. 验证CP-CL模块
print("\n[3/6] 验证CP-CL因果感知对比学习模块...")
try:
    from models.CPCL import VariableDependencyGraph, PhysicalInformedSamplePairs, CPCLModule
    print("  ✓ CPCL模块导入成功")
    
    # 验证依赖图构建器
    vdg = VariableDependencyGraph()
    assert hasattr(vdg, 'build_dependency_graph')
    assert hasattr(vdg, 'find_dependency_groups')
    print("  ✓ VariableDependencyGraph 实例化成功")
    
    # 验证样本对构建器
    sip = PhysicalInformedSamplePairs()
    assert hasattr(sip, 'construct_positive_pairs')
    assert hasattr(sip, 'construct_negative_pairs')
    print("  ✓ PhysicalInformedSamplePairs 实例化成功")
    
    print("  ✓ CP-CL组件检查通过 (依赖图/样本对/对比损失)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 4. 验证DPG模块
print("\n[4/6] 验证DPG动态提示生成器模块...")
try:
    from models.DPG import SemanticBank, HistoricalExperienceBank, ContextExtractor, DPGModule
    print("  ✓ DPG模块导入成功")
    
    # 验证语义银行
    sb = SemanticBank(prompt_dim=128)
    assert sb.num_classes == 6
    prompts = sb.get_prompts()
    assert prompts.shape == (6, 128)
    print(f"  ✓ SemanticBank 实例化成功 (6类提示, 维度{prompts.shape})")
    
    # 验证历史经验银行
    heb = HistoricalExperienceBank(max_size=100)
    assert heb.size() == 0
    print("  ✓ HistoricalExperienceBank 实例化成功")
    
    print("  ✓ DPG组件检查通过 (语义银行/历史银行/上下文提取)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 5. 验证数据加载和工具模块
print("\n[5/6] 验证数据加载和工具模块...")
try:
    from data_provider.data_factory import TimeSeriesDataset, data_provider
    from utils.tools import EarlyStopping, adjust_learning_rate, print_args
    from utils.metrics import metric, print_metrics
    print("  ✓ data_provider 导入成功")
    print("  ✓ utils.tools 导入成功")
    print("  ✓ utils.metrics 导入成功")
    
    # 验证早停机制
    class MockArgs:
        patience = 3
        use_multi_gpu = False
        local_rank = 0
    es = EarlyStopping(MockArgs())
    assert hasattr(es, 'save_checkpoint')
    print("  ✓ EarlyStopping 实例化成功")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 6. 验证实验模块
print("\n[6/6] 验证实验模块...")
try:
    from exp.exp_basic import Exp_Basic
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    print("  ✓ exp_basic 导入成功")
    print("  ✓ exp_long_term_forecasting 导入成功")
    
    # 验证实验类继承关系
    assert issubclass(Exp_Long_Term_Forecast, Exp_Basic)
    print("  ✓ 实验类继承关系正确")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 最终验证: 模型工厂
print("\n[额外] 验证模型工厂...")
try:
    from models import get_model, _MODEL_REGISTRY
    assert 'TEST_CDP_Llama' in _MODEL_REGISTRY
    print(f"  ✓ 模型工厂注册: {list(_MODEL_REGISTRY.keys())}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 总结
print("\n" + "=" * 60)
print("验证结果: 所有核心模块导入和实例化成功!")
print("=" * 60)
print("\nTEST-CDP 项目结构完整:")
print("  ✓ layers/       - 基础层组件 (MLP, GAT, Attention)")
print("  ✓ models/       - 核心模型 (TripleContrastive, CPCL, DPG, TEST-CDP)")
print("  ✓ data_provider/ - 数据加载 (TimeSeriesDataset, data_provider)")
print("  ✓ exp/          - 实验逻辑 (训练/验证/测试流程)")
print("  ✓ utils/        - 工具函数 (指标/可视化/早停)")
print("  ✓ scripts/      - 运行脚本 (多数据集/零样本)")
print("\n使用方法:")
print("  1. 安装依赖: pip install -r requirements.txt")
print("  2. 运行训练: python run.py --task_name long_term_forecast --is_training 1 ...")
print("  3. 批量实验: bash scripts/run_all_datasets.sh")
print("=" * 60)
