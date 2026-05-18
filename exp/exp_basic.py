"""
实验基础类
定义实验的基本接口和通用功能
"""

import torch
import os


class Exp_Basic:
    """
    实验基础类
    所有具体实验类（长期预测、短期预测等）的基类
    """
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TEST_CDP_Llama': self._get_model_class(),
        }
        self.device = self._acquire_device()
        self.model = self._build_model()
    
    def _get_model_class(self):
        """获取模型类"""
        from models.TEST_CDP_Llama import Model
        return Model
    
    def _acquire_device(self):
        """获取计算设备"""
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                device = torch.device(f'cuda:{self.args.local_rank}')
            else:
                device = torch.device(f'cuda:{self.args.gpu}')
            print(f"[设备] 使用GPU: {device}")
        else:
            device = torch.device('cpu')
            print("[设备] 使用CPU")
        return device
    
    def _build_model(self):
        """构建模型"""
        model = self.model_dict[self.args.model](self.args)
        model.to(self.device)
        return model
    
    def _get_data(self, flag):
        """获取数据"""
        pass
    
    def _select_optimizer(self):
        """选择优化器"""
        pass
    
    def _select_criterion(self):
        """选择损失函数"""
        pass
    
    def train(self, setting):
        """训练入口"""
        pass
    
    def test(self, setting, test=0):
        """测试入口"""
        pass
