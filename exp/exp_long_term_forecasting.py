"""
长期预测实验类
实现TEST-CDP模型的训练、验证和测试流程
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_results, print_args
from utils.metrics import metric as calc_metric, print_metrics

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    长期时间序列预测实验
    支持ETT、Weather、ECL、Traffic、ILI等数据集
    """
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
    
    def _build_model(self):
        """
        构建TEST-CDP模型
        只训练轻量级模块，冻结LLM骨干
        """
        self.args.device = self.device
        model = super()._build_model()
        
        # 统计可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"[模型参数] 总计: {total_params:,} | 可训练: {trainable_params:,} | 冻结: {frozen_params:,}")
        print(f"[参数比例] 可训练占比: {trainable_params/total_params*100:.2f}%")
        
        return model
    
    def _get_data(self, flag):
        """获取数据集和数据加载器"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        """
        选择优化器
        使用AdamW优化器，仅优化可训练参数
        """
        # 收集可训练参数
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"[可训练参数] {name}: {param.shape}")
        
        # AdamW优化器
        model_optim = optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95)
        )
        return model_optim
    
    def _select_criterion(self):
        """选择损失函数"""
        return nn.MSELoss()
    
    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        """
        验证/测试函数
        :param vali_data: 验证数据集
        :param vali_loader: 验证数据加载器
        :param criterion: 损失函数
        :param is_test: 是否为测试模式
        :return: 平均损失
        """
        total_loss = []
        total_count = 0
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # 输入前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, aux_loss = self.model(batch_x)
                else:
                    outputs, aux_loss = self.model(batch_x)
                
                # 获取预测部分（去除label_len部分）
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                # 计算损失
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                
                total_loss.append(loss.item() * batch_x.size(0))
                total_count += batch_x.size(0)
                
                # 测试时保存预测结果
                if is_test and i < 5:
                    pass  # 可选：保存可视化结果
        
        avg_loss = np.sum(total_loss) / total_count
        return avg_loss
    
    def train(self, setting):
        """
        训练函数
        :param setting: 实验设置字符串
        :return: 训练后的模型
        """
        print(f"\n{'='*60}")
        print(f"开始训练: {setting}")
        print(f"{'='*60}\n")
        
        # 获取数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # 设置检查点路径
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        
        # 优化器和损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # 学习率调度器（余弦退火）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optim, 
            T_max=self.args.train_epochs,
            eta_min=self.args.learning_rate * 0.01
        )
        
        # 早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        
        # 训练循环
        for epoch in range(self.args.train_epochs):
            print(f"\n[Epoch {epoch+1}/{self.args.train_epochs}] {'-'*50}")
            
            self.model.train()
            epoch_loss = 0.0
            epoch_aux_loss = 0.0
            num_batches = 0
            
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                model_optim.zero_grad()
                
                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, aux_loss = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        total_loss = loss + aux_loss
                else:
                    outputs, aux_loss = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    total_loss = loss + aux_loss
                
                # 反向传播
                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()
                
                epoch_loss += loss.item()
                epoch_aux_loss += aux_loss.item()
                num_batches += 1
                
                if i % 100 == 0:
                    print(f"  Batch {i}/{len(train_loader)} | 主损失: {loss.item():.6f} | 辅助损失: {aux_loss.item():.6f}")
            
            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss / num_batches
            avg_aux_loss = epoch_aux_loss / num_batches
            print(f"[Epoch {epoch+1} 完成] 平均主损失: {avg_epoch_loss:.6f} | 平均辅助损失: {avg_aux_loss:.6f}")
            
            # 学习率更新
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"[学习率] 当前学习率: {current_lr:.6f}")
            
            # 验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(f"[验证] 验证损失: {vali_loss:.6f}")
            
            # 早停检查
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("[早停] 触发早停机制，训练终止")
                break
        
        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"[加载模型] 加载最佳模型: {best_model_path}")
        
        return self.model
    
    def test(self, setting, test=0):
        """
        测试函数
        :param setting: 实验设置字符串
        :param test: 是否加载预训练模型
        :return: 测试结果指标
        """
        print(f"\n{'='*60}")
        print(f"开始测试: {setting}")
        print(f"{'='*60}\n")
        
        # 获取测试数据
        test_data, test_loader = self._get_data(flag='test')
        
        # 加载最佳模型
        if test:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
                print(f"[加载模型] 加载模型: {best_model_path}")
        
        self.model.eval()
        
        # 收集所有预测和真实值
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x)
                
                # 获取预测部分
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                
                # 可视化部分样本
                if i < 5 and self.args.visualize:
                    folder_path = os.path.join('./test_results', setting)
                    os.makedirs(folder_path, exist_ok=True)
                    
                    # 可视化第一个变量的预测结果
                    var_idx = 0
                    true_plot = np.concatenate([batch_x[0, :, var_idx].detach().cpu().numpy(), 
                                               true[0, :, var_idx]], axis=0)
                    pred_plot = np.concatenate([batch_x[0, :, var_idx].detach().cpu().numpy(), 
                                               pred[0, :, var_idx]], axis=0)
                    
                    visual(true_plot, pred_plot, 
                          os.path.join(folder_path, f'var{var_idx}_sample{i}.pdf'))
        
        # 合并所有批次结果
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 计算评估指标
        results = calc_metric(preds, trues)
        mse = results['MSE']
        mae = results['MAE']
        rmse = results['RMSE']
        mape = results['MAPE']
        mspe = results['MSPE']
        
        print(f"\n[测试结果] 数据集: {self.args.data}")
        print(f"{'='*40}")
        print_metrics({'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MSPE': mspe})
        print(f"{'='*40}\n")
        
        # 保存结果
        folder_path = os.path.join('./results', setting)
        save_results(folder_path, {
            'true': trues,
            'pred': preds,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MSPE': mspe
        }, setting)
        
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MSPE': mspe}
