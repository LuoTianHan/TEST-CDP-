"""
工具函数模块
包含学习率调整、早停、可视化等工具
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    学习率调整函数
    支持多种衰减策略
    """
    if args.lradj == 'type1':
        # 指数衰减
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # 阶梯衰减
        lr_adjust = {epoch: args.learning_rate * (0.6 ** epoch)}
    elif args.lradj == 'type3':
        # 固定衰减点
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == "cosine":
        # 余弦退火
        lr_adjust = {
            epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    else:
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[学习率调整] Epoch {epoch}: 学习率设置为 {lr:.6f}')
        return lr
    return args.learning_rate


class EarlyStopping:
    """
    早停机制
    当验证损失不再改善时提前终止训练
    """
    def __init__(self, patience=3, verbose=False, delta=0):
        """
        :param patience: 容忍多少个epoch没有改善
        :param verbose: 是否打印信息
        :param delta: 改善的最小阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        
    def __call__(self, val_loss, model, path):
        """
        检查是否需要早停
        :param val_loss: 当前验证损失
        :param model: 模型
        :param path: 保存路径
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[早停计数器] {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """保存模型检查点"""
        if self.verbose:
            print(f'[模型保存] 验证损失改善: {self.val_loss_min:.6f} -> {val_loss:.6f}')
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """支持点号访问的字典"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    可视化预测结果
    :param true: 真实值 [seq_len, n_vars]
    :param preds: 预测值 [pred_len, n_vars]
    :param name: 保存路径
    """
    os.makedirs(os.path.dirname(name) if os.path.dirname(name) else '.', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    if preds is not None:
        plt.plot(preds, label='预测值', linewidth=2)
    plt.plot(true, label='真实值', linewidth=2, alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    plt.title('时间序列预测结果对比', fontsize=14)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def save_results(folder_path, results, setting):
    """
    保存实验结果
    :param folder_path: 保存目录
    :param results: 结果字典
    :param setting: 实验设置字符串
    """
    os.makedirs(folder_path, exist_ok=True)
    
    # 保存为numpy文件
    np.save(os.path.join(folder_path, 'true.npy'), results['true'])
    np.save(os.path.join(folder_path, 'pred.npy'), results['pred'])
    
    # 保存指标
    with open(os.path.join(folder_path, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write(f'实验设置: {setting}\n')
        for key, value in results.items():
            if key not in ['true', 'pred']:
                f.write(f'{key}: {value:.6f}\n')


def print_args(args):
    """打印参数配置"""
    print("=" * 50)
    print("TEST-CDP 参数配置")
    print("=" * 50)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print("=" * 50)
