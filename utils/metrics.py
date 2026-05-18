"""
评估指标模块
包含MSE、MAE、RMSE、MAPE、SMAPE等时间序列预测常用指标
"""

import numpy as np
import torch


def RSE(pred, true):
    """相对平方误差 (Relative Squared Error)"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """相关系数 (Correlation Coefficient)"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """平均绝对误差 (Mean Absolute Error)"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """均方误差 (Mean Squared Error)"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """均方根误差 (Root Mean Squared Error)"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """平均绝对百分比误差 (Mean Absolute Percentage Error)"""
    return np.mean(np.abs((pred - true) / (true + 1e-8))) * 100


def SMAPE(pred, true):
    """对称平均绝对百分比误差 (Symmetric MAPE)"""
    return np.mean(2.0 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100


def MSPE(pred, true):
    """均方百分比误差 (Mean Squared Percentage Error)"""
    return np.mean(np.square((pred - true) / (true + 1e-8))) * 100


def metric(pred, true):
    """
    计算所有评估指标
    :param pred: 预测值 [batch, pred_len, n_vars]
    :param true: 真实值 [batch, pred_len, n_vars]
    :return: 指标字典
    """
    # 转换为numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe
    }


def print_metrics(metrics, prefix=''):
    """
    打印评估指标
    :param metrics: 指标字典
    :param prefix: 前缀字符串
    """
    print(f"{prefix}MSE: {metrics['MSE']:.6f} | MAE: {metrics['MAE']:.6f} | "
          f"RMSE: {metrics['RMSE']:.6f} | MAPE: {metrics['MAPE']:.4f}% | "
          f"MSPE: {metrics['MSPE']:.4f}%")
