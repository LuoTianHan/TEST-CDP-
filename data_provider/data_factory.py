"""
数据加载模块
实现时间序列数据的加载、预处理和批次生成
支持ETT、Weather、ECL、Traffic、ILI等数据集
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类
    """
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None):
        """
        :param root_path: 数据根目录
        :param data_path: 数据文件名
        :param flag: 'train'/'val'/'test'
        :param size: [seq_len, label_len, pred_len]
        :param features: 'M'(多变量) / 'S'(单变量) / 'MS'(多输入单输出)
        :param target: 目标列名
        :param scale: 是否标准化
        :param timeenc: 时间编码方式
        :param freq: 采样频率
        """
        self.flag = flag
        if size is None:
            self.seq_len = 672
            self.label_len = 576
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        """读取并划分数据集"""
        # 读取CSV文件
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 根据数据集类型确定划分比例
        if 'ETT' in self.data_path:
            # ETT数据集: 12个月数据, 前12个月训练, 后2个月验证, 最后2个月测试
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            if 'm' in self.data_path:  # 15分钟粒度
                border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
                border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif 'Weather' in self.data_path:
            # Weather数据集
            border1s = [0, 36887 - self.seq_len, 42159 - self.seq_len]
            border2s = [36887, 42159, 52696]
        elif 'ECL' in self.data_path:
            # ECL数据集
            border1s = [0, 18317 - self.seq_len, 21920 - self.seq_len]
            border2s = [18317, 21920, 26304]
        elif 'Traffic' in self.data_path:
            # Traffic数据集
            border1s = [0, 12280 - self.seq_len, 14036 - self.seq_len]
            border2s = [12280, 14036, 17544]
        elif 'ILI' in self.data_path:
            # ILI数据集 (周数据)
            border1s = [0, 601 - self.seq_len, 673 - self.seq_len]
            border2s = [601, 673, 966]
        else:
            # 默认按6:2:2划分
            num_train = int(len(df_raw) * 0.7)
            num_val = int(len(df_raw) * 0.9)
            border1s = [0, num_train - self.seq_len, num_val - self.seq_len]
            border2s = [num_train, num_val, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 排除日期列
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 时间戳处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop('date', axis=1).values
        elif self.timeenc == 1:
            # 使用时间特征编码
            data_stamp = self.time_features(df_stamp, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp if self.timeenc else None
    
    def time_features(self, df_stamp, freq='h'):
        """时间特征编码"""
        features = []
        features.append((df_stamp['date'].dt.hour / 23.0 - 0.5) * 2)  # hour [-1, 1]
        features.append((df_stamp['date'].dt.day / 30.0 - 0.5) * 2)  # day
        features.append((df_stamp['date'].dt.month / 12.0 - 0.5) * 2)  # month
        features.append((df_stamp['date'].dt.weekday / 6.0 - 0.5) * 2)  # weekday
        return np.stack(features, axis=0)
    
    def __getitem__(self, index):
        """
        获取单个样本
        :return: (历史序列, 历史时间戳, 目标序列, 目标时间戳)
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        if self.data_stamp is not None:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        else:
            seq_x_mark = np.zeros((self.seq_len, 1))
            seq_y_mark = np.zeros((self.label_len + self.pred_len, 1))
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_x_mark), \
               torch.FloatTensor(seq_y), torch.FloatTensor(seq_y_mark)
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """反标准化"""
        return self.scaler.inverse_transform(data)


class M4Dataset(Dataset):
    """M4数据集（短期预测）"""
    def __init__(self, root_path, seasonal_patterns='Monthly', flag='train', size=None):
        self.seasonal_patterns = seasonal_patterns
        self.flag = flag
        if size is None:
            self.seq_len = 36
            self.pred_len = 18
        else:
            self.seq_len, _, self.pred_len = size
        
        self.root_path = root_path
        self.__read_data__()
    
    def __read_data__(self):
        # M4数据集读取逻辑
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass


def data_provider(args, flag):
    """
    数据提供器工厂函数
    :param args: 配置参数
    :param flag: 'train'/'val'/'test'
    :return: (数据集, 数据加载器)
    """
    if args.data == 'm4':
        dataset = M4Dataset(
            root_path=args.root_path,
            seasonal_patterns=args.seasonal_patterns,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len]
        )
    else:
        dataset = TimeSeriesDataset(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=args.timeenc,
            freq=args.freq
        )
    
    shuffle_flag = (flag == 'train')
    drop_last_flag = getattr(args, 'drop_last', False)
    batch_size = args.batch_size
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last_flag
    )
    
    return dataset, data_loader
