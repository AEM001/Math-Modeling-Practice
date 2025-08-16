# -*- coding: utf-8 -*-
"""
精简版特征工程模块
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """精简版特征工程器"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化特征工程器"""
        self.config = self.load_config(config_path)
        self.data_paths = self.config['data_paths']
        self.output_paths = self.config['output_paths']
        self.fe_config = self.config['feature_engineering']
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_clean_data(self):
        """加载清洗后的数据"""
        print("加载清洗后的数据...")
        
        # 合并训练和测试数据进行特征工程
        train_data = pd.read_csv(self.data_paths['train_data'])
        test_data = pd.read_csv(self.data_paths['test_data'])
        
        full_data = pd.concat([train_data, test_data], ignore_index=True)
        full_data['销售日期'] = pd.to_datetime(full_data['销售日期'])
        
        # 基础对数变量
        full_data['ln_price'] = np.log(full_data['正常销售单价(元/千克)'])
        full_data['ln_quantity'] = np.log(full_data['正常销量(千克)'])
        full_data['ln_wholesale'] = np.log(full_data['批发价格(元/千克)'])
        
        print(f"总数据量: {len(full_data):,} 条记录")
        return full_data
    
    def create_time_features(self, data):
        """创建时间特征"""
        print("创建时间特征...")
        
        # 基础时间特征
        data['weekday'] = data['销售日期'].dt.dayofweek
        data['is_weekend'] = (data['weekday'] >= 5).astype(int)
        data['month'] = data['销售日期'].dt.month
        
        # 周几哑变量
        weekday_dummies = pd.get_dummies(data['weekday'], prefix='weekday')
        data = pd.concat([data, weekday_dummies], axis=1)
        
        # 时间趋势
        data['days_since_start'] = (data['销售日期'] - data['销售日期'].min()).dt.days
        data['time_trend'] = data['days_since_start'] / data['days_since_start'].max()
        
        return data
    
    def create_price_features(self, data):
        """创建价格特征"""
        print("创建价格特征...")
        
        # 价格相对特征
        data['markup_ratio'] = data['正常销售单价(元/千克)'] / data['批发价格(元/千克)']
        data['ln_markup_ratio'] = np.log(data['markup_ratio'])
        
        # 品类日均价
        category_daily_price = data.groupby(['分类名称', '销售日期'])['正常销售单价(元/千克)'].mean().reset_index()
        category_daily_price.columns = ['分类名称', '销售日期', 'category_avg_price']
        data = data.merge(category_daily_price, on=['分类名称', '销售日期'], how='left')
        
        data['relative_price_to_category'] = data['正常销售单价(元/千克)'] / data['category_avg_price']
        data['ln_relative_price'] = np.log(data['relative_price_to_category'])
        
        return data
    
    def create_lag_features(self, data):
        """创建滞后特征"""
        print("创建滞后特征...")
        
        lag_features = []
        lag_periods = self.fe_config['lag_periods']
        
        for item_code in data['单品编码'].unique():
            item_data = data[data['单品编码'] == item_code].sort_values('销售日期').copy()
            
            if len(item_data) < 3:
                continue
                
            # 创建滞后特征
            for lag in lag_periods:
                item_data[f'price_lag{lag}'] = item_data['正常销售单价(元/千克)'].shift(lag)
                item_data[f'quantity_lag{lag}'] = item_data['正常销量(千克)'].shift(lag)
                item_data[f'ln_price_lag{lag}'] = item_data['ln_price'].shift(lag)
                
                # 价格变化率
                if lag == 1:
                    item_data['price_change_1d'] = (
                        (item_data['正常销售单价(元/千克)'] - item_data[f'price_lag{lag}']) / 
                        item_data[f'price_lag{lag}']
                    )
            
            lag_features.append(item_data)
        
        if lag_features:
            return pd.concat(lag_features, ignore_index=True)
        else:
            return data
    
    def create_rolling_features(self, data):
        """创建滚动特征"""
        print("创建滚动特征...")
        
        rolling_features = []
        rolling_windows = self.fe_config['rolling_windows']
        
        for item_code in data['单品编码'].unique():
            item_data = data[data['单品编码'] == item_code].sort_values('销售日期').copy()
            
            if len(item_data) < 5:
                continue
                
            # 创建滚动特征
            for window in rolling_windows:
                item_data[f'price_rolling_{window}d_mean'] = (
                    item_data['正常销售单价(元/千克)'].rolling(window=window, min_periods=1).mean()
                )
                item_data[f'quantity_rolling_{window}d_mean'] = (
                    item_data['正常销量(千克)'].rolling(window=window, min_periods=1).mean()
                )
                
                if window == 7:  # 只为7天窗口创建波动率
                    price_std = item_data['正常销售单价(元/千克)'].rolling(window=window, min_periods=1).std()
                    item_data['price_volatility_7d'] = price_std / item_data[f'price_rolling_{window}d_mean']
            
            rolling_features.append(item_data)
        
        if rolling_features:
            return pd.concat(rolling_features, ignore_index=True)
        else:
            return data
    
    def create_interaction_features(self, data):
        """创建交互特征"""
        if not self.fe_config['enable_interaction_features']:
            return data
            
        print("创建交互特征...")
        
        # 价格与时间的交互
        data['ln_price_x_time_trend'] = data['ln_price'] * data['time_trend']
        data['ln_price_x_weekend'] = data['ln_price'] * data['is_weekend']
        
        return data
    
    def select_modeling_features(self, data):
        """选择建模特征"""
        print("选择建模特征...")
        
        # 基础特征
        base_features = ['ln_price', 'ln_wholesale', 'ln_markup_ratio']
        
        # 时间特征
        time_features = ['time_trend', 'is_weekend'] + [col for col in data.columns if col.startswith('weekday_')]
        
        # 价格特征
        price_features = ['ln_relative_price']
        
        # 滞后特征
        lag_features = [col for col in data.columns if 'lag' in col or 'change' in col]
        
        # 滚动特征
        rolling_features = [col for col in data.columns if 'rolling' in col or 'volatility' in col]
        
        # 交互特征
        interaction_features = [col for col in data.columns if '_x_' in col]
        
        # 促销特征
        promotion_features = ['is_promotion'] if 'is_promotion' in data.columns else []
        
        # 合并所有特征
        all_features = (base_features + time_features + price_features + 
                       lag_features + rolling_features + interaction_features + promotion_features)
        
        # 过滤存在的特征
        available_features = [f for f in all_features if f in data.columns]
        
        # 目标变量和标识符
        target_and_ids = ['ln_quantity', '单品编码', '单品名称', '分类名称', '销售日期']
        
        # 创建特征数据集
        feature_columns = target_and_ids + available_features
        feature_data = data[feature_columns].copy()
        
        # 处理缺失值
        numeric_features = feature_data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature_data[feature].isnull().sum() > 0:
                median_value = feature_data[feature].median()
                feature_data[feature].fillna(median_value, inplace=True)
        
        print(f"选择了 {len(available_features)} 个建模特征")
        return feature_data, available_features
    
    def split_and_save_features(self, feature_data):
        """分割并保存特征数据"""
        print("分割并保存特征数据...")
        
        # 按时间分割（与原始清洗时保持一致）
        split_ratio = self.config['data_cleaning']['train_split_ratio']
        feature_data_sorted = feature_data.sort_values('销售日期')
        split_date = feature_data_sorted['销售日期'].quantile(split_ratio)
        
        train_features = feature_data_sorted[feature_data_sorted['销售日期'] <= split_date]
        test_features = feature_data_sorted[feature_data_sorted['销售日期'] > split_date]
        
        # 确保目录存在
        os.makedirs(self.data_paths['processed_data_dir'], exist_ok=True)
        
        # 保存文件
        train_path = self.data_paths['train_features']
        test_path = self.data_paths['test_features']
        
        train_features.to_csv(train_path, index=False, encoding='utf-8')
        test_features.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"训练特征: {len(train_features):,} 条记录 -> {train_path}")
        print(f"测试特征: {len(test_features):,} 条记录 -> {test_path}")
        
        return train_features, test_features
    
    def generate_report(self, feature_data, available_features):
        """生成特征工程报告"""
        report_content = [
            "# 特征工程报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 数据概览",
            f"- 总样本数: {len(feature_data):,}",
            f"- 特征数量: {len(available_features)}",
            f"- 单品数量: {feature_data['单品编码'].nunique()}",
            f"- 品类数量: {feature_data['分类名称'].nunique()}",
            "",
            "## 特征类别",
            f"- 基础特征: 3",
            f"- 时间特征: {len([f for f in available_features if 'weekday' in f or 'time' in f or 'weekend' in f])}",
            f"- 价格特征: {len([f for f in available_features if 'price' in f and 'lag' not in f and 'rolling' not in f])}",
            f"- 滞后特征: {len([f for f in available_features if 'lag' in f or 'change' in f])}",
            f"- 滚动特征: {len([f for f in available_features if 'rolling' in f or 'volatility' in f])}",
            f"- 交互特征: {len([f for f in available_features if '_x_' in f])}",
        ]
        
        # 保存报告
        os.makedirs(self.output_paths['reports_dir'], exist_ok=True)
        report_path = os.path.join(self.output_paths['reports_dir'], 'feature_engineering_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"特征工程报告已保存: {report_path}")
    
    def run_feature_engineering(self):
        """运行完整的特征工程流程"""
        print("=== 开始特征工程 ===")
        
        # 加载数据
        data = self.load_clean_data()
        
        # 创建特征
        data = self.create_time_features(data)
        data = self.create_price_features(data)
        data = self.create_lag_features(data)
        data = self.create_rolling_features(data)
        data = self.create_interaction_features(data)
        
        # 选择特征
        feature_data, available_features = self.select_modeling_features(data)
        
        # 分割保存
        self.split_and_save_features(feature_data)
        
        # 生成报告
        self.generate_report(feature_data, available_features)
        
        print("=== 特征工程完成 ===\n")
        return True

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run_feature_engineering()
