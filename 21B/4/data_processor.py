#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块 - GPR实验设计系统
功能：数据加载、清洗、特征工程、标准化
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ['T', 'total_mass', 'loading_ratio', 'Co_loading', 'ethanol_conc', 'loading_method']
        self.target_name = 'C4_yield'
        
    def load_and_prepare_data(self, attachment1_path: str, indicators_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        加载并预处理数据
        
        Args:
            attachment1_path: 附件1.csv路径
            indicators_path: 每组指标.csv路径
            
        Returns:
            X: 特征矩阵 (标准化后)
            y: 目标变量 (C4烯烃收率)
            data_info: 数据信息字典
        """
        print("🔄 开始数据加载与预处理...")
        
        # 1. 加载附件1数据
        print("  📂 加载附件1数据...")
        attachment1 = pd.read_csv(attachment1_path)
        
        # 2. 计算C4烯烃收率
        print("  🧮 计算C4烯烃收率...")
        attachment1['C4_yield'] = (attachment1['乙醇转化率(%)'] * attachment1['C4烯烃选择性(%)']) / 100
        
        # 3. 加载并清洗每组指标数据
        print("  📂 加载每组指标数据...")
        indicators = pd.read_csv(indicators_path)
        indicators_cleaned = self._clean_indicators_data(indicators)
        
        # 4. 合并数据
        print("  🔗 合并数据表...")
        merged_data = pd.merge(attachment1, indicators_cleaned, on='催化剂组合编号', how='left')
        
        # 5. 特征工程
        print("  ⚙️ 进行特征工程...")
        processed_data = self._feature_engineering(merged_data)
        
        # 6. 数据清洗和验证
        print("  🧹 数据清洗...")
        clean_data = self._clean_data(processed_data)
        
        # 7. 准备训练数据
        print("  📊 准备训练数据...")
        X, y, data_info = self._prepare_training_data(clean_data)
        
        print(f"✅ 数据预处理完成！")
        print(f"   📈 样本数量: {len(X)}")
        print(f"   📋 特征维度: {X.shape[1]}")
        print(f"   🎯 收率范围: {y.min():.4f} - {y.max():.4f}")
        
        return X, y, data_info
    
    def _clean_indicators_data(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """清洗每组指标数据"""
        indicators_clean = indicators.copy()
        
        # 清洗数值列，去除单位
        def extract_numeric(text):
            if pd.isna(text):
                return np.nan
            # 提取数字部分
            numbers = re.findall(r'\d+\.?\d*', str(text))
            return float(numbers[0]) if numbers else np.nan
        
        # 清洗各列
        indicators_clean['Co_SiO2_mass'] = indicators_clean['Co/SiO2用量'].apply(extract_numeric)
        indicators_clean['HAP_mass'] = indicators_clean['HAP用量'].apply(extract_numeric)
        indicators_clean['Co_loading'] = indicators_clean['Co负载量'].apply(extract_numeric)
        indicators_clean['ethanol_conc'] = indicators_clean['乙醇浓度'].apply(extract_numeric)
        
        return indicators_clean
    
    def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        processed = data.copy()
        
        # 计算总质量 (Co/SiO2 和 HAP 质量之和)
        processed['total_mass'] = processed['Co_SiO2_mass'] + processed['HAP_mass']
        
        # 计算装料比 (Co/SiO2 和 HAP 装料比)
        processed['loading_ratio'] = processed['Co_SiO2_mass'] / processed['HAP_mass']
        
        # 创建装料方式哑变量 (A=0, B=1)
        processed['loading_method'] = processed['催化剂组合编号'].str[0].map({'A': 0, 'B': 1})
        
        # 重命名温度列
        processed['T'] = processed['温度']
        
        return processed
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        clean_data = data.copy()
        
        # 移除缺失值
        clean_data = clean_data.dropna(subset=['C4_yield', 'T', 'total_mass', 'loading_ratio', 
                                              'Co_loading', 'ethanol_conc', 'loading_method'])
        
        # 修正：对于小数据集，IQR方法可能过于激进，暂时禁用以保留所有数据点
        # Q1 = clean_data['C4_yield'].quantile(0.25)
        # Q3 = clean_data['C4_yield'].quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + 1.5 * IQR
        
        # # 保留合理范围内的数据
        # clean_data = clean_data[(clean_data['C4_yield'] >= max(0, lower_bound)) & 
        #                        (clean_data['C4_yield'] <= upper_bound)]
        
        return clean_data
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """准备训练数据"""
        # 选择特征列
        X = data[self.feature_names].values
        y = data[self.target_name].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 数据信息
        data_info = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_ranges': self._get_feature_ranges(data),
            'target_stats': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            },
            'scaler_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }
        
        return X_scaled, y, data_info
    
    def _get_feature_ranges(self, data: pd.DataFrame) -> Dict:
        """获取特征范围"""
        ranges = {}
        for feature in self.feature_names:
            if feature in ['Co_loading', 'ethanol_conc', 'loading_method']:
                # 离散变量
                ranges[feature] = {
                    'type': 'discrete',
                    'values': sorted(data[feature].unique().tolist())
                }
            else:
                # 连续变量
                ranges[feature] = {
                    'type': 'continuous',
                    'min': float(data[feature].min()),
                    'max': float(data[feature].max())
                }
        return ranges
    
    def get_variable_bounds(self) -> Dict:
        """获取变量边界（用于候选点生成）"""
        bounds = {
            'T': (250.0, 450.0),
            'total_mass': (20.0, 400.0),
            'loading_ratio': (0.33, 2.03),
            'Co_loading': [0.5, 1.0, 2.0, 5.0],  # 离散值
            'ethanol_conc': [0.3, 0.9, 1.68, 2.1],  # 离散值
            'loading_method': [0, 1]  # 离散值
        }
        return bounds
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """标准化特征（用于新数据）"""
        return self.scaler.transform(X)
    
    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """反标准化特征"""
        return self.scaler.inverse_transform(X_scaled)

def main():
    """测试数据处理模块"""
    processor = DataProcessor()
    
    # 测试数据加载
    try:
        X, y, data_info = processor.load_and_prepare_data('附件1.csv', '每组指标.csv')
        
        print("\n📊 数据信息:")
        print(f"特征名称: {data_info['feature_names']}")
        print(f"样本数量: {data_info['n_samples']}")
        print(f"特征维度: {data_info['n_features']}")
        print(f"目标变量统计: {data_info['target_stats']}")
        
        print("\n📏 特征范围:")
        for feature, range_info in data_info['feature_ranges'].items():
            if range_info['type'] == 'continuous':
                print(f"  {feature}: {range_info['min']:.2f} - {range_info['max']:.2f}")
            else:
                print(f"  {feature}: {range_info['values']}")
        
        print("\n✅ 数据处理模块测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()