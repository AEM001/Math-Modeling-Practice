import pandas as pd
import numpy as np
import re

def load_and_prepare_data(path_to_attachment1, path_to_indicators):
    """
    数据加载与预处理模块
    
    功能：
    1. 加载附件1.csv和每组指标.csv
    2. 计算C4烯烃收率
    3. 清洗每组指标数据
    4. 合并数据表
    5. 计算新特征：总质量和装料比
    6. 创建装料方式哑变量
    
    Args:
        path_to_attachment1: 附件1.csv文件路径
        path_to_indicators: 每组指标.csv文件路径
        
    Returns:
        merged_dataframe: 处理完毕的DataFrame，包含所有自变量和目标变量Y
    """
    
    # 1. 加载附件1数据
    print("正在加载附件1数据...")
    attachment1 = pd.read_csv(path_to_attachment1)
    
    # 2. 计算C4烯烃收率
    print("正在计算C4烯烃收率...")
    attachment1['C4烯烃收率'] = (attachment1['乙醇转化率(%)'] * 
                               attachment1['C4烯烃选择性(%)']) / 100
    
    # 3. 加载并清洗每组指标数据
    print("正在加载和清洗每组指标数据...")
    indicators = pd.read_csv(path_to_indicators)
    
    # 清洗数值列，去除单位并转换为数字
    def clean_numeric_column(column):
        """清洗数值列，去除单位并转换为数字"""
        if column.dtype == 'object':
            # 去除单位并转换为数字
            cleaned = column.str.replace(r'[^\d.]', '', regex=True)
            return pd.to_numeric(cleaned, errors='coerce')
        return column
    
    # 清洗各列
    indicators['Co/SiO2用量'] = clean_numeric_column(indicators['Co/SiO2用量'])
    indicators['HAP用量'] = clean_numeric_column(indicators['HAP用量'])
    indicators['Co负载量'] = clean_numeric_column(indicators['Co负载量'])
    indicators['乙醇浓度'] = clean_numeric_column(indicators['乙醇浓度'])
    
    # 4. 合并数据表
    print("正在合并数据表...")
    merged_data = pd.merge(attachment1, indicators, on='催化剂组合编号', how='left')
    
    # 计算新特征：总质量和装料比
    print("正在计算新特征：总质量和装料比...")
    # 确保用量不为0，避免除零错误
    merged_data = merged_data[merged_data['HAP用量'] > 0]
    merged_data['total_mass'] = merged_data['Co/SiO2用量'] + merged_data['HAP用量']
    merged_data['loading_ratio'] = merged_data['Co/SiO2用量'] / merged_data['HAP用量']
    
    # 5. 创建装料方式哑变量
    print("正在创建装料方式哑变量...")
    merged_data['装料方式'] = merged_data['催化剂组合编号'].str[0].map({'A': 0, 'B': 1})
    
    # 6. 选择最终需要的列
    final_columns = [
        '催化剂组合编号', '温度', 'C4烯烃收率',
        'total_mass', 'loading_ratio',
        'Co负载量', '乙醇浓度', '装料方式'
    ]
    
    final_data = merged_data[final_columns].copy()
    
    # 7. 重命名列以便后续使用
    final_data.columns = [
        'catalyst_id', 'T', 'Y',
        'total_mass', 'loading_ratio',
        'C', 'C_e', 'M'
    ]
    
    print(f"数据预处理完成！最终数据集包含 {len(final_data)} 行数据")
    print(f"特征变量：{list(final_data.columns[2:])}")
    print(f"目标变量：Y (C4烯烃收率)")
    
    return final_data

def get_discrete_options(data):
    """
    获取离散变量的所有可能取值
    
    Args:
        data: 预处理后的数据（用于兼容性，实际不使用）
        
    Returns:
        dict: 包含各离散变量的所有可能取值
    """
    # C 和 C_e 已变为连续变量，只保留 M
    discrete_options = {
        'M': [0, 1]
    }
    
    return discrete_options

def get_continuous_bounds(data):
    """
    获取所有连续变量的边界
    
    Args:
        data: 预处理后的数据
        
    Returns:
        dict: 包含各连续变量的边界
    """
    bounds = {
        'T': (data['T'].min(), data['T'].max()),
        'total_mass': (83, 400),
        'loading_ratio': (0.49, 2.03),
        # 新增：将 C 和 C_e 定义为连续变量并给出范围
        'C': (0.5, 5.0),
        'C_e': (0.3, 2.1)
    }
    return bounds

def get_temperature_bounds(data):
    """
    获取温度范围
    
    Args:
        data: 预处理后的数据
        
    Returns:
        tuple: (温度最小值, 温度最大值)
    """
    return (data['T'].min(), data['T'].max()) 