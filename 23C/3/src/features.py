"""
Feature engineering for sales and pricing prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import FORECAST_PARAMS, TARGET_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_time_features(df):
    """
    添加时间相关特征
    """
    logger.info("Adding time features...")
    
    df = df.copy()
    
    # 确保日期列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['销售日期']):
        df['销售日期'] = pd.to_datetime(df['销售日期'])
    
    # 基础时间特征
    df['day_of_week'] = df['销售日期'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
    df['month'] = df['销售日期'].dt.month
    df['day'] = df['销售日期'].dt.day
    df['week_num'] = df['销售日期'].dt.isocalendar().week
    
    # 相对于目标日期的天数
    target_dt = pd.to_datetime(TARGET_DATE)
    df['days_to_target'] = (target_dt - df['销售日期']).dt.days
    
    # 月内相对位置
    df['day_of_month'] = df['销售日期'].dt.day
    df['days_in_month'] = df['销售日期'].dt.daysinmonth
    df['month_progress'] = df['day_of_month'] / df['days_in_month']
    
    logger.info("Time features added successfully")
    return df

def add_moving_average_features(df, windows=None):
    """
    添加移动平均特征
    """
    if windows is None:
        windows = FORECAST_PARAMS['moving_window_days']
    
    logger.info(f"Adding moving average features with windows: {windows}")
    
    df = df.copy()
    df = df.sort_values(['单品编码', '销售日期'])
    
    # 按单品分组计算移动特征
    features_list = []
    
    for product_code in df['单品编码'].unique():
        product_df = df[df['单品编码'] == product_code].copy()
        
        # 计算移动窗口特征
        for window in windows:
            # 销量移动特征
            product_df[f'sales_ma_{window}d'] = product_df['总销量(千克)'].rolling(
                window=window, min_periods=1
            ).mean()
            
            product_df[f'sales_std_{window}d'] = product_df['总销量(千克)'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)
            
            product_df[f'sales_max_{window}d'] = product_df['总销量(千克)'].rolling(
                window=window, min_periods=1
            ).max()
            
            product_df[f'sales_min_{window}d'] = product_df['总销量(千克)'].rolling(
                window=window, min_periods=1
            ).min()
            
            # 价格移动特征
            product_df[f'price_ma_{window}d'] = product_df['平均销售单价(元/千克)'].rolling(
                window=window, min_periods=1
            ).mean()
            
            product_df[f'wholesale_price_ma_{window}d'] = product_df['批发价格(元/千克)'].rolling(
                window=window, min_periods=1
            ).mean()
            
            # 加成率移动特征
            product_df[f'markup_ma_{window}d'] = product_df['成本加成率'].rolling(
                window=window, min_periods=1
            ).mean()
            
            # 增长率特征
            if window >= 2:
                product_df[f'sales_growth_{window}d'] = (
                    product_df[f'sales_ma_{window}d'] / 
                    product_df[f'sales_ma_{window}d'].shift(window) - 1
                ).fillna(0)
        
        features_list.append(product_df)
    
    df_with_features = pd.concat(features_list, ignore_index=True)
    
    logger.info("Moving average features added successfully")
    return df_with_features

def add_lag_features(df, lags=None):
    """
    添加滞后特征
    """
    if lags is None:
        lags = [1, 2, 3, 7]
    
    logger.info(f"Adding lag features with lags: {lags}")
    
    df = df.copy()
    df = df.sort_values(['单品编码', '销售日期'])
    
    features_list = []
    
    for product_code in df['单品编码'].unique():
        product_df = df[df['单品编码'] == product_code].copy()
        
        for lag in lags:
            # 销量滞后
            product_df[f'sales_lag_{lag}d'] = product_df['总销量(千克)'].shift(lag)
            
            # 价格滞后
            product_df[f'price_lag_{lag}d'] = product_df['平均销售单价(元/千克)'].shift(lag)
            
            # 批发价滞后
            product_df[f'wholesale_price_lag_{lag}d'] = product_df['批发价格(元/千克)'].shift(lag)
        
        features_list.append(product_df)
    
    df_with_lags = pd.concat(features_list, ignore_index=True)
    
    # 填充缺失值
    lag_columns = [col for col in df_with_lags.columns if 'lag_' in col]
    df_with_lags[lag_columns] = df_with_lags[lag_columns].fillna(0)
    
    logger.info("Lag features added successfully")
    return df_with_lags

def add_weekend_features(df):
    """
    添加周末相关特征
    """
    logger.info("Adding weekend-specific features...")
    
    df = df.copy()
    
    # 按单品和周几分组的历史平均
    df_weekend_stats = df.groupby(['单品编码', 'day_of_week'])['总销量(千克)'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    df_weekend_stats.columns = ['单品编码', 'day_of_week', 'dow_sales_mean', 'dow_sales_std', 'dow_count']
    df_weekend_stats['dow_sales_std'] = df_weekend_stats['dow_sales_std'].fillna(0)
    
    # 合并周几统计
    df = df.merge(df_weekend_stats, on=['单品编码', 'day_of_week'], how='left')
    
    # 周六效应特征
    if FORECAST_PARAMS['use_weekend_correction']:
        df['saturday_factor'] = np.where(
            df['is_saturday'] == 1, 
            FORECAST_PARAMS['weekend_factor'], 
            1.0
        )
    else:
        df['saturday_factor'] = 1.0
    
    logger.info("Weekend features added successfully")
    return df

def add_category_features(df):
    """
    添加品类相关特征
    """
    logger.info("Adding category features...")
    
    df = df.copy()
    
    # 品类统计特征
    category_stats = df.groupby('分类编码')['总销量(千克)'].agg([
        'mean', 'std', 'median'
    ]).reset_index()
    category_stats.columns = ['分类编码', 'category_sales_mean', 'category_sales_std', 'category_sales_median']
    category_stats['category_sales_std'] = category_stats['category_sales_std'].fillna(0)
    
    df = df.merge(category_stats, on='分类编码', how='left')
    
    # 单品在品类中的相对位置
    df['sales_vs_category_mean'] = df['总销量(千克)'] / (df['category_sales_mean'] + 1e-8)
    
    logger.info("Category features added successfully")
    return df

def add_price_elasticity_features(df):
    """
    添加价格弹性相关特征
    """
    logger.info("Adding price elasticity features...")
    
    df = df.copy()
    
    # 价格变化特征
    df = df.sort_values(['单品编码', '销售日期'])
    
    features_list = []
    
    for product_code in df['单品编码'].unique():
        product_df = df[df['单品编码'] == product_code].copy()
        
        # 价格变化率
        product_df['price_change'] = product_df['平均销售单价(元/千克)'].pct_change().fillna(0)
        product_df['wholesale_price_change'] = product_df['批发价格(元/千克)'].pct_change().fillna(0)
        
        # 价格与销量的关系特征
        product_df['price_sales_ratio'] = (
            product_df['平均销售单价(元/千克)'] / (product_df['总销量(千克)'] + 1e-8)
        )
        
        # 加成率变化
        product_df['markup_change'] = product_df['成本加成率'].diff().fillna(0)
        
        features_list.append(product_df)
    
    df_with_elasticity = pd.concat(features_list, ignore_index=True)
    
    logger.info("Price elasticity features added successfully")
    return df_with_elasticity

def create_prediction_features(df, target_date=TARGET_DATE):
    """
    为预测目标日期创建特征
    """
    logger.info(f"Creating features for prediction date: {target_date}")
    
    target_dt = pd.to_datetime(target_date)
    
    # 获取最新数据的日期
    latest_date = df['销售日期'].max()
    logger.info(f"Latest data date: {latest_date}")
    
    # 为每个产品创建预测样本
    products = df['单品编码'].unique()
    prediction_rows = []
    
    for product_code in products:
        product_df = df[df['单品编码'] == product_code]
        
        # 获取产品基本信息
        latest_product_info = product_df.iloc[-1].copy()
        
        # 创建预测行
        pred_row = latest_product_info.copy()
        pred_row['销售日期'] = target_dt
        
        # 更新时间特征
        pred_row['day_of_week'] = target_dt.dayofweek
        pred_row['is_weekend'] = int(target_dt.dayofweek >= 5)
        pred_row['is_saturday'] = int(target_dt.dayofweek == 5)
        pred_row['month'] = target_dt.month
        pred_row['day'] = target_dt.day
        pred_row['week_num'] = target_dt.isocalendar().week
        pred_row['days_to_target'] = 0
        
        prediction_rows.append(pred_row)
    
    # 创建预测DataFrame
    pred_df = pd.DataFrame(prediction_rows)
    
    # 重新计算移动特征（基于历史数据）
    combined_df = pd.concat([df, pred_df], ignore_index=True)
    combined_df = combined_df.sort_values(['单品编码', '销售日期'])
    
    # 重新计算所有特征以确保一致性
    combined_df = build_feature_matrix(combined_df)
    
    # 提取预测样本
    pred_features = combined_df[combined_df['销售日期'] == target_dt].copy()
    
    logger.info(f"Created prediction features for {len(pred_features)} products")
    
    return pred_features

def build_feature_matrix(df):
    """
    构建最终的特征矩阵
    """
    logger.info("Building final feature matrix...")
    
    # 依次添加各种特征
    df = add_time_features(df)
    df = add_moving_average_features(df)
    df = add_lag_features(df)
    df = add_weekend_features(df)
    df = add_category_features(df)
    df = add_price_elasticity_features(df)
    
    # 填充剩余缺失值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    logger.info(f"Feature matrix built with shape: {df.shape}")
    logger.info(f"Features: {numeric_columns.tolist()}")
    
    return df