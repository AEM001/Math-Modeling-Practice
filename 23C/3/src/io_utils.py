"""
Data loading and cleaning utilities.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import DATA_PATHS, TARGET_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_single_product_data(file_path=None):
    """
    加载单品级汇总表数据
    """
    if file_path is None:
        file_path = DATA_PATHS['single_product_summary']
    
    logger.info(f"Loading single product data from {file_path}")
    
    try:
        # 读取数据，处理BOM
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 清理列名
        df.columns = df.columns.str.strip()
        
        # 转换日期列
        df['销售日期'] = pd.to_datetime(df['销售日期'])
        
        # 数据类型转换
        numeric_cols = [
            '总销量(千克)', '平均销售单价(元/千克)', '批发价格(元/千克)', 
            '成本加成率', '损耗率(%)'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df)} records from {df['销售日期'].min()} to {df['销售日期'].max()}")
        logger.info(f"Found {df['单品编码'].nunique()} unique products")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading single product data: {e}")
        raise

def load_timeseries_data(file_path=None):
    """
    加载可售品种时间序列数据
    """
    if file_path is None:
        file_path = DATA_PATHS['sellable_timeseries']
    
    logger.info(f"Loading timeseries data from {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        df['销售日期'] = pd.to_datetime(df['销售日期'])
        
        # 数据类型转换
        numeric_cols = [
            '总销量(千克)', '平均销售单价(元/千克)', '批发价格(元/千克)', 
            '成本加成率', '损耗率(%)'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df)} timeseries records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading timeseries data: {e}")
        return pd.DataFrame()

def load_weekly_stats(file_path=None):
    """
    加载品种周统计数据
    """
    if file_path is None:
        file_path = DATA_PATHS['weekly_stats']
    
    logger.info(f"Loading weekly stats from {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        
        logger.info(f"Loaded {len(df)} weekly stats records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading weekly stats: {e}")
        return pd.DataFrame()

def clean_data(df):
    """
    数据清洗
    """
    logger.info("Starting data cleaning...")
    original_len = len(df)
    
    # 1. 去除重复值
    df = df.drop_duplicates(subset=['销售日期', '单品编码'])
    logger.info(f"Removed {original_len - len(df)} duplicate records")
    
    # 2. 处理缺失值
    # 销量和价格不能为负数或空值
    df = df[df['总销量(千克)'] >= 0]
    df = df[df['批发价格(元/千克)'] > 0]
    df = df[df['平均销售单价(元/千克)'] > 0]
    
    # 3. 异常值处理 - 使用IQR方法
    def remove_outliers_iqr(series, factor=1.5):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (series >= lower_bound) & (series <= upper_bound)
    
    # 按产品分别处理异常值
    mask = pd.Series(True, index=df.index)
    
    for product_code in df['单品编码'].unique():
        product_mask = df['单品编码'] == product_code
        product_data = df[product_mask]
        
        if len(product_data) > 5:  # 只对有足够数据的产品处理异常值
            sales_mask = remove_outliers_iqr(product_data['总销量(千克)'])
            price_mask = remove_outliers_iqr(product_data['批发价格(元/千克)'])
            
            mask[product_data.index] = sales_mask & price_mask
    
    df_clean = df[mask].copy()
    logger.info(f"Removed {len(df) - len(df_clean)} outlier records")
    
    # 4. 填补缺失的成本加成率
    df_clean['成本加成率'] = df_clean['成本加成率'].fillna(
        (df_clean['平均销售单价(元/千克)'] - df_clean['批发价格(元/千克)']) / df_clean['批发价格(元/千克)']
    )
    
    # 5. 损耗率缺失值用中位数填补
    df_clean['损耗率(%)'] = df_clean['损耗率(%)'].fillna(df_clean['损耗率(%)'].median())
    
    logger.info(f"Data cleaning completed. Final dataset: {len(df_clean)} records")
    
    return df_clean

def get_sellable_products(target_date=TARGET_DATE):
    """
    获取可售单品列表
    基于时间序列数据确定2023-07-01可售的单品
    """
    timeseries_df = load_timeseries_data()
    
    if timeseries_df.empty:
        logger.warning("No timeseries data found, using all products from historical data")
        historical_df = load_single_product_data()
        # 获取最近一周有销售记录的产品
        recent_date = pd.to_datetime(target_date) - timedelta(days=7)
        recent_products = historical_df[
            historical_df['销售日期'] >= recent_date
        ]['单品编码'].unique()
        return recent_products
    
    # 从时间序列数据获取可售产品
    sellable_products = timeseries_df['单品编码'].unique()
    logger.info(f"Found {len(sellable_products)} sellable products from timeseries data")
    
    return sellable_products

def prepare_training_data(target_date=TARGET_DATE):
    """
    准备训练数据
    """
    # 加载历史数据
    df = load_single_product_data()
    df = clean_data(df)
    
    # 获取可售产品列表
    sellable_products = get_sellable_products(target_date)
    
    # 过滤出可售产品的历史数据
    df = df[df['单品编码'].isin(sellable_products)].copy()
    
    # 按日期排序
    df = df.sort_values(['单品编码', '销售日期'])
    
    logger.info(f"Training data prepared: {len(df)} records for {len(sellable_products)} products")
    
    return df, sellable_products

def save_results(df, file_path):
    """
    保存结果到CSV
    """
    try:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise