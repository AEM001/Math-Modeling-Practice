"""
Data Audit and Cleaning Module (Stage 1)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

class DataCleaner:
    def __init__(self):
        self.logger = setup_logger('data_cleaner')
        self.cleaning_stats = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load item-level and category-level data"""
        self.logger.info("Loading data files...")
        
        # Load item-level data
        item_df = pd.read_csv(Config.get_data_path(Config.ITEM_DAILY_FILE), encoding='utf-8-sig')
        category_df = pd.read_csv(Config.get_data_path(Config.CATEGORY_DAILY_FILE), encoding='utf-8-sig')
        
        self.logger.info(f"Loaded item data: {item_df.shape[0]} rows, {item_df.shape[1]} columns")
        self.logger.info(f"Loaded category data: {category_df.shape[0]} rows, {category_df.shape[1]} columns")
        
        return item_df, category_df
    
    def detect_outliers_mad(self, series: pd.Series, threshold: float = 3) -> np.ndarray:
        """Detect outliers using MAD (Median Absolute Deviation)"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return np.zeros(len(series), dtype=bool)
        
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def clean_item_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean item-level data according to business rules"""
        self.logger.info("Starting item-level data cleaning...")
        original_count = len(df)
        
        # Convert date column
        df['销售日期'] = pd.to_datetime(df['销售日期'])
        
        # Initialize cleaning stats
        self.cleaning_stats = {
            'original_count': original_count,
            'zero_price_removed': 0,
            'zero_quantity_removed': 0,
            'negative_markup_removed': 0,
            'extreme_price_outliers_removed': 0,
            'discount_sales_filtered': 0
        }
        
        # Rule 1: Remove zero prices
        zero_price_mask = (df['正常销售单价(元/千克)'] <= Config.MIN_NORMAL_PRICE)
        zero_price_count = zero_price_mask.sum()
        df = df[~zero_price_mask]
        self.cleaning_stats['zero_price_removed'] = zero_price_count
        
        # Rule 2: Remove zero quantities  
        zero_qty_mask = (df['正常销量(千克)'] <= Config.MIN_NORMAL_QUANTITY)
        zero_qty_count = zero_qty_mask.sum()
        df = df[~zero_qty_mask]
        self.cleaning_stats['zero_quantity_removed'] = zero_qty_count
        
        # Rule 3: Remove negative or extreme markup rates
        markup_mask = ((df['成本加成率'] < Config.MIN_MARKUP_RATE) | 
                      (df['成本加成率'] > Config.MAX_MARKUP_RATE))
        markup_count = markup_mask.sum()
        df = df[~markup_mask]
        self.cleaning_stats['negative_markup_removed'] = markup_count
        
        # Rule 4: Remove extreme price outliers by category
        outlier_mask = np.zeros(len(df), dtype=bool)
        for category in df['分类名称'].unique():
            category_mask = df['分类名称'] == category
            category_prices = df.loc[category_mask, '正常销售单价(元/千克)']
            
            if len(category_prices) > 10:  # Only apply to categories with sufficient data
                outliers = self.detect_outliers_mad(category_prices, Config.OUTLIER_THRESHOLD_MAD)
                outlier_mask[category_mask] = outliers
        
        outlier_count = outlier_mask.sum()
        df = df[~outlier_mask]
        self.cleaning_stats['extreme_price_outliers_removed'] = outlier_count
        
        # Rule 5: Mark discount sales (keep for now but flag)
        df['is_discount_sale'] = df['打折销量(千克)'] > 0
        discount_count = df['is_discount_sale'].sum()
        self.cleaning_stats['discount_sales_filtered'] = discount_count
        
        # Add additional derived fields
        df['total_revenue'] = df['正常销量(千克)'] * df['正常销售单价(元/千克)']
        df['total_cost'] = df['正常销量(千克)'] * df['批发价格(元/千克)']
        df['profit'] = df['total_revenue'] - df['total_cost']
        df['profit_margin'] = df['profit'] / df['total_revenue'].clip(lower=0.001)
        
        final_count = len(df)
        self.cleaning_stats['final_count'] = final_count
        self.cleaning_stats['total_removed'] = original_count - final_count
        self.cleaning_stats['removal_rate'] = self.cleaning_stats['total_removed'] / original_count
        
        self.logger.info(f"Data cleaning completed. Removed {self.cleaning_stats['total_removed']} rows "
                        f"({self.cleaning_stats['removal_rate']:.2%})")
        
        return df
    
    def log_cleaning_stats(self):
        """Log detailed cleaning statistics"""
        self.logger.info("=== Data Cleaning Statistics ===")
        self.logger.info(f"Original records: {self.cleaning_stats['original_count']:,}")
        self.logger.info(f"Zero price removed: {self.cleaning_stats['zero_price_removed']:,}")
        self.logger.info(f"Zero quantity removed: {self.cleaning_stats['zero_quantity_removed']:,}")
        self.logger.info(f"Negative markup removed: {self.cleaning_stats['negative_markup_removed']:,}")
        self.logger.info(f"Price outliers removed: {self.cleaning_stats['extreme_price_outliers_removed']:,}")
        self.logger.info(f"Discount sales flagged: {self.cleaning_stats['discount_sales_filtered']:,}")
        self.logger.info(f"Final records: {self.cleaning_stats['final_count']:,}")
        self.logger.info(f"Total removal rate: {self.cleaning_stats['removal_rate']:.2%}")
    
    def save_cleaned_data(self, df: pd.DataFrame):
        """Save cleaned data to output directory"""
        output_path = Config.get_output_path(Config.CLEAN_ITEMS_FILE)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Cleaned data saved to: {output_path}")
    
    def run(self) -> pd.DataFrame:
        """Run the complete data cleaning pipeline"""
        self.logger.info("Starting data cleaning pipeline...")
        
        # Load data
        item_df, category_df = self.load_data()
        
        # Clean item data
        clean_df = self.clean_item_data(item_df)
        
        # Log statistics
        self.log_cleaning_stats()
        
        # Save cleaned data
        self.save_cleaned_data(clean_df)
        
        self.logger.info("Data cleaning pipeline completed successfully!")
        return clean_df

if __name__ == "__main__":
    cleaner = DataCleaner()
    clean_data = cleaner.run()