"""
Feature Engineering Module (Stage 2)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

class FeatureBuilder:
    def __init__(self):
        self.logger = setup_logger('feature_builder')
        
    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date-based features"""
        self.logger.info("Adding date features...")
        
        df = df.copy()
        df['销售日期'] = pd.to_datetime(df['销售日期'])
        
        # Date features
        df['year'] = df['销售日期'].dt.year
        df['month'] = df['销售日期'].dt.month
        df['day_of_week'] = df['销售日期'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_month'] = df['销售日期'].dt.day
        
        return df
    
    def add_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log-transformed features"""
        self.logger.info("Adding log-transformed features...")
        
        df = df.copy()
        
        # Log transformations (add small constant to avoid log(0))
        df['ln_price'] = np.log(df['正常销售单价(元/千克)'] + 0.001)
        df['ln_quantity'] = np.log(df['正常销量(千克)'] + 0.001)
        df['ln_cost'] = np.log(df['批发价格(元/千克)'] + 0.001)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for each item"""
        self.logger.info("Adding lag features...")
        
        df = df.copy()
        df = df.sort_values(['单品编码', '销售日期'])
        
        # Lag features by item
        lag_features = []
        for item_code in df['单品编码'].unique():
            item_mask = df['单品编码'] == item_code
            item_data = df.loc[item_mask].copy()
            
            # Quantity lags
            item_data['quantity_lag1'] = item_data['正常销量(千克)'].shift(1)
            item_data['quantity_lag7'] = item_data['正常销量(千克)'].shift(7)
            
            # Price lags
            item_data['price_lag1'] = item_data['正常销售单价(元/千克)'].shift(1)
            
            lag_features.append(item_data)
        
        df_with_lags = pd.concat(lag_features, ignore_index=True)
        
        return df_with_lags
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features for each item"""
        self.logger.info("Adding rolling window features...")
        
        df = df.copy()
        df = df.sort_values(['单品编码', '销售日期'])
        
        rolling_features = []
        for item_code in df['单品编码'].unique():
            item_mask = df['单品编码'] == item_code
            item_data = df.loc[item_mask].copy()
            
            # Rolling quantity features
            item_data['quantity_rolling_7d_mean'] = item_data['正常销量(千克)'].rolling(
                window=Config.ROLLING_WINDOW_7D, min_periods=1).mean()
            item_data['quantity_rolling_14d_mean'] = item_data['正常销量(千克)'].rolling(
                window=Config.ROLLING_WINDOW_14D, min_periods=1).mean()
            item_data['quantity_rolling_7d_std'] = item_data['正常销量(千克)'].rolling(
                window=Config.ROLLING_WINDOW_7D, min_periods=2).std()
            
            # Rolling price features
            item_data['price_rolling_7d_mean'] = item_data['正常销售单价(元/千克)'].rolling(
                window=Config.ROLLING_WINDOW_7D, min_periods=1).mean()
            
            rolling_features.append(item_data)
        
        df_with_rolling = pd.concat(rolling_features, ignore_index=True)
        
        return df_with_rolling
    
    def add_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category-level features"""
        self.logger.info("Adding category features...")
        
        df = df.copy()
        
        # Calculate daily category totals
        category_daily = df.groupby(['销售日期', '分类名称']).agg({
            '正常销量(千克)': 'sum',
            '正常销售单价(元/千克)': 'mean',
            '批发价格(元/千克)': 'mean'
        }).reset_index()
        
        category_daily.columns = ['销售日期', '分类名称', 'category_total_quantity', 
                                 'category_avg_price', 'category_avg_cost']
        
        # Merge back to main dataframe
        df = df.merge(category_daily, on=['销售日期', '分类名称'], how='left')
        
        # Calculate item share in category
        df['item_share_in_category'] = df['正常销量(千克)'] / df['category_total_quantity'].clip(lower=0.001)
        
        return df
    
    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets based on time"""
        self.logger.info("Splitting data into train/test sets...")
        
        df = df.sort_values('销售日期')
        
        # Calculate split point
        split_idx = int(len(df) * Config.TRAIN_TEST_SPLIT_RATIO)
        split_date = df.iloc[split_idx]['销售日期']
        
        train_df = df[df['销售日期'] < split_date].copy()
        test_df = df[df['销售日期'] >= split_date].copy()
        
        self.logger.info(f"Train set: {len(train_df)} rows (dates: {train_df['销售日期'].min()} to {train_df['销售日期'].max()})")
        self.logger.info(f"Test set: {len(test_df)} rows (dates: {test_df['销售日期'].min()} to {test_df['销售日期'].max()})")
        
        return train_df, test_df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final feature set"""
        self.logger.info("Selecting final features...")
        
        # Core features for modeling
        feature_columns = [
            # Identifiers
            '销售日期', '单品编码', '单品名称', '分类编码', '分类名称',
            
            # Target variable
            '正常销量(千克)',
            
            # Core price/cost features
            '正常销售单价(元/千克)', '批发价格(元/千克)', '成本加成率',
            'ln_price', 'ln_quantity', 'ln_cost',
            
            # Time features
            'is_weekend', 'month', 'day_of_month',
            
            # Lag features
            'quantity_lag1', 'quantity_lag7', 'price_lag1',
            
            # Rolling features
            'quantity_rolling_7d_mean', 'quantity_rolling_14d_mean', 
            'quantity_rolling_7d_std', 'price_rolling_7d_mean',
            
            # Category features
            'category_total_quantity', 'category_avg_price', 'category_avg_cost',
            'item_share_in_category',
            
            # Additional business features
            'profit_margin', 'is_discount_sale'
        ]
        
        # Keep only available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        
        self.logger.info(f"Selected {len(available_columns)} features")
        
        return df[available_columns]
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        self.logger.info("Handling missing values...")
        
        df = df.copy()
        
        # Fill numeric missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != '正常销量(千克)':  # Don't fill target variable
                df[col] = df[col].fillna(df[col].median())
        
        # Log missing value summary
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.info("Missing values after filling:")
            for col, count in missing_counts.items():
                if count > 0:
                    self.logger.info(f"  {col}: {count}")
        
        return df
    
    def save_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save feature sets to output directory"""
        train_path = Config.get_output_path(Config.TRAIN_FEATURES_FILE)
        test_path = Config.get_output_path(Config.TEST_FEATURES_FILE)
        
        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"Train features saved to: {train_path}")
        self.logger.info(f"Test features saved to: {test_path}")
    
    def run(self, clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete feature engineering pipeline"""
        self.logger.info("Starting feature engineering pipeline...")
        
        # Add all feature types
        df = self.add_date_features(clean_df)
        df = self.add_log_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_category_features(df)
        
        # Select final features
        df = self.select_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Split train/test
        train_df, test_df = self.split_train_test(df)
        
        # Save features
        self.save_features(train_df, test_df)
        
        self.logger.info("Feature engineering pipeline completed successfully!")
        return train_df, test_df

if __name__ == "__main__":
    # Load cleaned data and run feature engineering
    clean_df = pd.read_csv(Config.get_output_path(Config.CLEAN_ITEMS_FILE))
    
    builder = FeatureBuilder()
    train_features, test_features = builder.run(clean_df)