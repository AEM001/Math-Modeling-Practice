"""
Configuration file for the pricing and replenishment project
"""

import os
from datetime import datetime, date

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Data files
    ITEM_DAILY_FILE = '单品级每日汇总表.csv'
    CATEGORY_DAILY_FILE = '品类级每日汇总表.csv'
    
    # Time split parameters
    TRAIN_TEST_SPLIT_RATIO = 0.7
    ROLLING_WINDOW_7D = 7
    ROLLING_WINDOW_14D = 14
    
    # Data cleaning parameters
    OUTLIER_THRESHOLD_MAD = 3
    MIN_NORMAL_PRICE = 0.1
    MIN_NORMAL_QUANTITY = 0.001
    MAX_MARKUP_RATE = 5.0
    MIN_MARKUP_RATE = -0.5
    
    # Model parameters
    RF_N_ESTIMATORS = 100  # Reduced for faster execution
    RF_MAX_DEPTH = None
    RF_RANDOM_STATE = 42
    
    GB_N_ESTIMATORS = 200  # Reduced for faster execution
    GB_LEARNING_RATE = 0.08
    GB_MAX_DEPTH = 6
    GB_RANDOM_STATE = 42
    
    # Cross-validation parameters
    CV_MAX_SPLITS = 4  # Reduced for faster execution
    CV_MIN_TRAIN_SIZE = 30
    
    # Pricing and replenishment parameters
    DEFAULT_MARKUP = 0.30
    MIN_MARKUP = 0.20
    MAX_MARKUP = 0.40
    MIN_PRICE_COST_RATIO = 1.0
    MAX_PRICE_COST_RATIO = 2.0
    MAX_DAILY_PRICE_CHANGE = 0.10
    
    # Service level parameters
    SERVICE_LEVEL = 0.95  # 95%
    Z_SCORE_95 = 1.645
    Z_SCORE_90 = 1.282
    Z_SCORE_975 = 1.96
    
    # Forecast period
    FORECAST_START_DATE = date(2023, 7, 1)
    FORECAST_END_DATE = date(2023, 7, 7)
    
    # Output files
    CLEAN_ITEMS_FILE = 'clean_items.csv'
    TRAIN_FEATURES_FILE = 'train_features.csv'
    TEST_FEATURES_FILE = 'test_features.csv'
    MODEL_RESULTS_FILE = 'enhanced_demand_model_results.csv'
    BACKTEST_SPLITS_FILE = 'backtest_splits_results.csv'
    BACKTEST_STABILITY_FILE = 'backtest_stability_results.csv'
    PRICING_REPLENISHMENT_FILE = 'pricing_and_replenishment_2023-07-01_07.csv'
    ANALYSIS_REPORT_FILE = 'comprehensive_analysis_report.md'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def get_output_path(cls, filename):
        """Get full path for output file"""
        return os.path.join(cls.OUTPUT_DIR, filename)
    
    @classmethod
    def get_log_path(cls, filename):
        """Get full path for log file"""
        return os.path.join(cls.LOGS_DIR, filename)
    
    @classmethod
    def get_data_path(cls, filename):
        """Get full path for data file"""
        return os.path.join(cls.DATA_DIR, filename)