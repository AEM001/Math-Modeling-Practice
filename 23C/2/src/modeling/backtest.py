"""
Backtest and Stability Assessment Module (Stage 4)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

class BacktestEvaluator:
    def __init__(self):
        self.logger = setup_logger('backtest_evaluator')
        self.split_results = []
        self.stability_results = []
        
    def create_time_series_splits(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create rolling-origin cross-validation splits"""
        
        df_sorted = df.sort_values('销售日期')
        n_samples = len(df_sorted)
        
        # Calculate minimum training size
        min_train_size = max(Config.CV_MIN_TRAIN_SIZE, int(n_samples * 0.3))
        
        # Calculate number of splits
        max_splits = min(Config.CV_MAX_SPLITS, 
                        (n_samples - min_train_size) // 10)
        
        if max_splits < 2:
            self.logger.warning("Insufficient data for cross-validation")
            return []
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=max_splits, test_size=None)
        splits = list(tscv.split(df_sorted))
        
        self.logger.info(f"Created {len(splits)} time series splits")
        return splits
    
    def prepare_features_for_cv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for cross-validation"""
        
        feature_cols = [
            'ln_price', 'ln_cost', '成本加成率', 'is_weekend', 
            'month', 'day_of_month',
            'quantity_lag1', 'quantity_lag7', 'price_lag1',
            'quantity_rolling_7d_mean', 'quantity_rolling_14d_mean', 
            'quantity_rolling_7d_std', 'price_rolling_7d_mean',
            'category_avg_price', 'category_avg_cost',
            'item_share_in_category', 'profit_margin'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].copy().fillna(method='ffill').fillna(0)
        y = df['ln_quantity'].copy()
        
        return X, y
    
    def calculate_cv_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate cross-validation metrics"""
        
        y_true_exp = np.exp(y_true)
        y_pred_exp = np.exp(y_pred)
        
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': self.mean_absolute_percentage_error(y_true_exp, y_pred_exp)
        }
        
        return metrics
    
    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE"""
        y_true = np.clip(y_true, 0.001, None)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def run_single_cv_fold(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series, 
                          model_type: str) -> Dict[str, float]:
        """Run a single cross-validation fold"""
        
        try:
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=Config.RF_N_ESTIMATORS,
                    max_depth=Config.RF_MAX_DEPTH,
                    random_state=Config.RF_RANDOM_STATE,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=Config.GB_N_ESTIMATORS,
                    learning_rate=Config.GB_LEARNING_RATE,
                    max_depth=Config.GB_MAX_DEPTH,
                    random_state=Config.GB_RANDOM_STATE
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_type == 'huber':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = HuberRegressor(epsilon=1.35, max_iter=1000, alpha=0.01)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return self.calculate_cv_metrics(y_test, y_pred)
            
        except Exception as e:
            self.logger.warning(f"CV fold failed for {model_type}: {e}")
            return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
    
    def run_category_backtest(self, df: pd.DataFrame, category: str) -> Dict[str, Any]:
        """Run backtest for a single category"""
        
        self.logger.info(f"Running backtest for category: {category}")
        
        category_df = df[df['分类名称'] == category].copy()
        
        if len(category_df) < Config.CV_MIN_TRAIN_SIZE:
            self.logger.warning(f"Insufficient data for category {category}")
            return {}
        
        # Create time series splits
        splits = self.create_time_series_splits(category_df)
        
        if not splits:
            return {}
        
        # Prepare features
        X, y = self.prepare_features_for_cv(category_df)
        
        # Run cross-validation for each model type
        model_types = ['random_forest', 'gradient_boosting', 'huber']
        cv_results = {model_type: [] for model_type in model_types}
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.debug(f"Processing fold {fold_idx + 1}/{len(splits)} for {category}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            for model_type in model_types:
                fold_metrics = self.run_single_cv_fold(X_train, y_train, X_test, y_test, model_type)
                cv_results[model_type].append(fold_metrics)
                
                # Store individual split results
                self.split_results.append({
                    'category': category,
                    'fold': fold_idx + 1,
                    'model_type': model_type,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    **fold_metrics
                })
        
        # Calculate stability metrics
        stability_metrics = {}
        for model_type in model_types:
            fold_metrics = cv_results[model_type]
            
            if fold_metrics and not all(np.isnan([m['r2'] for m in fold_metrics])):
                r2_scores = [m['r2'] for m in fold_metrics if not np.isnan(m['r2'])]
                rmse_scores = [m['rmse'] for m in fold_metrics if not np.isnan(m['rmse'])]
                
                stability_metrics[model_type] = {
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'cv_r2': np.std(r2_scores) / np.mean(r2_scores) if np.mean(r2_scores) > 0 else np.inf,
                    'mean_rmse': np.mean(rmse_scores),
                    'std_rmse': np.std(rmse_scores),
                    'cv_rmse': np.std(rmse_scores) / np.mean(rmse_scores) if np.mean(rmse_scores) > 0 else np.inf,
                    'n_folds': len(r2_scores)
                }
        
        return stability_metrics
    
    def run_backtest_pipeline(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete backtest pipeline for all categories"""
        
        self.logger.info("Starting backtest pipeline...")
        
        categories = train_df['分类名称'].unique()
        
        for category in categories:
            stability_metrics = self.run_category_backtest(train_df, category)
            
            # Store stability results
            for model_type, metrics in stability_metrics.items():
                self.stability_results.append({
                    'category': category,
                    'model_type': model_type,
                    **metrics
                })
        
        # Convert results to DataFrames
        splits_df = pd.DataFrame(self.split_results)
        stability_df = pd.DataFrame(self.stability_results)
        
        return splits_df, stability_df
    
    def save_backtest_results(self, splits_df: pd.DataFrame, stability_df: pd.DataFrame):
        """Save backtest results"""
        
        # Save split results
        splits_path = Config.get_output_path(Config.BACKTEST_SPLITS_FILE)
        splits_df.to_csv(splits_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Backtest splits results saved to: {splits_path}")
        
        # Save stability results
        stability_path = Config.get_output_path(Config.BACKTEST_STABILITY_FILE)
        stability_df.to_csv(stability_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Backtest stability results saved to: {stability_path}")
    
    def generate_stability_summary(self, stability_df: pd.DataFrame):
        """Generate and log stability summary"""
        
        self.logger.info("=== Backtest Stability Summary ===")
        
        for model_type in stability_df['model_type'].unique():
            model_results = stability_df[stability_df['model_type'] == model_type]
            
            mean_r2 = model_results['mean_r2'].mean()
            mean_cv_r2 = model_results['cv_r2'].mean()
            mean_rmse = model_results['mean_rmse'].mean()
            
            self.logger.info(f"{model_type}:")
            self.logger.info(f"  Average R²: {mean_r2:.3f}")
            self.logger.info(f"  Average CV(R²): {mean_cv_r2:.3f}")
            self.logger.info(f"  Average RMSE: {mean_rmse:.3f}")
    
    def run(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete backtest evaluation"""
        
        self.logger.info("Starting backtest and stability assessment...")
        
        # Run backtest pipeline
        splits_df, stability_df = self.run_backtest_pipeline(train_df)
        
        # Save results
        self.save_backtest_results(splits_df, stability_df)
        
        # Generate summary
        self.generate_stability_summary(stability_df)
        
        self.logger.info("Backtest and stability assessment completed successfully!")
        
        return splits_df, stability_df

if __name__ == "__main__":
    # Load training data and run backtest
    train_df = pd.read_csv(Config.get_output_path(Config.TRAIN_FEATURES_FILE))
    
    evaluator = BacktestEvaluator()
    splits_results, stability_results = evaluator.run(train_df)