"""
Demand Modeling Module (Stage 3)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

class DemandModeler:
    def __init__(self):
        self.logger = setup_logger('demand_modeler')
        self.models = {}
        self.scalers = {}
        self.model_results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        
        # Define feature columns for modeling
        feature_cols = [
            'ln_price', 'ln_cost', '成本加成率', 'is_weekend', 
            'month', 'day_of_month',
            'quantity_lag1', 'quantity_lag7', 'price_lag1',
            'quantity_rolling_7d_mean', 'quantity_rolling_14d_mean', 
            'quantity_rolling_7d_std', 'price_rolling_7d_mean',
            'category_avg_price', 'category_avg_cost',
            'item_share_in_category', 'profit_margin'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].copy()
        y = df['ln_quantity'].copy()  # Use log-transformed quantity as target
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model evaluation metrics"""
        
        # Convert back from log space for some metrics
        y_true_exp = np.exp(y_true)
        y_pred_exp = np.exp(y_pred)
        
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae_exp': mean_absolute_error(y_true_exp, y_pred_exp),
            'rmse_exp': np.sqrt(mean_squared_error(y_true_exp, y_pred_exp)),
            'mape': self.mean_absolute_percentage_error(y_true_exp, y_pred_exp)
        }
        
        return metrics
    
    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE (Mean Absolute Percentage Error)"""
        y_true = np.clip(y_true, 0.001, None)  # Avoid division by zero
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def fit_huber_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Fit Huber regression model for price elasticity estimation"""
        
        self.logger.info("Fitting Huber regression model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit Huber regression
        huber = HuberRegressor(epsilon=1.35, max_iter=1000, alpha=0.01)
        huber.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = huber.predict(X_train_scaled)
        y_test_pred = huber.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Extract price elasticity (coefficient of ln_price)
        if 'ln_price' in X_train.columns:
            price_elasticity = huber.coef_[X_train.columns.get_loc('ln_price')]
        else:
            price_elasticity = np.nan
        
        return {
            'model': huber,
            'scaler': scaler,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'price_elasticity': price_elasticity,
            'feature_importance': dict(zip(X_train.columns, np.abs(huber.coef_)))
        }
    
    def fit_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Fit Random Forest model"""
        
        self.logger.info("Fitting Random Forest model...")
        
        # Initialize model
        rf = RandomForestRegressor(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=Config.RF_MAX_DEPTH,
            random_state=Config.RF_RANDOM_STATE,
            n_jobs=-1
        )
        
        # Fit model
        rf.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        return {
            'model': rf,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(zip(X_train.columns, rf.feature_importances_))
        }
    
    def fit_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Fit Gradient Boosting model"""
        
        self.logger.info("Fitting Gradient Boosting model...")
        
        # Initialize model
        gb = GradientBoostingRegressor(
            n_estimators=Config.GB_N_ESTIMATORS,
            learning_rate=Config.GB_LEARNING_RATE,
            max_depth=Config.GB_MAX_DEPTH,
            random_state=Config.GB_RANDOM_STATE
        )
        
        # Fit model
        gb.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = gb.predict(X_train)
        y_test_pred = gb.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        return {
            'model': gb,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(zip(X_train.columns, gb.feature_importances_))
        }
    
    def select_best_model(self, models_dict: Dict[str, Dict]) -> str:
        """Select best model based on test R2 score"""
        
        best_model_name = None
        best_r2 = -np.inf
        
        for model_name, model_info in models_dict.items():
            if model_name != 'huber':  # Skip Huber for production model selection
                test_r2 = model_info['test_metrics']['r2']
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model_name = model_name
        
        return best_model_name
    
    def train_category_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """Train models for each category"""
        
        self.logger.info("Training models for each category...")
        
        category_results = {}
        categories = train_df['分类名称'].unique()
        
        for category in categories:
            self.logger.info(f"Training models for category: {category}")
            
            # Filter data for this category
            train_cat = train_df[train_df['分类名称'] == category].copy()
            test_cat = test_df[test_df['分类名称'] == category].copy()
            
            if len(train_cat) < 10 or len(test_cat) < 5:
                self.logger.warning(f"Insufficient data for category {category}, skipping...")
                continue
            
            # Prepare features
            X_train, y_train = self.prepare_features(train_cat)
            X_test, y_test = self.prepare_features(test_cat)
            
            # Train all models
            models_dict = {}
            
            # Huber regression
            try:
                models_dict['huber'] = self.fit_huber_model(X_train, y_train, X_test, y_test)
            except Exception as e:
                self.logger.warning(f"Huber model failed for {category}: {e}")
            
            # Random Forest
            try:
                models_dict['random_forest'] = self.fit_random_forest(X_train, y_train, X_test, y_test)
            except Exception as e:
                self.logger.warning(f"Random Forest failed for {category}: {e}")
            
            # Gradient Boosting
            try:
                models_dict['gradient_boosting'] = self.fit_gradient_boosting(X_train, y_train, X_test, y_test)
            except Exception as e:
                self.logger.warning(f"Gradient Boosting failed for {category}: {e}")
            
            # Select best model
            if len([k for k in models_dict.keys() if k != 'huber']) > 0:
                best_model_name = self.select_best_model(models_dict)
                
                # Get price elasticity from Huber model if available
                price_elasticity = models_dict.get('huber', {}).get('price_elasticity', np.nan)
                
                category_results[category] = {
                    'models': models_dict,
                    'best_model': best_model_name,
                    'price_elasticity': price_elasticity,
                    'best_model_metrics': models_dict[best_model_name]['test_metrics']
                }
                
                self.logger.info(f"Category {category}: Best model = {best_model_name}, "
                               f"Test R2 = {models_dict[best_model_name]['test_metrics']['r2']:.3f}")
        
        return category_results
    
    def save_model_results(self, category_results: Dict[str, Dict]):
        """Save model results to CSV"""
        
        results_list = []
        
        for category, result in category_results.items():
            best_model = result['best_model']
            metrics = result['best_model_metrics']
            
            results_list.append({
                'category': category,
                'best_model': best_model,
                'test_r2': metrics['r2'],
                'test_mae': metrics['mae'],
                'test_rmse': metrics['rmse'],
                'test_mape': metrics['mape'],
                'price_elasticity': result['price_elasticity']
            })
        
        results_df = pd.DataFrame(results_list)
        output_path = Config.get_output_path(Config.MODEL_RESULTS_FILE)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"Model results saved to: {output_path}")
    
    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """Run the complete demand modeling pipeline"""
        
        self.logger.info("Starting demand modeling pipeline...")
        
        # Train models by category
        category_results = self.train_category_models(train_df, test_df)
        
        # Save results
        self.save_model_results(category_results)
        
        # Store results for later use
        self.model_results = category_results
        
        self.logger.info("Demand modeling pipeline completed successfully!")
        return category_results

if __name__ == "__main__":
    # Load feature data and run modeling
    train_df = pd.read_csv(Config.get_output_path(Config.TRAIN_FEATURES_FILE))
    test_df = pd.read_csv(Config.get_output_path(Config.TEST_FEATURES_FILE))
    
    modeler = DemandModeler()
    results = modeler.run(train_df, test_df)