"""
Future Demand and Cost Estimation Module (Stage 5)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

class DemandForecaster:
    def __init__(self):
        self.logger = setup_logger('demand_forecaster')
        self.trained_models = {}
        self.forecast_results = {}
        
    def load_trained_models(self, model_results: Dict[str, Dict]):
        """Load trained models from previous stage"""
        self.trained_models = model_results
        self.logger.info(f"Loaded trained models for {len(model_results)} categories")
    
    def create_forecast_features(self, historical_df: pd.DataFrame, 
                               forecast_dates: List[datetime]) -> pd.DataFrame:
        """Create features for forecast period"""
        
        self.logger.info("Creating forecast features...")
        
        forecast_features = []
        
        for category in historical_df['分类名称'].unique():
            category_data = historical_df[historical_df['分类名称'] == category].copy()
            category_data = category_data.sort_values('销售日期')
            
            for forecast_date in forecast_dates:
                # Get most recent data for this category
                recent_data = category_data[category_data['销售日期'] < forecast_date].tail(30)
                
                if len(recent_data) == 0:
                    continue
                
                # Calculate features for each item in category
                for item_code in recent_data['单品编码'].unique():
                    item_data = recent_data[recent_data['单品编码'] == item_code].copy()
                    
                    if len(item_data) == 0:
                        continue
                    
                    # Get most recent values
                    latest_row = item_data.iloc[-1]
                    
                    # Create forecast row
                    forecast_row = {
                        '销售日期': forecast_date,
                        '单品编码': item_code,
                        '单品名称': latest_row['单品名称'],
                        '分类编码': latest_row['分类编码'],
                        '分类名称': latest_row['分类名称'],
                        
                        # Date features
                        'year': forecast_date.year,
                        'month': forecast_date.month,
                        'day_of_week': forecast_date.weekday(),
                        'is_weekend': 1 if forecast_date.weekday() >= 5 else 0,
                        'day_of_month': forecast_date.day,
                        
                        # Latest price and cost
                        '正常销售单价(元/千克)': latest_row['正常销售单价(元/千克)'],
                        '批发价格(元/千克)': latest_row['批发价格(元/千克)'],
                        '成本加成率': latest_row['成本加成率'],
                        
                        # Log features
                        'ln_price': np.log(latest_row['正常销售单价(元/千克)'] + 0.001),
                        'ln_cost': np.log(latest_row['批发价格(元/千克)'] + 0.001),
                        
                        # Lag features (use most recent available)
                        'quantity_lag1': item_data['正常销量(千克)'].iloc[-1] if len(item_data) >= 1 else 0,
                        'quantity_lag7': item_data['正常销量(千克)'].iloc[-7] if len(item_data) >= 7 else item_data['正常销量(千克)'].mean(),
                        'price_lag1': latest_row['正常销售单价(元/千克)'],
                        
                        # Rolling features (use recent period)
                        'quantity_rolling_7d_mean': item_data['正常销量(千克)'].tail(7).mean(),
                        'quantity_rolling_14d_mean': item_data['正常销量(千克)'].tail(14).mean(),
                        'quantity_rolling_7d_std': item_data['正常销量(千克)'].tail(7).std() if len(item_data) >= 2 else 0,
                        'price_rolling_7d_mean': item_data['正常销售单价(元/千克)'].tail(7).mean(),
                        
                        # Category features (from recent period)
                        'category_avg_price': recent_data['正常销售单价(元/千克)'].mean(),
                        'category_avg_cost': recent_data['批发价格(元/千克)'].mean(),
                        
                        # Item share (use recent average)
                        'item_share_in_category': (item_data['正常销量(千克)'].tail(7).mean() / 
                                                 recent_data.groupby('销售日期')['正常销量(千克)'].sum().tail(7).mean()),
                        
                        # Business features
                        'profit_margin': latest_row.get('profit_margin', 0.2),
                        'is_discount_sale': 0
                    }
                    
                    forecast_features.append(forecast_row)
        
        forecast_df = pd.DataFrame(forecast_features)
        
        # Fill missing values
        numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
        forecast_df[numeric_cols] = forecast_df[numeric_cols].fillna(0)
        
        self.logger.info(f"Created forecast features: {len(forecast_df)} rows")
        return forecast_df
    
    def predict_item_demand(self, forecast_features: pd.DataFrame) -> pd.DataFrame:
        """Predict demand for each item using trained models"""
        
        self.logger.info("Predicting item-level demand...")
        
        predictions = []
        
        for category in forecast_features['分类名称'].unique():
            if category not in self.trained_models:
                self.logger.warning(f"No trained model for category {category}")
                continue
            
            category_features = forecast_features[forecast_features['分类名称'] == category].copy()
            model_info = self.trained_models[category]
            best_model_name = model_info['best_model']
            best_model = model_info['models'][best_model_name]['model']
            
            # Prepare features for prediction
            feature_cols = [
                'ln_price', 'ln_cost', '成本加成率', 'is_weekend', 
                'month', 'day_of_month',
                'quantity_lag1', 'quantity_lag7', 'price_lag1',
                'quantity_rolling_7d_mean', 'quantity_rolling_14d_mean', 
                'quantity_rolling_7d_std', 'price_rolling_7d_mean',
                'category_avg_price', 'category_avg_cost',
                'item_share_in_category', 'profit_margin'
            ]
            
            available_features = [col for col in feature_cols if col in category_features.columns]
            X = category_features[available_features].fillna(0)
            
            # Make predictions
            if best_model_name == 'huber':
                scaler = model_info['models'][best_model_name]['scaler']
                X_scaled = scaler.transform(X)
                y_pred_log = best_model.predict(X_scaled)
            else:
                y_pred_log = best_model.predict(X)
            
            # Convert back from log space
            y_pred = np.exp(y_pred_log)
            
            # Store predictions
            for idx, pred in enumerate(y_pred):
                row = category_features.iloc[idx]
                predictions.append({
                    '销售日期': row['销售日期'],
                    '单品编码': row['单品编码'],
                    '单品名称': row['单品名称'],
                    '分类编码': row['分类编码'],
                    '分类名称': row['分类名称'],
                    'predicted_quantity': max(0, pred),  # Ensure non-negative
                    'predicted_ln_quantity': y_pred_log[idx]
                })
        
        predictions_df = pd.DataFrame(predictions)
        self.logger.info(f"Generated predictions for {len(predictions_df)} items")
        
        return predictions_df
    
    def aggregate_to_category_level(self, item_predictions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate item predictions to category level"""
        
        self.logger.info("Aggregating predictions to category level...")
        
        category_predictions = item_predictions.groupby(['销售日期', '分类编码', '分类名称']).agg({
            'predicted_quantity': 'sum'
        }).reset_index()
        
        category_predictions.columns = ['销售日期', '分类编码', '分类名称', 'predicted_demand']
        
        self.logger.info(f"Category-level predictions: {len(category_predictions)} rows")
        return category_predictions
    
    def estimate_future_costs(self, historical_df: pd.DataFrame, 
                            forecast_dates: List[datetime]) -> pd.DataFrame:
        """Estimate future costs using moving average"""
        
        self.logger.info("Estimating future costs...")
        
        cost_estimates = []
        
        for category in historical_df['分类名称'].unique():
            category_data = historical_df[historical_df['分类名称'] == category].copy()
            category_data = category_data.sort_values('销售日期')
            
            # Calculate recent cost trends
            recent_data = category_data.tail(14)  # Last 14 days
            
            if len(recent_data) == 0:
                continue
            
            # Calculate weighted average cost (more weight to recent data)
            weights = np.exp(np.linspace(-1, 0, len(recent_data)))
            weighted_avg_cost = np.average(recent_data['批发价格(元/千克)'], weights=weights)
            
            # Simple trend estimation
            if len(recent_data) >= 7:
                recent_costs = recent_data['批发价格(元/千克)'].values
                trend = (recent_costs[-3:].mean() - recent_costs[:3].mean()) / len(recent_costs)
            else:
                trend = 0
            
            # Estimate costs for each forecast date
            for i, forecast_date in enumerate(forecast_dates):
                estimated_cost = weighted_avg_cost + (trend * i)
                estimated_cost = max(0.1, estimated_cost)  # Ensure positive cost
                
                cost_estimates.append({
                    '销售日期': forecast_date,
                    '分类编码': recent_data['分类编码'].iloc[-1],
                    '分类名称': category,
                    'estimated_cost': estimated_cost
                })
        
        cost_df = pd.DataFrame(cost_estimates)
        self.logger.info(f"Cost estimates: {len(cost_df)} rows")
        
        return cost_df
    
    def estimate_prediction_uncertainty(self, historical_df: pd.DataFrame) -> Dict[str, float]:
        """Estimate prediction uncertainty by category using historical residuals"""
        
        self.logger.info("Estimating prediction uncertainty...")
        
        uncertainty = {}
        
        for category in historical_df['分类名称'].unique():
            category_data = historical_df[historical_df['分类名称'] == category].copy()
            
            if len(category_data) < 10:
                uncertainty[category] = 1.0  # Default uncertainty
                continue
            
            # Use rolling window to estimate residual variance
            category_data = category_data.sort_values('销售日期')
            quantities = category_data['正常销量(千克)'].values
            
            # Calculate rolling mean and residuals
            rolling_mean = pd.Series(quantities).rolling(window=7, min_periods=1).mean()
            residuals = quantities - rolling_mean
            
            # Estimate standard deviation of residuals
            residual_std = np.std(residuals)
            uncertainty[category] = residual_std
        
        self.logger.info(f"Estimated uncertainty for {len(uncertainty)} categories")
        return uncertainty
    
    def run_forecast(self, historical_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """Run complete forecast pipeline"""
        
        self.logger.info("Starting demand and cost forecasting...")
        
        # Generate forecast dates
        forecast_dates = [Config.FORECAST_START_DATE + timedelta(days=i) 
                         for i in range((Config.FORECAST_END_DATE - Config.FORECAST_START_DATE).days + 1)]
        
        self.logger.info(f"Forecasting for {len(forecast_dates)} days: "
                        f"{forecast_dates[0]} to {forecast_dates[-1]}")
        
        # Create forecast features
        forecast_features = self.create_forecast_features(historical_df, forecast_dates)
        
        # Predict item-level demand
        item_predictions = self.predict_item_demand(forecast_features)
        
        # Aggregate to category level
        category_predictions = self.aggregate_to_category_level(item_predictions)
        
        # Estimate future costs
        cost_estimates = self.estimate_future_costs(historical_df, forecast_dates)
        
        # Estimate uncertainty
        uncertainty_estimates = self.estimate_prediction_uncertainty(historical_df)
        
        # Combine predictions and costs
        forecast_results = category_predictions.merge(
            cost_estimates, 
            on=['销售日期', '分类编码', '分类名称'], 
            how='left'
        )
        
        # Add uncertainty estimates
        forecast_results['prediction_std'] = forecast_results['分类名称'].map(uncertainty_estimates)
        
        self.logger.info("Demand and cost forecasting completed successfully!")
        return forecast_results, item_predictions, uncertainty_estimates
    
    def run(self, historical_df: pd.DataFrame, model_results: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """Run complete forecasting pipeline"""
        
        # Load trained models
        self.load_trained_models(model_results)
        
        # Run forecast
        return self.run_forecast(historical_df)

if __name__ == "__main__":
    # This would be called from the main pipeline
    pass