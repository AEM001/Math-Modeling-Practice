# -*- coding: utf-8 -*-
"""
精简版优化器模块
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

warnings.filterwarnings('ignore')

# 导入字体配置
# from font_config import setup_chinese_font # Temporarily disable for compatibility

class VegetableOptimizer:
    """精简版蔬菜定价与补货优化器"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化优化器"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        with open(os.path.join(project_root, config_path), 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.output_paths = {
            'results_dir': os.path.join(project_root, self.config['output_paths']['results_dir']),
            'model_dir': os.path.join(project_root, self.config['output_paths']['model_dir']),
            'figures_dir': os.path.join(project_root, self.config['output_paths']['figures_dir']),
            'reports_dir': os.path.join(project_root, self.config['output_paths']['reports_dir'])
        }
        self.opt_config = self.config['optimization']
        self.demand_models = {}
        # Store the start date from training to calculate time trend
        self.time_series_start_date = pd.to_datetime(self.config['time_series_start_date'])
        # Denominator used to normalize time_trend, aligned with training data span
        self.time_trend_denominator = 365
        # Try to align time trend scaling with FeatureEngineer outputs
        self._init_time_trend_scaler()

    def _init_time_trend_scaler(self):
        """Initialize time trend normalization using the training feature file if available."""
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            train_feat_rel = self.config.get('data_paths', {}).get('train_features')
            if not train_feat_rel:
                return
            train_feat_path = os.path.join(project_root, train_feat_rel)
            if os.path.exists(train_feat_path):
                df = pd.read_csv(train_feat_path, parse_dates=['销售日期'])
                min_dt = pd.to_datetime(df['销售日期']).min()
                max_dt = pd.to_datetime(df['销售日期']).max()
                if pd.notnull(min_dt) and pd.notnull(max_dt) and max_dt > min_dt:
                    self.time_series_start_date = min_dt
                    self.time_trend_denominator = max(1, (max_dt - min_dt).days)
        except Exception as e:
            print(f"Warning: failed to initialize time_trend scaler: {e}")
    
    def load_demand_models(self):
        """Loads the fitted SARIMAX models and their metadata."""
        print("Loading SARIMAX demand models...")
        model_dir = self.output_paths['model_dir']
        results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')

        if not os.path.exists(results_path):
            print(f"ERROR: Demand model results not found at {results_path}")
            return

        models_df = pd.read_csv(results_path)
        exog_features = self.config['sarimax_exog_features']

        for index, row in models_df.iterrows():
            category = row['category']
            model_path = row['model_path']
            if os.path.exists(model_path):
                try:
                    model_obj = joblib.load(model_path)
                    self.demand_models[category] = {
                        'model_object': model_obj,
                        'price_elasticity': row['price_elasticity'],
                        'exog_features': exog_features, # Store the feature list
                        'test_r2': row['test_r2']
                    }
                except Exception as e:
                    print(f"  - Failed to load model for {category}: {e}")
            else:
                print(f"  - Model file not found for {category} at {model_path}")

        print(f"Successfully loaded {len(self.demand_models)} demand models.")
    
    def predict_demand(self, category, price, date):
        """Predicts demand for a given category, price, and date using its SARIMAX model."""
        if category not in self.demand_models:
            # Return a default baseline demand and high uncertainty if no model exists
            return 15.0, 15.0 * 0.5 

        model_info = self.demand_models[category]
        model = model_info['model_object']
        features = model_info['exog_features']

        # 1. Construct the exogenous feature vector (X)
        date = pd.to_datetime(date)
        exog_data = pd.DataFrame(index=[0])
        
        if 'ln_price' in features:
            exog_data['ln_price'] = np.log(price)
        if 'is_weekend' in features:
            exog_data['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        if 'time_trend' in features:
            # Calculate days since the (training) start date and normalize with the same span as training
            days_since_start = max(0, (date - self.time_series_start_date).days)
            denom = max(1, getattr(self, 'time_trend_denominator', 365))
            exog_data['time_trend'] = min(days_since_start / denom, 1.0)
        # Add other potential features if they were used in training
        # This part needs to be perfectly aligned with feature_engineer.py

        # Ensure the columns are in the same order as during training
        exog_data = exog_data[features]
        # Ensure numeric dtype
        exog_data = exog_data.astype(float)

        # 2. Predict using the model
        try:
            # Use predict with confidence interval for uncertainty estimation
            try:
                ln_pred, conf_int = model.predict(n_periods=1, exogenous=exog_data, return_conf_int=True)
            except TypeError:
                # Fallback for versions expecting X keyword
                ln_pred, conf_int = model.predict(n_periods=1, X=exog_data, return_conf_int=True)

            ln_pred_arr = np.asarray(ln_pred).reshape(-1)
            if ln_pred_arr.size < 1:
                raise ValueError("empty prediction array")
            ln_pred_val = float(ln_pred_arr[0])
            if not np.isfinite(ln_pred_val):
                raise ValueError(f"ln_pred not finite: {ln_pred_val}")
            predicted_demand = float(np.exp(ln_pred_val))
            
            # Guard against numerical underflow/overflow
            if not np.isfinite(predicted_demand) or predicted_demand <= 0:
                print(f"ERROR predicting for {category} on {date}: Invalid prediction {predicted_demand}")
                return 15.0, 15.0 * 0.5
                
        except Exception as e:
            print(f"ERROR predicting for {category} on {date}: {e}")
            return 15.0, 15.0 * 0.5 # Fallback on error

        # 3. Estimate uncertainty
        demand_std = None
        try:
            # If confidence interval was returned, approximate std from it on log-scale
            ci_arr = np.asarray(conf_int)
            # Handle shapes: (2,) or (1,2) or DataFrame-like
            if ci_arr.ndim == 1 and ci_arr.size == 2:
                ln_lower, ln_upper = float(ci_arr[0]), float(ci_arr[1])
            else:
                ln_lower, ln_upper = float(ci_arr[0, 0]), float(ci_arr[0, 1])
            ln_width = max(0.0, ln_upper - ln_lower)
            # 95% CI width ≈ 3.92 * sigma
            ln_sigma = max(ln_width / 3.92, 1e-6)
            # Delta method: std(Y) ≈ Y * sigma when Y = exp(X)
            demand_std = max(predicted_demand * ln_sigma, predicted_demand * 0.05)
        except Exception:
            pass

        if demand_std is None or not np.isfinite(demand_std) or demand_std <= 0:
            # Fallback: derive a conservative std from test R^2
            uncertainty_factor = np.sqrt(1 - max(0, model_info.get('test_r2', 0.5)))
            demand_std = max(predicted_demand * uncertainty_factor * 0.5, predicted_demand * 0.05)
        
        return predicted_demand, demand_std
    
    def calculate_profit_scenarios(self, category, price, date, quantity, wholesale_cost, 
                                 n_scenarios=20, wastage_rate=0.05):
        """Calculates profit scenarios using model-based demand prediction."""
        mean_demand, std_demand = self.predict_demand(category, price, date)
        
        if std_demand <= 0:
            std_demand = mean_demand * 0.1 # Assign a floor to uncertainty

        np.random.seed(hash(date) % (2**32 - 1)) # Seed with date for consistency
        demand_scenarios = np.random.normal(mean_demand, std_demand, n_scenarios)
        demand_scenarios = np.maximum(demand_scenarios, 0.1)
        
        profits = []
        for demand in demand_scenarios:
            available_quantity = quantity * (1 - wastage_rate)
            actual_sales = min(demand, available_quantity)
            revenue = actual_sales * price
            cost = quantity * wholesale_cost
            stockout = max(0, demand - actual_sales)
            stockout_penalty = stockout * wholesale_cost * self.opt_config['stockout_penalty_weight']
            profit = revenue - cost - stockout_penalty
            profits.append(profit)
        
        return np.array(profits)

    def evaluate_markup_profit(self, category, wholesale_cost, date, markup_ratio):
        """Evaluates the profit for a given markup ratio by predicting demand."""
        price = wholesale_cost * markup_ratio
        
        # Predict demand and determine order quantity with safety stock
        predicted_demand, std_demand = self.predict_demand(category, price, date)
        order_quantity = self.calculate_order_quantity(predicted_demand, std_demand)

        # Calculate expected profit using Monte Carlo simulation
        profit_scenarios = self.calculate_profit_scenarios(
            category, price, date, order_quantity, wholesale_cost,
            n_scenarios=self.opt_config['monte_carlo_samples']
        )
        
        expected_profit = np.mean(profit_scenarios)
        profit_std = np.std(profit_scenarios)
        
        # Risk-adjusted profit (penalize variance)
        risk_penalty = self.opt_config.get('risk_aversion_factor', 0.1) * profit_std
        return expected_profit - risk_penalty

    def golden_section_search(self, category, wholesale_cost, date, a, b, tol=1e-3):
        """Golden section search for the optimal markup ratio."""
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi
        
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        
        # Note: We maximize profit, so we don't negate the function value
        f1 = self.evaluate_markup_profit(category, wholesale_cost, date, x1)
        f2 = self.evaluate_markup_profit(category, wholesale_cost, date, x2)
        
        for _ in range(self.opt_config.get('golden_section_iterations', 15)):
            if f1 > f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = self.evaluate_markup_profit(category, wholesale_cost, date, x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - resphi * (b - a)
                f2 = self.evaluate_markup_profit(category, wholesale_cost, date, x2)
            
            if abs(b - a) < tol:
                break
        
        return (a + b) / 2

    def calculate_order_quantity(self, predicted_demand, std_demand):
        """Calculates order quantity including safety stock based on service level."""
        service_level = self.opt_config['service_level']
        # Z-score for standard normal distribution
        z_score_map = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.98: 2.05}
        safety_factor = z_score_map.get(service_level, 1.65) # Default to 95%
        
        safety_stock = safety_factor * std_demand
        order_quantity = predicted_demand + safety_stock
        return max(0, order_quantity) # Ensure non-negative

    def optimize_category(self, category, wholesale_cost, date):
        """Optimizes price and quantity for a category on a specific date."""
        if category not in self.demand_models:
            # Default strategy if no model is available
            optimal_markup = np.mean([self.opt_config['min_markup_ratio'], self.opt_config['max_markup_ratio']])
            optimal_price = wholesale_cost * optimal_markup
            # Cannot predict demand, so return a placeholder quantity
            return optimal_price, 10.0

        # Define search bounds for markup ratio from config
        min_markup = self.opt_config['min_markup_ratio']
        max_markup = self.opt_config['max_markup_ratio']

        # Find the optimal markup ratio using golden section search
        optimal_markup = self.golden_section_search(
            category, wholesale_cost, date, min_markup, max_markup
        )
        optimal_price = wholesale_cost * optimal_markup

        predicted_demand, std_demand = self.predict_demand(category, optimal_price, date)
        optimal_quantity = self.calculate_order_quantity(predicted_demand, std_demand)
        
        return optimal_price, optimal_quantity

    def generate_wholesale_forecasts(self, categories):
        """Generates simplified wholesale price forecasts for the given categories."""
        forecasts = {}
        horizon = self.opt_config['optimization_horizon']
        base_prices = self.config['base_wholesale_prices']

        for category in categories:
            base_price = base_prices.get(category, 5.0) # Default price if not in config
            days = np.arange(horizon)
            seasonal_effect = 0.1 * np.sin(2 * np.pi * days / 7) # Weekly seasonality
            noise = np.random.normal(0, 0.03, horizon)
            price_multipliers = 1 + seasonal_effect + noise
            forecasts[category] = base_price * price_multipliers
        return forecasts

    def run_daily_optimization(self):
        """Runs the daily optimization loop for all categories with models."""
        print("\nStarting daily optimization...")
        categories_with_models = list(self.demand_models.keys())
        if not categories_with_models:
            print("No models loaded, cannot run optimization.")
            return pd.DataFrame()
            
        wholesale_forecasts = self.generate_wholesale_forecasts(categories_with_models)
        
        results = []
        horizon = self.opt_config['optimization_horizon']
        start_date = datetime.now().date()
        
        for day in range(horizon):
            current_date = start_date + timedelta(days=day)
            print(f"\n-- Optimizing for Date: {current_date.strftime('%Y-%m-%d')} --")
            
            for category in categories_with_models:
                wholesale_cost = wholesale_forecasts[category][day]
                
                optimal_price, optimal_quantity = self.optimize_category(
                    category, wholesale_cost, current_date
                )
                
                # Recalculate expected profit for reporting
                profit_scenarios = self.calculate_profit_scenarios(
                    category, optimal_price, current_date, optimal_quantity, wholesale_cost
                )
                expected_profit = np.mean(profit_scenarios)
                
                results.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'category': category,
                    'wholesale_cost': wholesale_cost,
                    'optimal_price': optimal_price,
                    'optimal_quantity': optimal_quantity,
                    'expected_profit': expected_profit
                })
                
                print(f"  - {category}: Cost={wholesale_cost:.2f}, Price={optimal_price:.2f}, Quantity={optimal_quantity:.2f}, Profit={expected_profit:.2f}")

        return pd.DataFrame(results)

    def save_optimization_results(self, results_df):
        """Saves optimization results to a CSV file."""
        if not os.path.exists(self.output_paths['results_dir']):
            os.makedirs(self.output_paths['results_dir'])
        path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
        results_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\nOptimization results saved to {path}")

    def plot_price_quantity_profit(self, category, wholesale_cost, date):
        """Plots the price-demand-profit relationship for a given date."""
        if category not in self.demand_models:
            print(f"Cannot plot for {category}: No demand model available.")
            return

        price_range = np.linspace(wholesale_cost * self.opt_config['min_markup_ratio'], 
                                  wholesale_cost * self.opt_config['max_markup_ratio'], 50)
        demands = []
        profits = []

        for price in price_range:
            predicted_demand, std_demand = self.predict_demand(category, price, date)
            demands.append(predicted_demand)
            
            order_quantity = self.calculate_order_quantity(predicted_demand, std_demand)
            profit_scenarios = self.calculate_profit_scenarios(
                category, price, date, order_quantity, wholesale_cost
            )
            profits.append(np.mean(profit_scenarios))

        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        color = 'tab:blue'
        ax1.set_xlabel(f'{category} - Selling Price', fontsize=14)
        ax1.set_ylabel('Predicted Demand', color=color, fontsize=14)
        ax1.plot(price_range, demands, color=color, marker='.', linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Expected Profit', color=color, fontsize=14)
        ax2.plot(price_range, profits, color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        max_profit_idx = np.argmax(profits)
        optimal_p = price_range[max_profit_idx]
        max_profit = profits[max_profit_idx]
        ax2.axvline(x=optimal_p, color='green', linestyle='-.', label=f'Optimal Price: {optimal_p:.2f}')
        ax2.scatter(optimal_p, max_profit, color='green', s=100, zorder=5, label=f'Max Profit: {max_profit:.2f}')

        plt.title(f'{category} - Price vs. Profit Analysis for {date.strftime("%Y-%m-%d")}', fontsize=16)
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        fig.tight_layout()
        
        if not os.path.exists(self.output_paths['figures_dir']):
            os.makedirs(self.output_paths['figures_dir'])
        figure_path = os.path.join(self.output_paths['figures_dir'], f'{category}_price_profit_analysis.png')
        plt.savefig(figure_path)
        plt.close()
        print(f"  - Saved price-profit analysis plot for {category}.")

if __name__ == '__main__':
    optimizer = VegetableOptimizer(config_path='config/config.json')
    optimizer.load_demand_models()
    
    if optimizer.demand_models:
        optimization_results = optimizer.run_daily_optimization()
        
        if not optimization_results.empty:
            optimizer.save_optimization_results(optimization_results)
            
            print("\nGenerating price-profit analysis plots for example categories...")
            example_categories = list(optimizer.demand_models.keys())[:3]
            today = datetime.now().date()
            
            for cat in example_categories:
                # Ensure the category from results exists before plotting
                if cat in optimization_results['category'].unique():
                    first_day_cost = optimization_results[optimization_results['category'] == cat]['wholesale_cost'].iloc[0]
                    optimizer.plot_price_quantity_profit(cat, first_day_cost, today)
    else:
        print("No demand models were loaded. Cannot run optimization.")
