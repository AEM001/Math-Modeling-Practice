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
        # Global default wastage rate for ordering and simulation
        self.default_wastage_rate = float(self.opt_config.get('wastage_rate', 0.05))
        # Store the start date from training to calculate time trend
        self.time_series_start_date = pd.to_datetime(self.config['time_series_start_date'])
        # Denominator used to normalize time_trend, aligned with training data span
        self.time_trend_denominator = 365
        # Try to align time trend scaling with FeatureEngineer outputs
        self._init_time_trend_scaler()
        # Initialize category baselines and caps from training features
        self._init_category_baselines_and_caps()

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
    
    def _init_category_baselines_and_caps(self):
        """Compute per-category baseline ln_price and demand caps using training features."""
        self.category_ln_price_ref = {}
        self.category_demand_cap = {}
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            train_feat_rel = self.config.get('data_paths', {}).get('train_features')
            if not train_feat_rel:
                return
            train_feat_path = os.path.join(project_root, train_feat_rel)
            if not os.path.exists(train_feat_path):
                return
            df = pd.read_csv(train_feat_path)
            if '分类名称' not in df.columns:
                return
            df['销售日期'] = pd.to_datetime(df['销售日期']) if '销售日期' in df.columns else pd.NaT
            # Compute ln_price reference as mean per category
            grp = df.groupby('分类名称')['ln_price']
            self.category_ln_price_ref = grp.mean().to_dict()
            # Aggregate to category-date quantity (sum of exp(ln_quantity))
            if 'ln_quantity' in df.columns:
                df['quantity'] = np.exp(df['ln_quantity']).clip(lower=1e-6)
                q = df.groupby(['分类名称', '销售日期'], as_index=False)['quantity'].sum()
                qtl = float(self.opt_config.get('demand_cap_quantile', 0.95))
                for cat, sub in q.groupby('分类名称'):
                    self.category_demand_cap[cat] = float(sub['quantity'].quantile(qtl))
        except Exception as e:
            print(f"Warning: failed to initialize category baselines/caps: {e}")
    
    def load_demand_models(self):
        """Loads the fitted RandomForest (or other) models and their metadata."""
        print("Loading demand models...")
        model_dir = self.output_paths['model_dir']
        results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')

        if not os.path.exists(results_path):
            print(f"ERROR: Demand model results not found at {results_path}")
            return

        models_df = pd.read_csv(results_path)
        default_features = self.config.get('tree_features', self.config.get('sarimax_exog_features', []))

        for index, row in models_df.iterrows():
            category = row['category']
            model_path = row['model_path']
            if os.path.exists(model_path):
                try:
                    model_obj = joblib.load(model_path)
                    # Parse features used by the model if present in results
                    features = default_features
                    if 'features' in row and pd.notna(row['features']):
                        raw = str(row['features'])
                        try:
                            import json as _json
                            features = _json.loads(raw)
                        except Exception:
                            # Fallback: strip brackets/quotes and split by comma
                            raw2 = raw.strip().strip('[]')
                            parts = [p.strip().strip("'\"") for p in raw2.split(',') if p.strip()]
                            if parts:
                                features = parts
                    ln_resid_std = row['ln_resid_std'] if 'ln_resid_std' in row and pd.notna(row['ln_resid_std']) else None
                    self.demand_models[category] = {
                        'model_object': model_obj,
                        'price_elasticity': row['price_elasticity'] if 'price_elasticity' in row else np.nan,
                        'exog_features': features, # Store the feature list actually used in training
                        'test_r2': row['test_r2'] if 'test_r2' in row else np.nan,
                        'ln_resid_std': ln_resid_std,
                        'model_type': row['model'] if 'model' in row else 'Unknown'
                    }
                except Exception as e:
                    print(f"  - Failed to load model for {category}: {e}")
            else:
                print(f"  - Model file not found for {category} at {model_path}")

        print(f"Successfully loaded {len(self.demand_models)} demand models.")
    
    def predict_demand(self, category, price, date, wholesale_cost=None):
        """Predict demand using the loaded model (RandomForest expects ln_quantity as target).
        To align with training features, this constructs key engineered features:
        - ln_price, ln_wholesale, ln_markup_ratio
        - time_trend, is_weekend, weekday dummies
        Other history-based features (lags/rollings) are set to neutral defaults.
        """
        if category not in self.demand_models:
            # Return a default baseline demand and high uncertainty if no model exists
            return 15.0, 15.0 * 0.5 

        model_info = self.demand_models[category]
        model = model_info['model_object']
        features = model_info['exog_features']

        # 1. Construct the exogenous feature vector (X)
        date = pd.to_datetime(date)
        exog_data = pd.DataFrame(index=[0])
        
        # Price-related features
        if 'ln_price' in features:
            exog_data['ln_price'] = float(np.log(max(price, 1e-6)))
        if 'ln_wholesale' in features:
            if wholesale_cost is None:
                # If not provided, approximate using price and an assumed markup
                approx_wholesale = max(price / max(self.opt_config.get('min_markup_ratio', 1.6), 1e-6), 1e-6)
                exog_data['ln_wholesale'] = float(np.log(approx_wholesale))
            else:
                exog_data['ln_wholesale'] = float(np.log(max(wholesale_cost, 1e-6)))
        if 'ln_markup_ratio' in features:
            w = wholesale_cost if wholesale_cost is not None else max(price / max(self.opt_config.get('min_markup_ratio', 1.6), 1e-6), 1e-6)
            exog_data['ln_markup_ratio'] = float(np.log(max(price / max(w, 1e-6), 1e-6)))
        if 'ln_relative_price' in features:
            # Use markup ratio as a proxy for relative price in absence of peers
            w = wholesale_cost if wholesale_cost is not None else max(price / max(self.opt_config.get('min_markup_ratio', 1.6), 1e-6), 1e-6)
            exog_data['ln_relative_price'] = float(np.log(max(price / max(w, 1e-6), 1e-6)))

        # Calendar features
        if 'is_weekend' in features:
            exog_data['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        if 'time_trend' in features:
            # Calculate days since the (training) start date and normalize with the same span as training
            days_since_start = max(0, (date - self.time_series_start_date).days)
            denom = max(1, getattr(self, 'time_trend_denominator', 365))
            exog_data['time_trend'] = min(days_since_start / denom, 1.0)
        # Weekday one-hot (weekday_0..weekday_6)
        for wd in range(7):
            col = f'weekday_{wd}'
            if col in features:
                exog_data[col] = 1 if date.weekday() == wd else 0

        # Ensure all required features exist; fill missing with 0.0 as a safe default
        missing_cols = [col for col in features if col not in exog_data.columns]
        for col in missing_cols:
            exog_data[col] = 0.0
        # Ensure the columns are in the same order as during training
        exog_data = exog_data[features]
        # Ensure numeric dtype
        exog_data = exog_data.astype(float)

        # 2. Predict using the model (RandomForest outputs ln_quantity prediction)
        try:
            ln_pred_val = float(model.predict(exog_data)[0])
            if not np.isfinite(ln_pred_val):
                raise ValueError(f"ln_pred not finite: {ln_pred_val}")
            # Blend in a negative elasticity prior to enforce monotonic relation with price
            ln_price_cur = float(np.log(max(price, 1e-6)))
            ln_price_ref = float(self.category_ln_price_ref.get(category, ln_price_cur))
            elasticity_prior = float(self.opt_config.get('demand_elasticity_prior', -1.0))  # should be negative
            blend = float(self.opt_config.get('elasticity_blend', 0.6))
            ln_pred_adj = ln_pred_val + blend * elasticity_prior * (ln_price_cur - ln_price_ref)
            predicted_demand = float(np.exp(ln_pred_adj))
            if not np.isfinite(predicted_demand) or predicted_demand <= 0:
                print(f"ERROR predicting for {category} on {date}: Invalid prediction {predicted_demand}")
                return 15.0, 15.0 * 0.5
        except Exception as e:
            print(f"ERROR predicting for {category} on {date}: {e}")
            return 15.0, 15.0 * 0.5

        # 3. Estimate uncertainty
        demand_std = None
        # Prefer residual std on ln-scale if available
        ln_sigma = model_info.get('ln_resid_std', None)
        if ln_sigma is not None and np.isfinite(ln_sigma) and ln_sigma > 0:
            demand_std = max(predicted_demand * float(ln_sigma), predicted_demand * 0.05)

        if demand_std is None or not np.isfinite(demand_std) or demand_std <= 0:
            # Fallback: derive a conservative std from test R^2
            uncertainty_factor = np.sqrt(1 - max(0, model_info.get('test_r2', 0.5)))
            demand_std = max(predicted_demand * uncertainty_factor * 0.5, predicted_demand * 0.05)
        
        # Cap demand to a reasonable historical quantile to avoid unrealistic spikes
        cap = self.category_demand_cap.get(category)
        if cap is not None and np.isfinite(cap) and cap > 0:
            predicted_demand = float(min(predicted_demand, cap * 1.1))
        return predicted_demand, demand_std
    
    def calculate_profit_scenarios(self, category, price, date, quantity, wholesale_cost, 
                                 n_scenarios=20, wastage_rate=None):
        """Calculates profit scenarios using model-based demand prediction.
        Improvements:
        - Use wholesale_cost in demand prediction for feature consistency.
        - Model stockout penalty and wastage (leftover) penalty explicitly.
        - Do not pre-reduce available stock by wastage; wastage applies to leftovers.
        Returns array of profit for each scenario.
        """
        if wastage_rate is None:
            wastage_rate = self.default_wastage_rate
        mean_demand, std_demand = self.predict_demand(category, price, date, wholesale_cost=wholesale_cost)
        
        if std_demand <= 0:
            std_demand = mean_demand * 0.1 # Assign a floor to uncertainty

        np.random.seed(hash(date) % (2**32 - 1)) # Seed with date for consistency
        demand_scenarios = np.random.normal(mean_demand, std_demand, n_scenarios)
        demand_scenarios = np.maximum(demand_scenarios, 0.0)
        
        profits = []
        for demand in demand_scenarios:
            available_quantity = max(quantity, 0.0)
            actual_sales = min(demand, available_quantity)
            revenue = actual_sales * price
            cost = quantity * wholesale_cost
            stockout = max(0.0, demand - actual_sales)
            leftover = max(0.0, available_quantity - actual_sales)
            stockout_penalty = stockout * wholesale_cost * self.opt_config.get('stockout_penalty_weight', 0.0)
            wastage_penalty = leftover * wholesale_cost * self.opt_config.get('wastage_penalty_weight', 0.0) * max(wastage_rate, 0.0)
            profit = revenue - cost - stockout_penalty - wastage_penalty
            profits.append(profit)
        
        return np.array(profits)

    def simulate_metrics(self, category, price, date, quantity, wholesale_cost, n_scenarios=None, wastage_rate=None):
        """Simulate demand to compute expected metrics used for reporting and optimization."""
        if n_scenarios is None:
            n_scenarios = int(self.opt_config.get('monte_carlo_samples', 50))
        if wastage_rate is None:
            wastage_rate = self.default_wastage_rate
        mean_demand, std_demand = self.predict_demand(category, price, date, wholesale_cost=wholesale_cost)
        if std_demand <= 0:
            std_demand = mean_demand * 0.1
        np.random.seed(hash(date) % (2**32 - 1))
        demand = np.maximum(np.random.normal(mean_demand, std_demand, n_scenarios), 0.0)
        available = max(quantity, 0.0)
        actual_sales = np.minimum(demand, available)
        revenue = actual_sales * price
        cost = available * wholesale_cost
        stockout = np.maximum(demand - actual_sales, 0.0)
        leftover = np.maximum(available - actual_sales, 0.0)
        stockout_penalty = stockout * wholesale_cost * self.opt_config.get('stockout_penalty_weight', 0.0)
        wastage_penalty = leftover * wholesale_cost * self.opt_config.get('wastage_penalty_weight', 0.0) * max(wastage_rate, 0.0)
        profit = revenue - cost - stockout_penalty - wastage_penalty
        # Aggregate metrics
        eps = 1e-9
        expected = {
            'expected_profit': float(np.mean(profit)),
            'profit_std': float(np.std(profit)),
            'expected_revenue': float(np.mean(revenue)),
            'expected_cost': float(cost),  # cost does not vary across scenarios here
            'expected_sales': float(np.mean(actual_sales)),
            'expected_leftover': float(np.mean(leftover)),
            'expected_stockout': float(np.mean(stockout)),
            'service_rate': float(np.mean(np.where(demand > eps, actual_sales / np.maximum(demand, eps), 1.0)))
        }
        return expected

    def evaluate_markup_profit(self, category, wholesale_cost, date, markup_ratio):
        """Evaluates the profit for a given markup ratio by predicting demand."""
        price = wholesale_cost * markup_ratio
        
        # Predict demand and determine order quantity with safety stock
        predicted_demand, std_demand = self.predict_demand(category, price, date, wholesale_cost=wholesale_cost)
        order_quantity = self.calculate_order_quantity(predicted_demand, std_demand)

        # Calculate expected profit using Monte Carlo simulation
        metrics = self.simulate_metrics(
            category, price, date, order_quantity, wholesale_cost,
            n_scenarios=self.opt_config['monte_carlo_samples']
        )
        
        expected_profit = metrics['expected_profit']
        profit_std = metrics['profit_std']
        
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

    def calculate_order_quantity(self, predicted_demand, std_demand, wastage_rate=None):
        """Calculates order quantity including safety stock based on service level.
        Enforce R*(1-W) \u003e= Q by scaling for wastage.
        """
        if wastage_rate is None:
            wastage_rate = self.default_wastage_rate
        service_level = self.opt_config['service_level']
        # Z-score for standard normal distribution
        z_score_map = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.98: 2.05}
        safety_factor = z_score_map.get(service_level, 1.65) # Default to 95%
        
        safety_stock = safety_factor * std_demand
        # Scale up to ensure net available after wastage meets expected demand
        net_required = predicted_demand + safety_stock
        order_quantity = net_required / max(1e-6, (1 - max(wastage_rate, 0.0)))
        return max(0.0, order_quantity) # Ensure non-negative

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

        predicted_demand, std_demand = self.predict_demand(category, optimal_price, date, wholesale_cost=wholesale_cost)
        optimal_quantity = self.calculate_order_quantity(predicted_demand, std_demand)
        
        return optimal_price, optimal_quantity

    def generate_wholesale_forecasts(self, categories):
        """Generates simplified wholesale price forecasts for the given categories."""
        forecasts = {}
        horizon = self.opt_config['optimization_horizon']
        base_prices = self.config['base_wholesale_prices']

        # Deterministic pseudo-forecast for reproducibility
        rng = np.random.default_rng(20230701)
        for category in categories:
            base_price = base_prices.get(category, 5.0) # Default price if not in config
            days = np.arange(horizon)
            seasonal_effect = 0.1 * np.sin(2 * np.pi * days / 7) # Weekly seasonality
            noise = rng.normal(0, 0.02, horizon)
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
        # Use a fixed target week start date per 23C requirement (2023-07-01), overridable via config
        cfg_start = self.config.get('target_week_start_date', '2023-07-01')
        try:
            start_date = pd.to_datetime(cfg_start).date()
        except Exception:
            start_date = datetime(2023, 7, 1).date()
        
        for day in range(horizon):
            current_date = start_date + timedelta(days=day)
            print(f"\n-- Optimizing for Date: {current_date.strftime('%Y-%m-%d')} --")
            
            for category in categories_with_models:
                wholesale_cost = wholesale_forecasts[category][day]
                
                optimal_price, optimal_quantity = self.optimize_category(
                    category, wholesale_cost, current_date
                )
                
                # Recalculate expected profit and extended metrics for reporting
                metrics = self.simulate_metrics(
                    category, optimal_price, current_date, optimal_quantity, wholesale_cost,
                    n_scenarios=self.opt_config['monte_carlo_samples']
                )
                markup_ratio = float(optimal_price / max(wholesale_cost, 1e-6))
                gross_margin = float((optimal_price - wholesale_cost) / max(optimal_price, 1e-6))
                
                results.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'category': category,
                    'wholesale_cost': wholesale_cost,
                    'optimal_price': optimal_price,
                    'optimal_quantity': optimal_quantity,
                    'markup_ratio': markup_ratio,
                    'gross_margin': gross_margin,
                    'expected_profit': metrics['expected_profit'],
                    'profit_std': metrics['profit_std'],
                    'expected_revenue': metrics['expected_revenue'],
                    'expected_cost': metrics['expected_cost'],
                    'expected_sales': metrics['expected_sales'],
                    'expected_leftover': metrics['expected_leftover'],
                    'expected_stockout': metrics['expected_stockout'],
                    'service_rate': metrics['service_rate']
                })
                
                print(f"  - {category}: Cost={wholesale_cost:.2f}, Price={optimal_price:.2f}, Quantity={optimal_quantity:.2f}, Profit={metrics['expected_profit']:.2f}")

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
            predicted_demand, std_demand = self.predict_demand(category, price, date, wholesale_cost=wholesale_cost)
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
