# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import r2_score, mean_absolute_error
import os
import sys
import json
import joblib

class DemandModeler:
    def __init__(self, config_path='config/config.json'):
        # Load configuration
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        with open(os.path.join(project_root, config_path), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Define paths
        self.train_features_path = os.path.join(project_root, self.config['data_paths']['train_features'])
        self.test_features_path = os.path.join(project_root, self.config['data_paths']['test_features'])
        self.output_paths = {
            'results_dir': os.path.join(project_root, self.config['output_paths']['results_dir']),
            'model_dir': os.path.join(project_root, self.config['output_paths']['model_dir'])
        }

        # Ensure output directories exist
        os.makedirs(self.output_paths['results_dir'], exist_ok=True)
        os.makedirs(self.output_paths['model_dir'], exist_ok=True)

    def fit_sarimax_model(self, train_data, test_data, category):
        """Fits a SARIMAX model for a single category using auto_arima."""
        print(f"Processing category: {category}")
        
        category_train = train_data[train_data['category'] == category].copy()
        category_test = test_data[test_data['category'] == category].copy()
        
        # Remove duplicates and set date as index
        category_train = category_train.drop_duplicates(subset=['date']).set_index('date').sort_index()
        category_test = category_test.drop_duplicates(subset=['date']).set_index('date').sort_index()
        
        if category_train.empty or category_test.empty:
            print(f"  - Skipping {category} due to insufficient data.")
            return None

        # Define target (y) and exogenous variables (X)
        target = self.config['target_variable'] # 'ln_quantity'
        exog_features = self.config['sarimax_exog_features'] # e.g., ['ln_price', 'is_weekend', 'time_trend']
        
        y_train = category_train[target]
        X_train = category_train[exog_features]
        y_test = category_test[target]
        X_test = category_test[exog_features]

        # Use auto_arima to find the best SARIMAX model
        print("  - Running auto_arima to find best SARIMAX parameters...")
        try:
            sarimax_model = pm.auto_arima(
                y_train, 
                exogenous=X_train,
                start_p=1, start_q=1,
                test='adf',       # use adftest to find optimal 'd'
                max_p=3, max_q=3, # maximum p and q
                m=7,              # weekly seasonality
                d=None,           # let model determine 'd'
                seasonal=True,    # enforce seasonality
                start_P=0, 
                D=1, 
                trace=True,
                error_action='ignore',  
                suppress_warnings=True, 
                stepwise=True
            )
        except Exception as e:
            print(f"  - ERROR: auto_arima failed for {category}: {e}")
            return None

        print(f"  - Best model found: {sarimax_model.order}, {sarimax_model.seasonal_order}")

        # Evaluate the model on the test set
        y_pred_test, conf_int = sarimax_model.predict(n_periods=len(y_test), exogenous=X_test, return_conf_int=True)
        
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Extract price elasticity (coefficient of ln_price)
        # +1 to account for intercept in pmdarima's result
        price_elasticity = sarimax_model.params()[X_train.columns.get_loc('ln_price') + 1] 

        # Save the fitted model
        model_path = os.path.join(self.output_paths['model_dir'], f'{category}_sarimax.pkl')
        joblib.dump(sarimax_model, model_path)
        
        print(f"  - Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}")
        print(f"  - Price Elasticity: {price_elasticity:.4f}")
        print(f"  - Model saved to: {model_path}")

        return {
            'category': category,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'price_elasticity': price_elasticity,
            'model_order': str(sarimax_model.order),
            'seasonal_order': str(sarimax_model.seasonal_order),
            'model_path': model_path
        }

    def run_modeling(self):
        """Runs the modeling pipeline for all categories and saves the results."""
        print("Loading pre-split training and testing feature data...")
        train_data = pd.read_csv(self.train_features_path, index_col='销售日期', parse_dates=['销售日期'])
        test_data = pd.read_csv(self.test_features_path, index_col='销售日期', parse_dates=['销售日期'])
        
        # Reset index to make date a column for easier filtering
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()
        
        # Rename to standard column name
        train_data = train_data.rename(columns={'销售日期': 'date', '分类名称': 'category'})
        test_data = test_data.rename(columns={'销售日期': 'date', '分类名称': 'category'})
        
        all_results = []
        for category in train_data['category'].unique():
            category_result = self.fit_sarimax_model(train_data, test_data, category)
            if category_result is not None:
                all_results.append(category_result)
        
        # Concatenate and save results
        if all_results:
            final_results_df = pd.DataFrame(all_results)
            output_file = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            final_results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\nModeling complete. Results saved to {output_file}")
        else:
            print("\nModeling finished, but no results were generated.")

if __name__ == '__main__':
    modeler = DemandModeler()
    modeler.run_modeling()