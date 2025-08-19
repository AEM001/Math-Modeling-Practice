# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

    def fit_random_forest_model(self, train_data, test_data, category):
        """Fit a RandomForest model for one category with fixed hyperparameters.
        Aggregate item-level records to category-by-date to match the task requirement.
        """
        print(f"Processing category: {category}")
        
        cat_train = train_data[train_data['category'] == category].copy()
        cat_test = test_data[test_data['category'] == category].copy()
        
        if cat_train.empty or cat_test.empty:
            print(f"  - Skipping {category} due to insufficient data.")
            return None
        
        # Reconstruct quantity from ln_quantity and aggregate to category-date
        for df in (cat_train, cat_test):
            df['quantity'] = np.exp(df['ln_quantity']).clip(lower=1e-6)
            # Aggregate key features at category-date level
            agg_funcs = {
                'quantity': 'sum',
                # Convert ln_price, ln_wholesale back to price level, average, then take log
                'ln_price': lambda x: np.log(np.mean(np.exp(x))),
                'ln_wholesale': lambda x: np.log(np.mean(np.exp(x))) if x.notna().any() else np.nan,
                'ln_markup_ratio': lambda x: np.log(np.mean(np.exp(x))) if x.notna().any() else np.nan,
                'time_trend': 'mean',
                'is_weekend': 'first',
            }
            # Weekday dummies if exist
            weekday_cols = [c for c in df.columns if c.startswith('weekday_')]
            for c in weekday_cols:
                agg_funcs[c] = 'first'
            # Relative price tends to average to ~1 (ln->0); take mean in ln-space
            if 'ln_relative_price' in df.columns:
                agg_funcs['ln_relative_price'] = 'mean'
            df_agg = df.groupby(['date', 'category'], as_index=False).agg(agg_funcs)
            # Recompute ln_quantity from aggregated quantity
            df_agg['ln_quantity'] = np.log(df_agg['quantity'].clip(lower=1e-6))
            # Store back
            if df is cat_train:
                category_train = df_agg.set_index('date').sort_index()
            else:
                category_test = df_agg.set_index('date').sort_index()
        
        # Target and features
        target = self.config['target_variable']  # 'ln_quantity'
        requested_features = self.config.get('tree_features', self.config.get('sarimax_exog_features', []))
        # Only keep features available after aggregation
        feature_list = [f for f in requested_features if f in category_train.columns]
        if len(feature_list) == 0:
            print(f"  - WARNING: No usable features for {category} after aggregation. Skipping.")
            return None

        # Align test set to same columns
        missing_in_test = [f for f in feature_list if f not in category_test.columns]
        for f in missing_in_test:
            category_test[f] = 0.0
        category_test = category_test[feature_list + [target]]

        y_train = category_train[target].astype(float)
        X_train = category_train[feature_list].astype(float)
        y_test = category_test[target].astype(float)
        X_test = category_test[feature_list].astype(float)

        # Train RandomForest with requested params
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=self.config.get('modeling', {}).get('random_state', 42),
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Evaluate
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        # Residual std on ln scale (test set)
        ln_resid_std = float(np.std(y_test - y_pred_test)) if len(y_pred_test) > 1 else float('nan')

        # Save model
        model_path = os.path.join(self.output_paths['model_dir'], f'{category}_rf.pkl')
        joblib.dump(rf, model_path)
        
        print(f"  - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}")
        print(f"  - Model saved to: {model_path}")

        return {
            'category': category,
            'model': 'RandomForest',
            'features': json.dumps(feature_list, ensure_ascii=False),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'price_elasticity': np.nan,
            'ln_resid_std': ln_resid_std,
            'n_estimators': 50,
            'max_depth': 8,
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
        
        # Ensure required basic columns exist
        required_cols = ['date', 'category', 'ln_quantity', 'ln_price']
        for col in required_cols:
            if col not in train_data.columns or col not in test_data.columns:
                print(f"ERROR: Required column missing for modeling: {col}")
        
        all_results = []
        for category in train_data['category'].unique():
            category_result = self.fit_random_forest_model(train_data, test_data, category)
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