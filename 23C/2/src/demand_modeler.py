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
        # 加载配置
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        with open(os.path.join(project_root, config_path), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 定义路径
        self.train_features_path = os.path.join(project_root, self.config['data_paths']['train_features'])
        self.test_features_path = os.path.join(project_root, self.config['data_paths']['test_features'])
        self.output_paths = {
            'results_dir': os.path.join(project_root, self.config['output_paths']['results_dir']),
            'model_dir': os.path.join(project_root, self.config['output_paths']['model_dir'])
        }

        # 确保输出目录存在
        os.makedirs(self.output_paths['results_dir'], exist_ok=True)
        os.makedirs(self.output_paths['model_dir'], exist_ok=True)

    def fit_random_forest_model(self, train_data, test_data, category):

        print(f"Processing category: {category}")
        
        cat_train = train_data[train_data['category'] == category].copy()
        cat_test = test_data[test_data['category'] == category].copy()
        
        if cat_train.empty or cat_test.empty:
            print(f"  - Skipping {category} due to insufficient data.")
            return None
        
        # 从ln_quantity还原数量，并聚合到类别-日期
        for df in (cat_train, cat_test):
            df['quantity'] = np.exp(df['ln_quantity']).clip(lower=1e-6)
            # 在类别-日期层面聚合关键特征
            agg_funcs = {
                'quantity': 'sum',
                # 将ln_price, ln_wholesale还原为价格水平，取均值后再取对数
                'ln_price': lambda x: np.log(np.mean(np.exp(x))),
                'ln_wholesale': lambda x: np.log(np.mean(np.exp(x))) if x.notna().any() else np.nan,
                'ln_markup_ratio': lambda x: np.log(np.mean(np.exp(x))) if x.notna().any() else np.nan,
                'time_trend': 'mean',
                'is_weekend': 'first',
            }
            # 如果存在星期几虚拟变量
            weekday_cols = [c for c in df.columns if c.startswith('weekday_')]
            for c in weekday_cols:
                agg_funcs[c] = 'first'
            # 相对价格在均值后约为1（ln约为0）；在ln空间取均值
            if 'ln_relative_price' in df.columns:
                agg_funcs['ln_relative_price'] = 'mean'
            df_agg = df.groupby(['date', 'category'], as_index=False).agg(agg_funcs)
            # 从聚合后的数量重新计算ln_quantity
            df_agg['ln_quantity'] = np.log(df_agg['quantity'].clip(lower=1e-6))
            # 存回
            if df is cat_train:
                category_train = df_agg.set_index('date').sort_index()
            else:
                category_test = df_agg.set_index('date').sort_index()
        
        # 目标变量和特征
        target = self.config['target_variable']  # 'ln_quantity'
        requested_features = self.config.get('tree_features', self.config.get('sarimax_exog_features', []))
        # 只保留聚合后仍然存在的特征
        feature_list = [f for f in requested_features if f in category_train.columns]
        if len(feature_list) == 0:
            print(f"  - WARNING: No usable features for {category} after aggregation. Skipping.")
            return None

        # 使测试集与训练集特征一致
        missing_in_test = [f for f in feature_list if f not in category_test.columns]
        for f in missing_in_test:
            category_test[f] = 0.0
        category_test = category_test[feature_list + [target]]

        y_train = category_train[target].astype(float)
        X_train = category_train[feature_list].astype(float)
        y_test = category_test[target].astype(float)
        X_test = category_test[feature_list].astype(float)

        # 用指定参数训练随机森林
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=self.config.get('modeling', {}).get('random_state', 42),
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # 评估模型
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        # 测试集ln残差标准差
        ln_resid_std = float(np.std(y_test - y_pred_test)) if len(y_pred_test) > 1 else float('nan')

        # 保存模型
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
        """对所有类别运行建模流程并保存结果。"""
        print("正在加载已划分好的训练和测试特征数据...")
        train_data = pd.read_csv(self.train_features_path, index_col='销售日期', parse_dates=['销售日期'])
        test_data = pd.read_csv(self.test_features_path, index_col='销售日期', parse_dates=['销售日期'])
        
        # 重置索引，使日期成为列，便于筛选
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()
        
        # 重命名为标准列名
        train_data = train_data.rename(columns={'销售日期': 'date', '分类名称': 'category'})
        test_data = test_data.rename(columns={'销售日期': 'date', '分类名称': 'category'})
        
        # 确保必须的基础列存在
        required_cols = ['date', 'category', 'ln_quantity', 'ln_price']
        for col in required_cols:
            if col not in train_data.columns or col not in test_data.columns:
                print(f"ERROR: Required column missing for modeling: {col}")
        
        all_results = []
        for category in train_data['category'].unique():
            category_result = self.fit_random_forest_model(train_data, test_data, category)
            if category_result is not None:
                all_results.append(category_result)
        
        # 合并并保存结果
        if all_results:
            final_results_df = pd.DataFrame(all_results)
            output_file = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            final_results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n建模完成，结果已保存至 {output_file}")
        else:
            print("\n建模结束，但未生成任何结果。")

if __name__ == '__main__':
    modeler = DemandModeler()