# -*- coding: utf-8 -*-
"""
增强需求建模模块
包含内生性修正、鲁棒回归、分层模型等多种方法
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedDemandModeler:
    """增强需求建模器"""
    
    def __init__(self, train_path='train_features.csv', test_path='test_features.csv'):
        """初始化建模器"""
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """加载特征数据"""
        print("加载特征数据...")
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        print(f"训练数据: {len(self.train_data):,} 条记录")
        print(f"测试数据: {len(self.test_data):,} 条记录")
        return self
        
    def prepare_modeling_data(self, category=None):
        """准备建模数据"""
        if category:
            train_cat = self.train_data[self.train_data['分类名称'] == category].copy()
            test_cat = self.test_data[self.test_data['分类名称'] == category].copy()
        else:
            train_cat = self.train_data.copy()
            test_cat = self.test_data.copy()
            
        # 移除非数值列和目标变量
        exclude_cols = ['单品编码', '单品名称', '分类名称', '销售日期', 'ln_quantity']
        feature_cols = [col for col in train_cat.columns if col not in exclude_cols]
        
        # 过滤无效值
        train_valid = train_cat.dropna(subset=['ln_quantity'] + feature_cols)
        test_valid = test_cat.dropna(subset=['ln_quantity'] + feature_cols)
        
        if len(train_valid) < 10 or len(test_valid) < 5:
            return None, None, None, None, None
            
        X_train = train_valid[feature_cols]
        y_train = train_valid['ln_quantity']
        X_test = test_valid[feature_cols]
        y_test = test_valid['ln_quantity']
        
        return X_train, y_train, X_test, y_test, feature_cols
        
    def fit_ols_baseline(self, X_train, y_train, X_test, y_test):
        """OLS基线模型"""
        try:
            # 添加常数项
            X_train_const = add_constant(X_train)
            X_test_const = add_constant(X_test)
            
            # 拟合模型
            model = OLS(y_train, X_train_const).fit()
            
            # 预测
            y_train_pred = model.predict(X_train_const)
            y_test_pred = model.predict(X_test_const)
            
            # 计算指标
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # 异方差检验
            _, het_p_value, _, _ = het_breuschpagan(model.resid, X_train_const)
            
            # 价格弹性（假设ln_price是第一个特征）
            price_elasticity = None
            price_elasticity_pvalue = None
            if 'ln_price' in X_train.columns:
                price_idx = list(X_train.columns).index('ln_price') + 1  # +1因为常数项
                price_elasticity = model.params.iloc[price_idx]
                price_elasticity_pvalue = model.pvalues.iloc[price_idx]
            
            return {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'price_elasticity': price_elasticity,
                'price_elasticity_pvalue': price_elasticity_pvalue,
                'heteroscedasticity_p': het_p_value,
                'aic': model.aic,
                'bic': model.bic
            }
        except Exception as e:
            print(f"OLS拟合失败: {e}")
            return None
            
    def fit_robust_regression(self, X_train, y_train, X_test, y_test):
        """鲁棒回归（Huber）"""
        try:
            model = HuberRegressor(epsilon=1.35, max_iter=200)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # 价格弹性
            price_elasticity = None
            if 'ln_price' in X_train.columns:
                price_idx = list(X_train.columns).index('ln_price')
                price_elasticity = model.coef_[price_idx]
            
            return {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'price_elasticity': price_elasticity,
                'n_outliers': len(model.outliers_) if hasattr(model, 'outliers_') else None
            }
        except Exception as e:
            print(f"鲁棒回归拟合失败: {e}")
            return None
            
    def fit_instrumental_variables(self, X_train, y_train, X_test, y_test):
        """工具变量法（2SLS）"""
        try:
            # 检查是否有合适的工具变量
            potential_instruments = ['ln_wholesale', 'ln_price_lag1', 'price_lag7']
            available_instruments = [iv for iv in potential_instruments if iv in X_train.columns]
            
            if len(available_instruments) < 1 or 'ln_price' not in X_train.columns:
                return None
                
            # 准备内生变量和工具变量
            endog = X_train['ln_price'].values
            instruments = X_train[available_instruments].values
            
            # 其他外生变量
            exog_vars = [col for col in X_train.columns if col not in ['ln_price'] + available_instruments]
            if len(exog_vars) > 0:
                exog = add_constant(X_train[exog_vars])
            else:
                exog = add_constant(pd.DataFrame(index=X_train.index))
            
            # 2SLS估计
            model = IV2SLS(y_train, exog, endog.reshape(-1, 1), instruments).fit()
            
            # 第一阶段F统计量
            first_stage = LinearRegression()
            first_stage.fit(np.column_stack([exog, instruments]), endog)
            first_stage_pred = first_stage.predict(np.column_stack([exog, instruments]))
            first_stage_r2 = r2_score(endog, first_stage_pred)
            
            # 简化的F统计量计算
            n, k = len(endog), len(available_instruments)
            first_stage_f = (first_stage_r2 / (1 - first_stage_r2)) * ((n - k - 1) / k)
            
            # 预测（简化版）
            price_coef = model.params[-1]  # 价格系数
            
            return {
                'model': model,
                'price_elasticity': price_coef,
                'first_stage_f': first_stage_f,
                'instruments_used': available_instruments,
                'weak_instruments': first_stage_f < 10
            }
        except Exception as e:
            print(f"2SLS拟合失败: {e}")
            return None
            
    def fit_random_forest(self, X_train, y_train, X_test, y_test):
        """随机森林回归"""
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # 特征重要性
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': feature_importance
            }
        except Exception as e:
            print(f"随机森林拟合失败: {e}")
            return None
            
    def fit_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """梯度提升回归"""
        try:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # 特征重要性
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            return {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': feature_importance
            }
        except Exception as e:
            print(f"梯度提升拟合失败: {e}")
            return None
            
    def cross_validate_models(self, X, y, n_splits=3):
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}
        
        models_to_test = {
            'OLS': lambda: LinearRegression(),
            'Huber': lambda: HuberRegressor(epsilon=1.35),
            'RandomForest': lambda: RandomForestRegressor(n_estimators=50, random_state=42),
            'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        for model_name, model_func in models_to_test.items():
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    model = model_func()
                    model.fit(X_train_cv, y_train_cv)
                    y_pred = model.predict(X_val_cv)
                    score = r2_score(y_val_cv, y_pred)
                    cv_scores.append(score)
                except:
                    continue
            
            if cv_scores:
                cv_results[model_name] = {
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores),
                    'cv_scores': cv_scores
                }
        
        return cv_results
        
    def model_category(self, category):
        """为单个品类建模"""
        print(f"\n--- 建模品类: {category} ---")
        
        # 准备数据
        X_train, y_train, X_test, y_test, feature_cols = self.prepare_modeling_data(category)
        
        if X_train is None:
            print(f"品类 {category} 数据不足，跳过")
            return None
            
        print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        print(f"特征数量: {len(feature_cols)}")
        
        category_results = {}
        
        # 1. OLS基线
        print("拟合OLS基线...")
        ols_result = self.fit_ols_baseline(X_train, y_train, X_test, y_test)
        if ols_result:
            category_results['OLS'] = ols_result
            print(f"  训练R²: {ols_result['train_r2']:.4f}, 测试R²: {ols_result['test_r2']:.4f}")
            if ols_result['price_elasticity']:
                print(f"  价格弹性: {ols_result['price_elasticity']:.4f}")
        
        # 2. 鲁棒回归
        print("拟合鲁棒回归...")
        robust_result = self.fit_robust_regression(X_train, y_train, X_test, y_test)
        if robust_result:
            category_results['Huber'] = robust_result
            print(f"  训练R²: {robust_result['train_r2']:.4f}, 测试R²: {robust_result['test_r2']:.4f}")
            if robust_result['price_elasticity']:
                print(f"  价格弹性: {robust_result['price_elasticity']:.4f}")
        
        # 3. 工具变量法
        print("拟合2SLS...")
        iv_result = self.fit_instrumental_variables(X_train, y_train, X_test, y_test)
        if iv_result:
            category_results['2SLS'] = iv_result
            print(f"  价格弹性: {iv_result['price_elasticity']:.4f}")
            print(f"  第一阶段F: {iv_result['first_stage_f']:.2f}")
            print(f"  工具变量: {iv_result['instruments_used']}")
        
        # 4. 随机森林
        print("拟合随机森林...")
        rf_result = self.fit_random_forest(X_train, y_train, X_test, y_test)
        if rf_result:
            category_results['RandomForest'] = rf_result
            print(f"  训练R²: {rf_result['train_r2']:.4f}, 测试R²: {rf_result['test_r2']:.4f}")
        
        # 5. 梯度提升
        print("拟合梯度提升...")
        gb_result = self.fit_gradient_boosting(X_train, y_train, X_test, y_test)
        if gb_result:
            category_results['GradientBoosting'] = gb_result
            print(f"  训练R²: {gb_result['train_r2']:.4f}, 测试R²: {gb_result['test_r2']:.4f}")
        
        # 6. 交叉验证
        print("进行交叉验证...")
        cv_results = self.cross_validate_models(X_train, y_train)
        category_results['CrossValidation'] = cv_results
        
        for model_name, cv_result in cv_results.items():
            print(f"  {model_name} CV R²: {cv_result['mean_cv_score']:.4f} ± {cv_result['std_cv_score']:.4f}")
        
        return category_results
        
    def run_enhanced_modeling(self):
        """运行增强建模流程"""
        print("开始增强需求建模...")
        
        self.load_data()
        
        # 获取所有品类
        categories = self.train_data['分类名称'].unique()
        print(f"待建模品类: {list(categories)}")
        
        # 为每个品类建模
        for category in categories:
            category_results = self.model_category(category)
            if category_results:
                self.results[category] = category_results
        
        # 生成结果报告
        self.generate_modeling_report()
        self.save_results()
        
        print("\n增强需求建模完成！")
        return self
        
    def generate_modeling_report(self):
        """生成建模报告"""
        print("\n生成建模报告...")
        
        report_content = []
        report_content.append("# 增强需求建模报告")
        report_content.append("")
        
        # 模型性能汇总
        report_content.append("## 模型性能汇总")
        report_content.append("")
        
        performance_summary = []
        
        for category, results in self.results.items():
            for model_name, result in results.items():
                if model_name == 'CrossValidation':
                    continue
                    
                row = {
                    'Category': category,
                    'Model': model_name,
                    'Train_R2': result.get('train_r2', None),
                    'Test_R2': result.get('test_r2', None),
                    'Price_Elasticity': result.get('price_elasticity', None)
                }
                performance_summary.append(row)
        
        if performance_summary:
            summary_df = pd.DataFrame(performance_summary)
            
            # 按品类汇总最佳模型
            report_content.append("### 各品类最佳模型（按测试R²）")
            for category in summary_df['Category'].unique():
                cat_results = summary_df[summary_df['Category'] == category]
                cat_results = cat_results.dropna(subset=['Test_R2'])
                
                if len(cat_results) > 0:
                    best_model = cat_results.loc[cat_results['Test_R2'].idxmax()]
                    report_content.append(f"- **{category}**: {best_model['Model']} (测试R²={best_model['Test_R2']:.4f})")
                    
                    if best_model['Price_Elasticity'] is not None:
                        report_content.append(f"  - 价格弹性: {best_model['Price_Elasticity']:.4f}")
        
        report_content.append("")
        
        # 方法论总结
        report_content.append("## 方法论总结")
        report_content.append("- **OLS**: 普通最小二乘法基线")
        report_content.append("- **Huber**: 鲁棒回归，减少异常值影响")
        report_content.append("- **2SLS**: 工具变量法，处理价格内生性")
        report_content.append("- **RandomForest**: 随机森林，捕捉非线性关系")
        report_content.append("- **GradientBoosting**: 梯度提升，强预测性能")
        
        # 保存报告
        report_text = "\n".join(report_content)
        with open('enhanced_modeling_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print("建模报告已保存: enhanced_modeling_report.md")
        
    def save_results(self):
        """保存建模结果"""
        print("保存建模结果...")
        
        # 转换为DataFrame格式
        results_list = []
        
        for category, results in self.results.items():
            for model_name, result in results.items():
                if model_name == 'CrossValidation':
                    continue
                    
                result_row = {
                    'category': category,
                    'model': model_name,
                    'train_r2': result.get('train_r2'),
                    'test_r2': result.get('test_r2'),
                    'train_mae': result.get('train_mae'),
                    'test_mae': result.get('test_mae'),
                    'price_elasticity': result.get('price_elasticity'),
                    'price_elasticity_pvalue': result.get('price_elasticity_pvalue'),
                    'heteroscedasticity_p': result.get('heteroscedasticity_p'),
                    'aic': result.get('aic'),
                    'bic': result.get('bic'),
                    'first_stage_f': result.get('first_stage_f'),
                    'weak_instruments': result.get('weak_instruments'),
                    'n_outliers': result.get('n_outliers')
                }
                results_list.append(result_row)
        
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv('enhanced_demand_model_results.csv', index=False, encoding='utf-8')
            print("结果已保存: enhanced_demand_model_results.csv")
        
        # 保存交叉验证结果
        cv_results_list = []
        for category, results in self.results.items():
            if 'CrossValidation' in results:
                for model_name, cv_result in results['CrossValidation'].items():
                    cv_row = {
                        'category': category,
                        'model': model_name,
                        'mean_cv_score': cv_result['mean_cv_score'],
                        'std_cv_score': cv_result['std_cv_score']
                    }
                    cv_results_list.append(cv_row)
        
        if cv_results_list:
            cv_df = pd.DataFrame(cv_results_list)
            cv_df.to_csv('cross_validation_results.csv', index=False, encoding='utf-8')
            print("交叉验证结果已保存: cross_validation_results.csv")

if __name__ == "__main__":
    modeler = EnhancedDemandModeler()
    modeler.run_enhanced_modeling()
