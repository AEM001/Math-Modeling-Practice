# -*- coding: utf-8 -*-
"""
精简版需求建模模块
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 导入字体配置
from font_config import setup_chinese_font

class DemandModeler:
    """精简版需求建模器"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化建模器"""
        self.config = self.load_config(config_path)
        self.data_paths = self.config['data_paths']
        self.output_paths = self.config['output_paths']
        self.modeling_config = self.config['modeling']
        self.results = {}
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_feature_data(self):
        """加载特征数据"""
        print("加载特征数据...")
        
        train_data = pd.read_csv(self.data_paths['train_features'])
        test_data = pd.read_csv(self.data_paths['test_features'])
        
        print(f"训练数据: {len(train_data):,} 条记录")
        print(f"测试数据: {len(test_data):,} 条记录")
        
        return train_data, test_data
    
    def prepare_modeling_data(self, train_data, test_data, category=None):
        """准备建模数据"""
        if category:
            train_cat = train_data[train_data['分类名称'] == category].copy()
            test_cat = test_data[test_data['分类名称'] == category].copy()
        else:
            train_cat = train_data.copy()
            test_cat = test_data.copy()
        
        # 排除非特征列
        exclude_cols = ['单品编码', '单品名称', '分类名称', '销售日期', 'ln_quantity']
        feature_cols = [col for col in train_cat.columns if col not in exclude_cols]
        
        # 过滤有效数据
        train_valid = train_cat.dropna(subset=['ln_quantity'] + feature_cols)
        test_valid = test_cat.dropna(subset=['ln_quantity'] + feature_cols)
        
        if len(train_valid) < 20 or len(test_valid) < 5:
            return None, None, None, None, None
        
        X_train = train_valid[feature_cols]
        y_train = train_valid['ln_quantity']
        X_test = test_valid[feature_cols]
        y_test = test_valid['ln_quantity']
        
        return X_train, y_train, X_test, y_test, feature_cols
    
    def fit_models(self, X_train, y_train, X_test, y_test):
        """拟合多个模型"""
        models_config = {
            'LinearRegression': LinearRegression(),
            'HuberRegressor': HuberRegressor(epsilon=1.35, max_iter=200),
            'RandomForest': RandomForestRegressor(
                n_estimators=50, max_depth=8, min_samples_split=10, 
                min_samples_leaf=5, random_state=self.modeling_config['random_state']
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=50, learning_rate=0.1, max_depth=6,
                min_samples_split=10, min_samples_leaf=5, 
                random_state=self.modeling_config['random_state']
            )
        }
        
        model_results = {}
        
        for model_name, model in models_config.items():
            if model_name not in self.modeling_config['models_to_test']:
                continue
                
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # 计算指标
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # 价格弹性（针对线性模型）
                price_elasticity = None
                if hasattr(model, 'coef_') and 'ln_price' in X_train.columns:
                    price_idx = list(X_train.columns).index('ln_price')
                    price_elasticity = model.coef_[price_idx]
                
                model_results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'price_elasticity': price_elasticity
                }
                
            except Exception as e:
                print(f"模型 {model_name} 训练失败: {e}")
                continue
        
        return model_results
    
    def cross_validate_model(self, X, y, model, cv_folds=3):
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                model.fit(X_train_cv, y_train_cv)
                y_pred = model.predict(X_val_cv)
                score = r2_score(y_val_cv, y_pred)
                cv_scores.append(score)
            except:
                continue
        
        if cv_scores:
            return {
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'cv_scores': cv_scores
            }
        else:
            return None
    
    def model_category(self, train_data, test_data, category):
        """为单个品类建模"""
        print(f"\n--- 建模品类: {category} ---")
        
        # 准备数据
        X_train, y_train, X_test, y_test, feature_cols = self.prepare_modeling_data(
            train_data, test_data, category
        )
        
        if X_train is None:
            print(f"品类 {category} 数据不足，跳过")
            return None
        
        print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        print(f"特征数量: {len(feature_cols)}")
        
        # 拟合模型
        model_results = self.fit_models(X_train, y_train, X_test, y_test)
        
        if not model_results:
            print("所有模型训练失败")
            return None
        
        # 交叉验证最佳模型
        best_model_name = max(model_results.keys(), 
                             key=lambda k: model_results[k]['test_r2'])
        best_model = model_results[best_model_name]['model']
        
        cv_result = self.cross_validate_model(
            X_train, y_train, type(best_model)(), self.modeling_config['cv_folds']
        )
        
        # 输出结果
        for model_name, result in model_results.items():
            print(f"  {model_name}: 训练R²={result['train_r2']:.4f}, "
                  f"测试R²={result['test_r2']:.4f}")
            if result['price_elasticity']:
                print(f"    价格弹性: {result['price_elasticity']:.4f}")
        
        print(f"  最佳模型: {best_model_name}")
        if cv_result:
            print(f"  交叉验证R²: {cv_result['mean_cv_score']:.4f} ± {cv_result['std_cv_score']:.4f}")
        
        # 保存结果
        category_results = model_results.copy()
        category_results['best_model'] = best_model_name
        category_results['cross_validation'] = cv_result
        
        return category_results
    
    def run_modeling(self):
        """运行建模流程"""
        print("=== 开始需求建模 ===")
        
        # 加载数据
        train_data, test_data = self.load_feature_data()
        
        # 获取品类列表
        categories = train_data['分类名称'].unique()
        print(f"待建模品类: {list(categories)}")
        
        # 为每个品类建模
        for category in categories:
            category_results = self.model_category(train_data, test_data, category)
            if category_results:
                self.results[category] = category_results
        
        # 保存结果
        self.save_results()
        self.generate_report()
        
        print("=== 需求建模完成 ===\n")
        return True
    
    def save_results(self):
        """保存建模结果"""
        print("保存建模结果...")
        
        results_list = []
        for category, results in self.results.items():
            for model_name, result in results.items():
                if model_name in ['best_model', 'cross_validation']:
                    continue
                
                result_row = {
                    'category': category,
                    'model': model_name,
                    'train_r2': result.get('train_r2'),
                    'test_r2': result.get('test_r2'),
                    'train_mae': result.get('train_mae'),
                    'test_mae': result.get('test_mae'),
                    'price_elasticity': result.get('price_elasticity'),
                    'is_best': model_name == results.get('best_model', '')
                }
                results_list.append(result_row)
        
        if results_list:
            results_df = pd.DataFrame(results_list)
            
            # 确保输出目录存在
            os.makedirs(self.output_paths['results_dir'], exist_ok=True)
            output_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"建模结果已保存: {output_path}")
    
    def plot_price_quantity_relationship(self, train_data, test_data):
        """绘制价格-销量关系分析图"""
        try:
            # 设置中文字体
            setup_chinese_font()
            
            # 合并训练和测试数据
            all_data = pd.concat([train_data, test_data], ignore_index=True)
            categories = all_data['分类名称'].unique()
            
            # 创建子图
            n_categories = len(categories)
            cols = 3
            rows = (n_categories + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle('各品类价格-销量关系分析', fontsize=16, fontweight='bold')
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, category in enumerate(categories):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                # 获取品类数据
                cat_data = all_data[all_data['分类名称'] == category]
                
                if len(cat_data) > 5 and 'ln_price' in cat_data.columns and 'ln_quantity' in cat_data.columns:
                    # 转换回原始尺度用于可视化
                    prices = np.exp(cat_data['ln_price'])
                    quantities = np.exp(cat_data['ln_quantity'])
                    
                    # 散点图
                    ax.scatter(prices, quantities, alpha=0.6, s=30)
                    
                    # 拟合线（使用对数线性回归）
                    if len(cat_data) > 10:
                        z = np.polyfit(cat_data['ln_price'], cat_data['ln_quantity'], 1)
                        p = np.poly1d(z)
                        
                        # 生成拟合线数据点
                        price_range = np.linspace(prices.min(), prices.max(), 100)
                        fitted_quantities = np.exp(p(np.log(price_range)))
                        ax.plot(price_range, fitted_quantities, 'r-', alpha=0.8, linewidth=2)
                        
                        # 显示弹性系数
                        elasticity = z[0]  # 对数-对数模型的斜率即为弹性
                        ax.text(0.05, 0.95, f'弹性: {elasticity:.3f}', 
                               transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                ax.set_title(f'{category}', fontweight='bold')
                ax.set_xlabel('价格 (元/千克)')
                ax.set_ylabel('销量 (千克)')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(n_categories, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图片
            os.makedirs(self.output_paths['figures_dir'], exist_ok=True)
            fig_path = os.path.join(self.output_paths['figures_dir'], 'price_quantity_relationship.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"价格-销量关系图已保存: {fig_path}")
            return fig_path
            
        except Exception as e:
            print(f"绘制价格-销量关系图失败: {e}")
            return None
    
    def analyze_markup_sensitivity(self, train_data, test_data):
        """分析加成率对销量的敏感性（使用特征数据）"""
        try:
            # 合并数据
            all_data = pd.concat([train_data, test_data], ignore_index=True)
            
            # 使用ln_markup_ratio来分析加成率敏感性
            if 'ln_markup_ratio' not in all_data.columns or 'ln_quantity' not in all_data.columns:
                print("缺少必要的特征列，跳过加成率敏感性分析")
                return None
            
            # 计算实际加成率
            all_data['markup_ratio'] = np.exp(all_data['ln_markup_ratio'])
            all_data['quantity'] = np.exp(all_data['ln_quantity'])
            
            # 定义加成率区间
            markup_ranges = [(1.0, 1.2), (1.2, 1.4), (1.4, 1.6), (1.6, 1.8), (1.8, 2.0), (2.0, 3.0)]
            
            sensitivity_results = []
            
            for category in all_data['分类名称'].unique():
                cat_data = all_data[all_data['分类名称'] == category].copy()
                
                if len(cat_data) < 20:
                    continue
                
                # 按加成率区间分析
                for min_markup, max_markup in markup_ranges:
                    range_data = cat_data[
                        (cat_data['markup_ratio'] >= min_markup) & 
                        (cat_data['markup_ratio'] < max_markup)
                    ]
                    
                    if len(range_data) >= 5:  # 至少需5个样本
                        avg_quantity = range_data['quantity'].mean()
                        median_quantity = range_data['quantity'].median()
                        std_quantity = range_data['quantity'].std()
                        count = len(range_data)
                        avg_markup = range_data['markup_ratio'].mean()
                        
                        sensitivity_results.append({
                            'category': category,
                            'markup_range': f'{min_markup:.1f}-{max_markup:.1f}',
                            'avg_markup_ratio': avg_markup,
                            'avg_quantity': avg_quantity,
                            'median_quantity': median_quantity,
                            'std_quantity': std_quantity,
                            'sample_count': count
                        })
            
            if sensitivity_results:
                sensitivity_df = pd.DataFrame(sensitivity_results)
                
                # 保存结果
                os.makedirs(self.output_paths['results_dir'], exist_ok=True)
                sensitivity_path = os.path.join(self.output_paths['results_dir'], 'markup_sensitivity_analysis.csv')
                sensitivity_df.to_csv(sensitivity_path, index=False, encoding='utf-8')
                print(f"加成率敏感性分析已保存: {sensitivity_path}")
                
                return sensitivity_df
            else:
                print("未找到足够的数据进行加成率敏感性分析")
                return None
            
        except Exception as e:
            print(f"加成率敏感性分析失败: {e}")
            return None
    
    def generate_report(self):
        """生成建模报告"""
        print("生成关系分析图表...")
        
        # 加载数据用于关系分析
        try:
            train_data = pd.read_csv(self.data_paths['train_features'])
            test_data = pd.read_csv(self.data_paths['test_features'])
            
            # 生成价格-销量关系图
            self.plot_price_quantity_relationship(train_data, test_data)
            
            # 生成加成率敏感性分析
            self.analyze_markup_sensitivity(train_data, test_data)
            
        except Exception as e:
            print(f"关系分析生成失败: {e}")
        
        # 原有的报告生成逻辑
        report_content = [
            "# 需求建模报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 建模结果汇总",
            f"- 建模品类数: {len(self.results)}",
            ""
        ]
        
        # 各品类最佳模型
        for category, results in self.results.items():
            best_model = results.get('best_model', 'Unknown')
            if best_model != 'Unknown' and best_model in results:
                best_result = results[best_model]
                report_content.append(f"### {category}")
                report_content.append(f"- 最佳模型: {best_model}")
                report_content.append(f"- 测试R²: {best_result['test_r2']:.4f}")
                if best_result.get('price_elasticity'):
                    report_content.append(f"- 价格弹性: {best_result['price_elasticity']:.4f}")
                
                cv_result = results.get('cross_validation')
                if cv_result:
                    report_content.append(f"- 交叉验证R²: {cv_result['mean_cv_score']:.4f}")
                report_content.append("")
        
        # 添加关系分析说明
        report_content.extend([
            "## 关系分析",
            "- 价格-销量关系图: 展示各品类的需求弹性特征",
            "- 加成率敏感性分析: 分析不同加成率区间下的销量和利润表现",
            "- 图表文件保存在 outputs/figures/ 目录下",
            "- 敏感性分析数据保存在 outputs/results/ 目录下",
            ""
        ])
        
        # 保存报告
        os.makedirs(self.output_paths['reports_dir'], exist_ok=True)
        report_path = os.path.join(self.output_paths['reports_dir'], 'demand_modeling_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"建模报告已保存: {report_path}")

if __name__ == "__main__":
    modeler = DemandModeler()
    modeler.run_modeling()
