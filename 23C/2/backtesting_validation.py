"""
回测与验证模块
实现滚动起点时间序列交叉验证，监控模型稳定性和漂移
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BacktestingValidator:
    """回测验证器"""
    
    def __init__(self, config=None):
        self.config = config or {
            'min_train_days': 30,
            'test_days': 7,
            'step_days': 7,
            'max_splits': 8,
            'stability_threshold': 0.1,
            'drift_threshold': 0.15
        }
        
        self.backtest_results = {}
        self.stability_metrics = {}
        self.drift_analysis = {}
        
    def load_data(self):
        """加载训练和测试数据"""
        try:
            print("加载回测数据...")
            self.train_features = pd.read_csv('train_features.csv')
            self.test_features = pd.read_csv('test_features.csv')
            
            # 确保日期列为datetime类型
            self.train_features['销售日期'] = pd.to_datetime(self.train_features['销售日期'])
            self.test_features['销售日期'] = pd.to_datetime(self.test_features['销售日期'])
            
            # 合并数据用于时间序列分割
            self.all_data = pd.concat([self.train_features, self.test_features], ignore_index=True)
            self.all_data = self.all_data.sort_values(['分类名称', '销售日期']).reset_index(drop=True)
            
            print(f"数据加载完成: {len(self.all_data)} 条记录")
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def create_time_splits(self, category_data):
        """创建时间序列交叉验证分割"""
        dates = sorted(category_data['销售日期'].unique())
        splits = []
        
        min_train_days = self.config['min_train_days']
        test_days = self.config['test_days']
        step_days = self.config['step_days']
        max_splits = self.config['max_splits']
        
        for i in range(max_splits):
            # 计算训练和测试期间
            test_start_idx = min_train_days + i * step_days
            test_end_idx = test_start_idx + test_days
            
            if test_end_idx >= len(dates):
                break
                
            train_end_date = dates[test_start_idx - 1]
            test_start_date = dates[test_start_idx]
            test_end_date = dates[min(test_end_idx - 1, len(dates) - 1)]
            
            # 创建训练和测试集
            train_mask = category_data['销售日期'] <= train_end_date
            test_mask = (category_data['销售日期'] >= test_start_date) & \
                       (category_data['销售日期'] <= test_end_date)
            
            if train_mask.sum() >= min_train_days and test_mask.sum() >= 1:
                splits.append({
                    'split_id': i,
                    'train_end_date': train_end_date,
                    'test_start_date': test_start_date,
                    'test_end_date': test_end_date,
                    'train_indices': category_data[train_mask].index.tolist(),
                    'test_indices': category_data[test_mask].index.tolist()
                })
        
        return splits
    
    def evaluate_model_performance(self, y_true, y_pred, split_info):
        """评估模型性能"""
        metrics = {
            'split_id': split_info['split_id'],
            'train_end_date': split_info['train_end_date'],
            'test_start_date': split_info['test_start_date'],
            'test_end_date': split_info['test_end_date'],
            'n_train': len(split_info['train_indices']),
            'n_test': len(split_info['test_indices']),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        return metrics
    
    def run_category_backtest(self, category):
        """对单个品类进行回测"""
        print(f"回测品类: {category}")
        
        # 获取品类数据
        category_data = self.all_data[self.all_data['分类名称'] == category].copy()
        
        if len(category_data) < self.config['min_train_days'] + self.config['test_days']:
            print(f"品类 {category} 数据不足，跳过")
            return None
        
        # 创建时间分割
        splits = self.create_time_splits(category_data)
        
        if len(splits) < 2:
            print(f"品类 {category} 无法创建足够的时间分割，跳过")
            return None
        
        # 准备特征和目标变量
        feature_cols = [col for col in category_data.columns 
                       if col not in ['销售日期', '分类名称', '单品编码', '单品名称', 'ln_quantity']]
        
        X = category_data[feature_cols]
        y = np.exp(category_data['ln_quantity'])  # 转换回原始销量
        
        # 对每个分割进行建模和评估
        split_results = []
        models_performance = {'RandomForest': [], 'GradientBoosting': [], 'Huber': []}
        
        for split in splits:
            train_idx = split['train_indices']
            test_idx = split['test_indices']
            
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            
            # 处理缺失值
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())
            
            # 测试多个模型
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'Huber': HuberRegressor(epsilon=1.35, max_iter=100)
            }
            
            split_result = {'split_info': split}
            
            for model_name, model in models.items():
                try:
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 评估
                    metrics = self.evaluate_model_performance(y_test, y_pred, split)
                    metrics['model'] = model_name
                    
                    models_performance[model_name].append(metrics)
                    split_result[model_name] = metrics
                    
                except Exception as e:
                    print(f"模型 {model_name} 在分割 {split['split_id']} 失败: {e}")
                    continue
            
            split_results.append(split_result)
        
        # 计算稳定性指标
        stability_results = self.calculate_stability_metrics(models_performance)
        
        return {
            'category': category,
            'split_results': split_results,
            'models_performance': models_performance,
            'stability_metrics': stability_results
        }
    
    def calculate_stability_metrics(self, models_performance):
        """计算模型稳定性指标"""
        stability_results = {}
        
        for model_name, results in models_performance.items():
            if len(results) < 2:
                continue
                
            # 提取性能指标
            rmse_values = [r['rmse'] for r in results]
            r2_values = [r['r2'] for r in results]
            mape_values = [r['mape'] for r in results]
            
            # 计算稳定性
            stability_results[model_name] = {
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'rmse_cv': np.std(rmse_values) / (np.mean(rmse_values) + 1e-8),
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values),
                'r2_cv': np.std(r2_values) / (np.abs(np.mean(r2_values)) + 1e-8),
                'mape_mean': np.mean(mape_values),
                'mape_std': np.std(mape_values),
                'n_splits': len(results),
                'is_stable': np.std(rmse_values) / (np.mean(rmse_values) + 1e-8) < self.config['stability_threshold']
            }
        
        return stability_results
    
    def detect_model_drift(self, category_results):
        """检测模型漂移"""
        drift_results = {}
        
        for model_name, results in category_results['models_performance'].items():
            if len(results) < 3:
                continue
                
            # 计算性能趋势
            rmse_values = [r['rmse'] for r in results]
            r2_values = [r['r2'] for r in results]
            
            # 线性趋势检验
            x = np.arange(len(rmse_values))
            
            # RMSE趋势
            rmse_trend = np.polyfit(x, rmse_values, 1)[0]
            
            # R2趋势
            r2_trend = np.polyfit(x, r2_values, 1)[0]
            
            # 检测显著漂移
            rmse_drift = abs(rmse_trend) > self.config['drift_threshold'] * np.mean(rmse_values)
            r2_drift = abs(r2_trend) > self.config['drift_threshold'] * abs(np.mean(r2_values))
            
            drift_results[model_name] = {
                'rmse_trend': rmse_trend,
                'r2_trend': r2_trend,
                'has_drift': rmse_drift or r2_drift,
                'drift_severity': max(abs(rmse_trend) / (np.mean(rmse_values) + 1e-8),
                                    abs(r2_trend) / (abs(np.mean(r2_values)) + 1e-8))
            }
        
        return drift_results
    
    def run_full_backtest(self):
        """运行完整回测"""
        if not self.load_data():
            return False
        
        print("开始回测验证...")
        
        # 获取所有品类
        categories = self.all_data['分类名称'].unique()
        print(f"待回测品类: {list(categories)}")
        
        all_results = {}
        
        for category in categories:
            try:
                result = self.run_category_backtest(category)
                if result:
                    # 检测漂移
                    drift_result = self.detect_model_drift(result)
                    result['drift_analysis'] = drift_result
                    
                    all_results[category] = result
                    
            except Exception as e:
                print(f"品类 {category} 回测失败: {e}")
                continue
        
        self.backtest_results = all_results
        
        # 保存结果
        self.save_backtest_results()
        self.generate_backtest_report()
        
        print("回测验证完成！")
        return True
    
    def save_backtest_results(self):
        """保存回测结果"""
        # 汇总所有分割结果
        all_splits = []
        
        for category, results in self.backtest_results.items():
            for split_result in results['split_results']:
                for model_name in ['RandomForest', 'GradientBoosting', 'Huber']:
                    if model_name in split_result:
                        metrics = split_result[model_name].copy()
                        metrics['category'] = category
                        all_splits.append(metrics)
        
        splits_df = pd.DataFrame(all_splits)
        splits_df.to_csv('backtest_splits_results.csv', index=False, encoding='utf-8')
        
        # 汇总稳定性结果
        stability_summary = []
        for category, results in self.backtest_results.items():
            for model_name, stability in results['stability_metrics'].items():
                row = stability.copy()
                row['category'] = category
                row['model'] = model_name
                stability_summary.append(row)
        
        stability_df = pd.DataFrame(stability_summary)
        stability_df.to_csv('backtest_stability_results.csv', index=False, encoding='utf-8')
        
        print("回测结果已保存:")
        print("  - backtest_splits_results.csv")
        print("  - backtest_stability_results.csv")
    
    def generate_backtest_report(self):
        """生成回测报告"""
        report = []
        report.append("# 回测与验证报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## 回测配置")
        report.append(f"- 最小训练天数: {self.config['min_train_days']}")
        report.append(f"- 测试天数: {self.config['test_days']}")
        report.append(f"- 步长天数: {self.config['step_days']}")
        report.append(f"- 最大分割数: {self.config['max_splits']}")
        report.append(f"- 稳定性阈值: {self.config['stability_threshold']}")
        report.append(f"- 漂移阈值: {self.config['drift_threshold']}\n")
        
        report.append("## 整体回测结果")
        report.append(f"- 回测品类数: {len(self.backtest_results)}")
        
        # 计算整体统计
        all_rmse = []
        all_r2 = []
        stable_models = 0
        total_models = 0
        drift_models = 0
        
        for category, results in self.backtest_results.items():
            for model_name, stability in results['stability_metrics'].items():
                all_rmse.append(stability['rmse_mean'])
                all_r2.append(stability['r2_mean'])
                total_models += 1
                if stability['is_stable']:
                    stable_models += 1
                    
            for model_name, drift in results['drift_analysis'].items():
                if drift['has_drift']:
                    drift_models += 1
        
        if all_rmse:
            report.append(f"- 平均RMSE: {np.mean(all_rmse):.4f}")
            report.append(f"- 平均R²: {np.mean(all_r2):.4f}")
            report.append(f"- 稳定模型比例: {stable_models/total_models:.2%}")
            report.append(f"- 存在漂移模型比例: {drift_models/total_models:.2%}\n")
        
        report.append("## 各品类详细结果\n")
        
        for category, results in self.backtest_results.items():
            report.append(f"### {category}")
            report.append(f"- 时间分割数: {len(results['split_results'])}")
            
            # 模型性能对比
            report.append("\n**模型性能对比:**")
            for model_name, stability in results['stability_metrics'].items():
                status = "✓ 稳定" if stability['is_stable'] else "⚠ 不稳定"
                report.append(f"- {model_name}: RMSE={stability['rmse_mean']:.4f}±{stability['rmse_std']:.4f}, "
                            f"R²={stability['r2_mean']:.4f}±{stability['r2_std']:.4f} {status}")
            
            # 漂移分析
            report.append("\n**漂移分析:**")
            for model_name, drift in results['drift_analysis'].items():
                drift_status = "⚠ 存在漂移" if drift['has_drift'] else "✓ 无漂移"
                report.append(f"- {model_name}: {drift_status} (严重程度: {drift['drift_severity']:.4f})")
            
            report.append("")
        
        # 保存报告
        with open('backtest_validation_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("回测报告已保存: backtest_validation_report.md")

def main():
    """主函数"""
    validator = BacktestingValidator()
    validator.run_full_backtest()

if __name__ == "__main__":
    main()
