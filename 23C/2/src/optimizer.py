# -*- coding: utf-8 -*-
"""
精简版优化器模块
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VegetableOptimizer:
    """精简版蔬菜定价与补货优化器"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化优化器"""
        self.config = self.load_config(config_path)
        self.data_paths = self.config['data_paths']
        self.output_paths = self.config['output_paths']
        self.opt_config = self.config['optimization']
        self.demand_models = {}
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_demand_models(self):
        """加载需求模型结果"""
        print("加载需求模型结果...")
        
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            models_df = pd.read_csv(results_path)
            
            # 选择最佳模型（is_best=True）
            best_models = models_df[models_df['is_best'] == True]
            
            for _, row in best_models.iterrows():
                if pd.notna(row['test_r2']) and row['test_r2'] > 0:
                    self.demand_models[row['category']] = {
                        'model_type': row['model'],
                        'price_elasticity': row['price_elasticity'] if pd.notna(row['price_elasticity']) else -0.5,
                        'test_r2': row['test_r2']
                    }
            
            print(f"已加载 {len(self.demand_models)} 个品类的需求模型")
            for category, model_info in self.demand_models.items():
                print(f"  {category}: {model_info['model_type']}, 弹性={model_info['price_elasticity']:.3f}")
                
        except Exception as e:
            print(f"加载需求模型失败: {e}")
            # 使用默认弹性
            categories = ['花叶类', '辣椒类', '花菜类', '食用菌', '茄类', '水生根茎类']
            default_elasticities = [-0.5, -0.6, -0.4, -0.3, -0.7, -0.8]
            
            for cat, elasticity in zip(categories, default_elasticities):
                self.demand_models[cat] = {
                    'model_type': 'default',
                    'price_elasticity': elasticity,
                    'test_r2': 0.3
                }
            print("使用默认需求模型")
    
    def predict_demand(self, category, price, base_quantity=15.0):
        """简化的需求预测"""
        if category not in self.demand_models:
            return base_quantity, base_quantity * 0.2
        
        model_info = self.demand_models[category]
        elasticity = model_info['price_elasticity']
        
        # 简化的需求预测（基于价格弹性）
        base_price = 8.0  # 假设基准价格
        price_effect = (price / base_price) ** elasticity
        predicted_demand = base_quantity * price_effect
        
        # 确保预测量为正
        predicted_demand = max(predicted_demand, 1.0)
        
        # 需求标准差（基于R²估算不确定性）
        uncertainty_factor = max(0.1, 1 - model_info['test_r2'])
        demand_std = predicted_demand * uncertainty_factor * 0.3
        
        return predicted_demand, demand_std
    
    def calculate_profit_scenarios(self, category, price, quantity, wholesale_cost, 
                                 n_scenarios=20, wastage_rate=0.1):
        """计算利润场景"""
        # 预测需求
        mean_demand, std_demand = self.predict_demand(category, price)
        
        # 生成需求场景
        demand_scenarios = np.random.normal(mean_demand, std_demand, n_scenarios)
        demand_scenarios = np.maximum(demand_scenarios, 0.5)  # 确保非负
        
        profits = []
        for demand in demand_scenarios:
            # 可销售量（考虑损耗）
            available_quantity = quantity * (1 - wastage_rate)
            actual_sales = min(demand, available_quantity)
            
            # 收入
            revenue = actual_sales * price
            
            # 成本
            cost = quantity * wholesale_cost
            
            # 缺货惩罚（简化）
            stockout = max(0, demand - actual_sales)
            stockout_penalty = stockout * price * 0.1  # 简化的缺货成本
            
            # 净利润
            profit = revenue - cost - stockout_penalty
            profits.append(profit)
        
        return np.array(profits)
    
    def optimize_category_heuristic(self, category, wholesale_cost, base_quantity):
        """启发式优化单个品类"""
        if category not in self.demand_models:
            # 默认策略
            optimal_price = wholesale_cost * 1.3
            optimal_quantity = base_quantity * 1.1
        else:
            model_info = self.demand_models[category]
            elasticity = model_info['price_elasticity']
            
            # 基于弹性的启发式定价
            if abs(elasticity) > 0.8:  # 高弹性：薄利多销
                markup = np.random.uniform(1.15, 1.25)
            elif abs(elasticity) > 0.4:  # 中等弹性
                markup = np.random.uniform(1.25, 1.4)
            else:  # 低弹性：可以高加成
                markup = np.random.uniform(1.4, 1.6)
            
            # 限制在配置范围内
            markup = np.clip(markup, self.opt_config['min_markup_ratio'], 
                           self.opt_config['max_markup_ratio'])
            
            optimal_price = wholesale_cost * markup
            
            # 基于需求预测的补货量
            predicted_demand, _ = self.predict_demand(category, optimal_price, base_quantity)
            safety_factor = 1.1 + np.random.normal(0, 0.05)  # 安全系数
            optimal_quantity = predicted_demand * safety_factor
        
        # 限制最小值
        optimal_quantity = max(optimal_quantity, base_quantity * 0.5)
        
        return optimal_price, optimal_quantity
    
    def generate_wholesale_forecasts(self):
        """生成简化的批发价预测"""
        forecasts = {}
        horizon = self.opt_config['optimization_horizon']
        
        # 各品类基准批发价
        base_wholesale_prices = {
            '花叶类': 5.0, '辣椒类': 6.0, '花菜类': 8.0, 
            '食用菌': 12.0, '茄类': 4.5, '水生根茎类': 7.0
        }
        
        for category, base_price in base_wholesale_prices.items():
            daily_forecasts = []
            current_price = base_price
            
            for day in range(horizon):
                # 添加随机波动
                price_change = np.random.normal(0, 0.03)  # 3%的日波动
                current_price *= (1 + price_change)
                current_price = max(current_price, base_price * 0.7)  # 最低不低于基价的70%
                daily_forecasts.append(current_price)
            
            forecasts[category] = daily_forecasts
        
        return forecasts
    
    def run_optimization(self):
        """运行优化流程"""
        print("=== 开始优化算法 ===")
        
        # 加载需求模型
        self.load_demand_models()
        
        # 生成批发价预测
        wholesale_forecasts = self.generate_wholesale_forecasts()
        
        # 优化结果
        optimization_results = {}
        horizon = self.opt_config['optimization_horizon']
        
        # 各品类基准销量
        base_quantities = {
            '花叶类': 25.0, '辣椒类': 15.0, '花菜类': 20.0,
            '食用菌': 12.0, '茄类': 18.0, '水生根茎类': 10.0
        }
        
        print("执行启发式优化...")
        
        for category in self.demand_models.keys():
            daily_results = []
            
            wholesale_costs = wholesale_forecasts.get(category, [8.0] * horizon)
            base_quantity = base_quantities.get(category, 15.0)
            
            for day in range(horizon):
                date = f"2023-07-{day+1:02d}"
                wholesale_cost = wholesale_costs[day]
                
                # 每日基准量加随机波动
                daily_base_qty = base_quantity * (1 + np.random.normal(0, 0.1))
                daily_base_qty = max(daily_base_qty, base_quantity * 0.7)
                
                # 启发式优化
                optimal_price, optimal_quantity = self.optimize_category_heuristic(
                    category, wholesale_cost, daily_base_qty
                )
                
                # 计算预期利润
                profit_scenarios = self.calculate_profit_scenarios(
                    category, optimal_price, optimal_quantity, wholesale_cost,
                    n_scenarios=self.opt_config['monte_carlo_samples']
                )
                expected_profit = np.mean(profit_scenarios)
                profit_std = np.std(profit_scenarios)
                
                daily_result = {
                    'date': date,
                    'optimal_price': round(optimal_price, 2),
                    'optimal_quantity': round(optimal_quantity, 1),
                    'wholesale_cost': round(wholesale_cost, 2),
                    'expected_profit': round(expected_profit, 2),
                    'profit_std': round(profit_std, 2),
                    'markup_ratio': round(optimal_price / wholesale_cost, 3),
                    'service_level': self.opt_config['service_level']
                }
                daily_results.append(daily_result)
            
            optimization_results[category] = daily_results
        
        # 保存结果
        self.save_optimization_results(optimization_results)
        self.generate_optimization_report(optimization_results)
        
        print("=== 优化算法完成 ===\n")
        return True
    
    def save_optimization_results(self, results):
        """保存优化结果"""
        print("保存优化结果...")
        
        # 详细结果
        all_results = []
        for category, daily_results in results.items():
            for daily_result in daily_results:
                row = {
                    'category': category,
                    'date': daily_result['date'],
                    'optimal_price': daily_result['optimal_price'],
                    'optimal_quantity': daily_result['optimal_quantity'],
                    'wholesale_cost': daily_result['wholesale_cost'],
                    'expected_profit': daily_result['expected_profit'],
                    'profit_std': daily_result['profit_std'],
                    'markup_ratio': daily_result['markup_ratio'],
                    'service_level': daily_result['service_level']
                }
                all_results.append(row)
        
        # 确保输出目录存在
        os.makedirs(self.output_paths['results_dir'], exist_ok=True)
        
        # 保存详细结果
        results_df = pd.DataFrame(all_results)
        detailed_path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        
        # 生成周策略汇总
        weekly_strategy = []
        for category, daily_results in results.items():
            total_quantity = sum(r['optimal_quantity'] for r in daily_results)
            total_profit = sum(r['expected_profit'] for r in daily_results)
            avg_price = np.mean([r['optimal_price'] for r in daily_results])
            avg_markup = np.mean([r['markup_ratio'] for r in daily_results])
            
            weekly_strategy.append({
                'category': category,
                'avg_weekly_price': round(avg_price, 2),
                'total_weekly_quantity': round(total_quantity, 1),
                'total_expected_profit': round(total_profit, 2),
                'avg_markup_ratio': round(avg_markup, 3),
                'price_range_min': min(r['optimal_price'] for r in daily_results),
                'price_range_max': max(r['optimal_price'] for r in daily_results)
            })
        
        weekly_df = pd.DataFrame(weekly_strategy)
        weekly_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
        weekly_df.to_csv(weekly_path, index=False, encoding='utf-8')
        
        print(f"优化结果已保存:")
        print(f"  - 详细结果: {detailed_path}")
        print(f"  - 周策略: {weekly_path}")
    
    def generate_optimization_report(self, results):
        """生成优化报告"""
        report_content = [
            "# 优化策略报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 优化配置",
            f"- 优化时间窗口: {self.opt_config['optimization_horizon']} 天",
            f"- 服务水平: {self.opt_config['service_level']:.0%}",
            f"- 最小加成率: {self.opt_config['min_markup_ratio']:.0%}",
            f"- 最大加成率: {self.opt_config['max_markup_ratio']:.0%}",
            "",
            "## 优化结果汇总"
        ]
        
        total_profit = 0
        for category, daily_results in results.items():
            category_profit = sum(r['expected_profit'] for r in daily_results)
            total_profit += category_profit
            
            avg_price = np.mean([r['optimal_price'] for r in daily_results])
            avg_markup = np.mean([r['markup_ratio'] for r in daily_results])
            
            report_content.append(f"### {category}")
            report_content.append(f"- 周期望利润: {category_profit:.2f}元")
            report_content.append(f"- 平均定价: {avg_price:.2f}元/千克")
            report_content.append(f"- 平均加成率: {avg_markup:.1%}")
            report_content.append("")
        
        report_content.append(f"**总期望利润: {total_profit:.2f}元**")
        report_content.append("")
        report_content.append("## 方法说明")
        report_content.append("- 使用启发式算法基于需求弹性进行定价")
        report_content.append("- 考虑库存风险和缺货成本")
        report_content.append("- 基于蒙特卡洛模拟评估利润期望")
        
        # 保存报告
        os.makedirs(self.output_paths['reports_dir'], exist_ok=True)
        report_path = os.path.join(self.output_paths['reports_dir'], 'optimization_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"优化报告已保存: {report_path}")

if __name__ == "__main__":
    optimizer = VegetableOptimizer()
    optimizer.run_optimization()
