# -*- coding: utf-8 -*-
"""
精简版优化器模块
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 导入字体配置
from font_config import setup_chinese_font

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
            
            # 获取每个品类的最佳模型和弹性信息
            categories = models_df['category'].unique()
            
            for category in categories:
                category_models = models_df[models_df['category'] == category]
                
                # 选择最佳模型（is_best=True）
                best_model = category_models[category_models['is_best'] == True].iloc[0]
                
                # 如果最佳模型没有弹性值（如RandomForest），就使用LinearRegression的弹性
                elasticity = best_model['price_elasticity']
                if pd.isna(elasticity):
                    # 查找线性模型的弹性
                    linear_models = category_models[category_models['model'] == 'LinearRegression']
                    if len(linear_models) > 0 and pd.notna(linear_models.iloc[0]['price_elasticity']):
                        elasticity = linear_models.iloc[0]['price_elasticity']
                        print(f"  {category}: 使用LinearRegression的弹性值 {elasticity:.3f}")
                    else:
                        # 使用基于品类的默认弹性
                        category_elasticity_map = {
                            '花叶类': -0.8, '辣椒类': -0.6, '花菜类': -1.2, 
                            '食用菌': -0.4, '茄类': -0.9, '水生根茎类': -1.0
                        }
                        elasticity = category_elasticity_map.get(category, -0.8)
                        print(f"  {category}: 使用默认弹性值 {elasticity:.3f}")
                
                if pd.notna(best_model['test_r2']) and best_model['test_r2'] > 0:
                    self.demand_models[category] = {
                        'model_type': best_model['model'],
                        'price_elasticity': elasticity,
                        'test_r2': best_model['test_r2']
                    }
            
            print(f"已加载 {len(self.demand_models)} 个品类的需求模型")
            for category, model_info in self.demand_models.items():
                print(f"  {category}: {model_info['model_type']}, 弹性={model_info['price_elasticity']:.3f}")
                
        except Exception as e:
            print(f"加载需求模型失败: {e}")
            # 使用更差异化的默认弹性
            categories = ['花叶类', '辣椒类', '花菜类', '食用菌', '茄类', '水生根茎类']
            default_elasticities = [-0.8, -0.6, -1.2, -0.4, -0.9, -1.0]
            
            for cat, elasticity in zip(categories, default_elasticities):
                self.demand_models[cat] = {
                    'model_type': 'default',
                    'price_elasticity': elasticity,
                    'test_r2': 0.3
                }
            print("使用默认需求模型")
    
    def predict_demand(self, category, price, base_quantity=15.0):
        """改进的需求预测函数"""
        if category not in self.demand_models:
            return base_quantity, base_quantity * 0.15
        
        model_info = self.demand_models[category]
        elasticity = model_info['price_elasticity']
        
        # 使用更合理的基准价格（基于品类）
        base_price_map = {
            '花叶类': 7.5, '辣椒类': 8.0, '花菜类': 12.0, 
            '食用菌': 16.0, '茄类': 6.0, '水生根茎类': 9.0
        }
        base_price = base_price_map.get(category, 8.0)
        
        # 改进的弹性计算
        if abs(elasticity) < 0.1:  # 避免极小弹性值
            elasticity = -0.5
        
        # 价格效应计算
        price_ratio = price / base_price
        price_effect = price_ratio ** elasticity
        
        # 预测需求，确保合理范围
        predicted_demand = base_quantity * price_effect
        predicted_demand = np.clip(predicted_demand, base_quantity * 0.3, base_quantity * 3.0)
        
        # 需求不确定性
        uncertainty_factor = max(0.05, 1 - model_info['test_r2']) * 0.2
        demand_std = predicted_demand * uncertainty_factor
        
        return predicted_demand, demand_std
    
    def calculate_profit_scenarios(self, category, price, quantity, wholesale_cost, 
                                 base_quantity=15.0, n_scenarios=20, wastage_rate=0.05):
        """改进的利润场景计算"""
        # 预测需求，传入正确的基准数量
        mean_demand, std_demand = self.predict_demand(category, price, base_quantity)
        
        # 生成需求场景
        np.random.seed(42)  # 设置随机种子保证结果稳定
        demand_scenarios = np.random.normal(mean_demand, std_demand, n_scenarios)
        demand_scenarios = np.maximum(demand_scenarios, 0.1)  # 确保非负
        
        profits = []
        for demand in demand_scenarios:
            # 可销售量（减少损耗率）
            available_quantity = quantity * (1 - wastage_rate)
            actual_sales = min(demand, available_quantity)
            
            # 收入
            revenue = actual_sales * price
            
            # 成本
            cost = quantity * wholesale_cost
            
            # 增强缺货惩罚机制
            stockout = max(0, demand - actual_sales)
            # 使用配置中的权重来增强缺货惩罚
            stockout_penalty = stockout * wholesale_cost * 0.05 * self.opt_config['stockout_penalty_weight']
            
            # 净利润
            profit = revenue - cost - stockout_penalty
            profits.append(profit)
        
        return np.array(profits)
    
    def calculate_theoretical_optimal_markup(self, elasticity):
        """基于经济学理论计算最优加价率"""
        # 先处理异常弹性值
        if elasticity >= 0:  # 正弹性或零弹性不合理，使用默认负弹性
            elasticity = self.opt_config.get('demand_elasticity_prior', -1.0)
            print(f"    检测到非负弹性，使用默认弹性: {elasticity}")
        
        abs_elasticity = abs(elasticity)
        
        # 处理弹性值过小的情况
        if abs_elasticity < 0.3:  # 弹性太小，使用先验弹性
            abs_elasticity = abs(self.opt_config.get('demand_elasticity_prior', -1.0))
            print(f"    弹性值过小，使用先验弹性: {abs_elasticity}")
        
        # 计算理论最优加价率，避免数值问题
        if abs_elasticity > 1.05:  # 给出一些缓冲，避免接近零除
            # 弹性大于1时，使用经典公式
            theoretical_markup = abs_elasticity / (abs_elasticity - 1)
        else:
            # 弹性小于等于1时，使用简化公式：随弹性增大而加价减小
            # 对于弹性=0.5，给出加价约1.6-1.7左右
            # 对于弹性=1.0，给出加价约1.4-1.5左右
            theoretical_markup = 1.3 + (1.0 - abs_elasticity) * 0.4
        
        # 限制在合理范围内
        theoretical_markup = np.clip(theoretical_markup, 1.3, 2.2)
        
        return theoretical_markup
    
    def golden_section_search(self, category, wholesale_cost, base_quantity, a, b, tol=1e-3):
        """黄金分割搜索最优加价率"""
        # 黄金比例
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi
        
        # 初始点
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        
        # 计算初始函数值
        f1 = -self.evaluate_markup_profit(category, wholesale_cost, base_quantity, x1)
        f2 = -self.evaluate_markup_profit(category, wholesale_cost, base_quantity, x2)
        
        # 迭代搜索
        for _ in range(self.opt_config.get('golden_section_iterations', 15)):
            if f2 > f1:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = -self.evaluate_markup_profit(category, wholesale_cost, base_quantity, x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - resphi * (b - a)
                f2 = -self.evaluate_markup_profit(category, wholesale_cost, base_quantity, x2)
            
            # 收敛判断
            if abs(b - a) < tol:
                break
        
        return (a + b) / 2
    
    def evaluate_markup_profit(self, category, wholesale_cost, base_quantity, markup_ratio):
        """评估给定加价率下的利润"""
        price = wholesale_cost * markup_ratio
        
        # 预测需求
        predicted_demand, _ = self.predict_demand(category, price, base_quantity)
        
        # 检查最低销量约束
        min_sales = base_quantity * self.opt_config.get('min_service_sales_ratio', 0.7)
        if predicted_demand < min_sales:
            return -float('inf')  # 惩罚不满足最低销量约束的方案
        
        # 计算安全库存（基于服务水平和需求不确定性）
        service_level = self.opt_config['service_level']
        # 简化的安全系数映射
        if service_level >= 0.95:
            safety_factor = 1.65
        elif service_level >= 0.90:
            safety_factor = 1.28
        elif service_level >= 0.80:
            safety_factor = 0.84
        else:
            safety_factor = 0.50
        
        # 需求不确定性系数（简化处理）
        uncertainty_cv = 0.15  # 15%的变异系数
        test_quantity = predicted_demand * (1 + safety_factor * uncertainty_cv)
        
        # 计算期望利润
        profit_scenarios = self.calculate_profit_scenarios(
            category, price, test_quantity, wholesale_cost, base_quantity, 
            n_scenarios=20
        )
        
        expected_profit = np.mean(profit_scenarios)
        profit_std = np.std(profit_scenarios)
        
        # 引入风险调整（简化的风险厌恶）
        risk_penalty = 0.1 * profit_std  # 对高风险方案的小幅惩罚
        risk_adjusted_profit = expected_profit - risk_penalty
        
        return risk_adjusted_profit
    
    def optimize_category_heuristic(self, category, wholesale_cost, base_quantity):
        """改进的启发式优化 - 使用闭式解锚点和黄金分割搜索"""
        if category not in self.demand_models:
            optimal_price = wholesale_cost * 1.7  # 使用区间中点作为默认
            optimal_quantity = base_quantity * 1.1
        else:
            model_info = self.demand_models[category]
            elasticity = model_info['price_elasticity']
            
            # 计算理论最优加价率作为锚点
            theoretical_markup = self.calculate_theoretical_optimal_markup(elasticity)
            print(f"{category} 弹性={elasticity:.3f}, 理论最优加价率={theoretical_markup:.3f}")
            
            # 设定搜索区间
            min_markup = self.opt_config['min_markup_ratio']
            max_markup = self.opt_config['max_markup_ratio']
            
            # 如果理论最优在区间内，以它为中心缩小搜索区间
            if min_markup <= theoretical_markup <= max_markup:
                # 在理论最优附近搜索
                search_range = (max_markup - min_markup) * 0.3
                search_min = max(min_markup, theoretical_markup - search_range)
                search_max = min(max_markup, theoretical_markup + search_range)
            else:
                # 理论最优在区间外，搜索整个区间
                search_min = min_markup
                search_max = max_markup
            
            print(f"  搜索区间: [{search_min:.3f}, {search_max:.3f}]")
            
            # 使用黄金分割搜索找最优加价率
            optimal_markup = self.golden_section_search(
                category, wholesale_cost, base_quantity, search_min, search_max
            )
            
            print(f"  最终选择加价率: {optimal_markup:.3f}")
            
            optimal_price = wholesale_cost * optimal_markup
            
            # 计算最优数量
            predicted_demand, _ = self.predict_demand(category, optimal_price, base_quantity)
            service_level = self.opt_config['service_level']
            
            # 安全库存计算
            if service_level >= 0.95:
                safety_factor = 1.65
            elif service_level >= 0.90:
                safety_factor = 1.28
            elif service_level >= 0.80:
                safety_factor = 0.84
            else:
                safety_factor = 0.50
            
            uncertainty_cv = 0.15
            optimal_quantity = predicted_demand * (1 + safety_factor * uncertainty_cv)
        
        # 确保最小值
        optimal_quantity = max(optimal_quantity, base_quantity * 0.7)
        
        return optimal_price, optimal_quantity
    
    def generate_wholesale_forecasts(self):
        """生成简化的批发价预测"""
        forecasts = {}
        horizon = self.opt_config['optimization_horizon']
        
        # 各品类基准批发价（更合理的价格）
        base_wholesale_prices = {
            '花叶类': 5.5, '辣椒类': 6.5, '花菜类': 9.0, 
            '食用菌': 13.0, '茄类': 4.8, '水生根茎类': 7.5
        }
        
        for category, base_price in base_wholesale_prices.items():
            daily_forecasts = []
            current_price = base_price
            
            for day in range(horizon):
                # 减少随机波动，使用更稳定的预测
                price_change = np.random.normal(0, 0.01)  # 1%的日波动
                current_price *= (1 + price_change)
                current_price = max(current_price, base_price * 0.9)  # 最低不低于基价的90%
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
                
                # 每日基准量，减少随机波动
                daily_base_qty = base_quantity * (1 + np.random.normal(0, 0.05))
                daily_base_qty = max(daily_base_qty, base_quantity * 0.8)
                
                # 启发式优化
                optimal_price, optimal_quantity = self.optimize_category_heuristic(
                    category, wholesale_cost, daily_base_qty
                )
                
                # 计算预期利润
                profit_scenarios = self.calculate_profit_scenarios(
                    category, optimal_price, optimal_quantity, wholesale_cost,
                    daily_base_qty, n_scenarios=self.opt_config['monte_carlo_samples']
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
    
    def plot_price_quantity_profit_analysis(self):
        """绘制价格-销量-利润扫描图"""
        try:
            # 设置中文字体
            setup_chinese_font()
            
            categories = list(self.demand_models.keys())
            if not categories:
                print("没有可用的需求模型")
                return None
            
            # 创建子图
            n_categories = len(categories)
            cols = 2
            rows = (n_categories + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
            fig.suptitle('各品类价格-销量-利润关系分析', fontsize=16, fontweight='bold')
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            # 各品类基础数据
            base_quantities = {
                '花叶类': 25.0, '辣椒类': 15.0, '花菜类': 20.0,
                '食用菌': 12.0, '茄类': 18.0, '水生根茎类': 10.0
            }
            
            base_wholesale_prices = {
                '花叶类': 5.5, '辣椒类': 6.5, '花菜类': 9.0, 
                '食用菌': 13.0, '茄类': 4.8, '水生根茎类': 7.5
            }
            
            for i, category in enumerate(categories):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                base_quantity = base_quantities.get(category, 15.0)
                wholesale_cost = base_wholesale_prices.get(category, 8.0)
                
                # 生成价格区间（聚焦在1.6-1.8加成率）
                markup_ratios = np.linspace(1.6, 1.8, 20)
                prices = wholesale_cost * markup_ratios
                
                quantities = []
                profits = []
                
                for price in prices:
                    # 预测需求
                    predicted_demand, _ = self.predict_demand(category, price, base_quantity)
                    quantities.append(predicted_demand)
                    
                    # 计算利润（简化）
                    quantity_to_stock = predicted_demand * 1.1  # 安全库存
                    profit = (price - wholesale_cost) * predicted_demand - \
                             max(0, quantity_to_stock - predicted_demand) * wholesale_cost * 0.1
                    profits.append(profit)
                
                # 双 y 轴图
                ax2 = ax.twinx()
                
                # 绘制价格-销量关系
                line1 = ax.plot(prices, quantities, 'b-', linewidth=2, label='预测需求')
                ax.set_xlabel('价格 (元/千克)')
                ax.set_ylabel('需求量 (千克)', color='b')
                ax.tick_params(axis='y', labelcolor='b')
                
                # 绘制价格-利润关系
                line2 = ax2.plot(prices, profits, 'r-', linewidth=2, label='预期利润')
                ax2.set_ylabel('预期利润 (元)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                
                # 找到最优利润点
                max_profit_idx = np.argmax(profits)
                optimal_price = prices[max_profit_idx]
                optimal_quantity = quantities[max_profit_idx]
                optimal_profit = profits[max_profit_idx]
                
                # 标记最优点
                ax.scatter([optimal_price], [optimal_quantity], color='green', s=100, zorder=5)
                ax2.scatter([optimal_price], [optimal_profit], color='green', s=100, zorder=5)
                
                # 添加最优点标注
                ax.annotate(f'最优点\n价格: {optimal_price:.2f}\n需求: {optimal_quantity:.1f}', 
                           xy=(optimal_price, optimal_quantity), xytext=(10, 10),
                           textcoords='offset points', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                ax.set_title(f'{category}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 显示弹性系数
                elasticity = self.demand_models[category]['price_elasticity']
                ax.text(0.02, 0.98, f'价格弹性: {elasticity:.3f}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # 隐藏多余的子图
            for i in range(n_categories, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图片
            os.makedirs(self.output_paths['figures_dir'], exist_ok=True)
            fig_path = os.path.join(self.output_paths['figures_dir'], 'price_quantity_profit_analysis.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"价格-销量-利润分析图已保存: {fig_path}")
            return fig_path
            
        except Exception as e:
            print(f"绘制价格-销量-利润分析图失败: {e}")
            return None
    
    def generate_optimization_report(self, results):
        """生成优化报告"""
        # 生成价格-销量-利润分析图
        print("生成价格-销量-利润关系分析...")
        self.plot_price_quantity_profit_analysis()
        
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
        report_content.append("")
        report_content.append("## 关系分析")
        report_content.append("- 价格-销量-利润关系图: 展示成本加成定价对销售总量和利润的影响")
        report_content.append("- 图表中绿色点标示理论上的最优定价点")
        report_content.append("- 蓝线显示价格对需求的影响，红线显示价格对利润的影响")
        report_content.append("- 分析结果保存在 outputs/figures/ 目录下")
        
        # 保存报告
        os.makedirs(self.output_paths['reports_dir'], exist_ok=True)
        report_path = os.path.join(self.output_paths['reports_dir'], 'optimization_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"优化报告已保存: {report_path}")

if __name__ == "__main__":
    optimizer = VegetableOptimizer()
    optimizer.run_optimization()
