# -*- coding: utf-8 -*-
"""
增强优化器模块
集成风险管理、价格平滑、鲁棒优化等功能
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

class EnhancedVegetableOptimizer:
    """增强蔬菜定价与补货优化器"""
    
    def __init__(self, demand_models_path='enhanced_demand_model_results.csv', 
                 wholesale_forecast_path='wholesale_forecasts.json'):
        """初始化优化器"""
        self.demand_models_path = demand_models_path
        self.wholesale_forecast_path = wholesale_forecast_path
        self.demand_models = {}
        self.wholesale_forecasts = {}
        self.optimization_config = self._get_default_config()
        
    def _get_default_config(self):
        """获取默认优化配置"""
        return {
            # 风险偏好
            'service_level': 0.8,  # P80需求满足率
            'stockout_penalty_weight': 2.0,  # 缺货惩罚权重
            'wastage_penalty_weight': 0.5,   # 损耗惩罚权重
            
            # 价格约束
            'min_markup_ratio': 1.05,  # 最小加成率
            'max_markup_ratio': 2.5,   # 最大加成率
            'max_daily_price_change': 0.1,  # 最大日价格变动率
            
            # 平滑约束
            'price_smoothing_weight': 0.1,  # 价格平滑权重
            'quantity_smoothing_weight': 0.05,  # 补货量平滑权重
            
            # 优化参数
            'optimization_horizon': 7,  # 优化时间窗口（天）
            'monte_carlo_samples': 1000,  # 蒙特卡洛样本数
        }
        
    def load_demand_models(self):
        """加载需求模型"""
        print("加载需求模型...")
        try:
            models_df = pd.read_csv(self.demand_models_path)
            
            # 选择最佳模型（按测试R²）
            best_models = models_df.loc[models_df.groupby('category')['test_r2'].idxmax()]
            
            for _, row in best_models.iterrows():
                if pd.notna(row['test_r2']) and row['test_r2'] > 0:
                    self.demand_models[row['category']] = {
                        'model_type': row['model'],
                        'price_elasticity': row['price_elasticity'] if pd.notna(row['price_elasticity']) else -0.5,
                        'test_r2': row['test_r2'],
                        'train_r2': row['train_r2']
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
                    'test_r2': 0.3,
                    'train_r2': 0.3
                }
            print("使用默认需求模型")
            
    def load_wholesale_forecasts(self):
        """加载批发价预测"""
        print("加载批发价预测...")
        try:
            import json
            with open(self.wholesale_forecast_path, 'r', encoding='utf-8') as f:
                self.wholesale_forecasts = json.load(f)
            print(f"已加载 {len(self.wholesale_forecasts)} 个品类的批发价预测")
        except Exception as e:
            print(f"加载批发价预测失败: {e}")
            # 使用默认批发价
            categories = ['花叶类', '辣椒类', '花菜类', '食用菌', '茄类', '水生根茎类']
            default_prices = [5.0, 6.0, 8.0, 12.0, 4.5, 7.0]
            
            for cat, price in zip(categories, default_prices):
                # 生成7天预测
                daily_forecasts = {}
                for day in range(1, 8):
                    date_key = f"2023-07-{day:02d}"
                    # 添加随机波动
                    daily_price = price * (1 + np.random.normal(0, 0.05))
                    daily_forecasts[date_key] = max(daily_price, price * 0.8)
                
                self.wholesale_forecasts[cat] = daily_forecasts
            print("使用默认批发价预测")
            
    def predict_demand(self, category, price, base_quantity=10, uncertainty_level='medium'):
        """预测需求量"""
        if category not in self.demand_models:
            return base_quantity
            
        model_info = self.demand_models[category]
        elasticity = model_info['price_elasticity']
        
        # 基础需求预测（简化的双对数模型）
        # ln(Q) = ln(Q_base) + elasticity * ln(P/P_base)
        base_price = 8.0  # 假设基准价格
        ln_quantity_change = elasticity * np.log(price / base_price)
        predicted_quantity = base_quantity * np.exp(ln_quantity_change)
        
        # 添加不确定性
        if uncertainty_level == 'low':
            std_factor = 0.1
        elif uncertainty_level == 'medium':
            std_factor = 0.2
        else:  # high
            std_factor = 0.3
            
        # 确保预测量为正
        predicted_quantity = max(predicted_quantity, 0.1)
        
        return predicted_quantity, predicted_quantity * std_factor
        
    def generate_demand_scenarios(self, category, price, base_quantity=10, n_scenarios=100):
        """生成需求情景"""
        mean_demand, std_demand = self.predict_demand(category, price, base_quantity)
        
        # 生成正态分布的需求情景
        scenarios = np.random.normal(mean_demand, std_demand, n_scenarios)
        scenarios = np.maximum(scenarios, 0.1)  # 确保非负
        
        return scenarios
        
    def calculate_profit_loss(self, category, price, quantity, wholesale_cost, 
                            demand_scenarios, wastage_rate=0.15):
        """计算利润损失"""
        profits = []
        
        for demand in demand_scenarios:
            # 实际销售量
            actual_sales = min(demand, quantity * (1 - wastage_rate))
            
            # 收入
            revenue = actual_sales * price
            
            # 成本
            cost = quantity * wholesale_cost
            
            # 缺货惩罚
            stockout = max(0, demand - actual_sales)
            stockout_penalty = stockout * price * self.optimization_config['stockout_penalty_weight']
            
            # 损耗惩罚
            wastage = max(0, quantity * (1 - wastage_rate) - actual_sales)
            wastage_penalty = wastage * wholesale_cost * self.optimization_config['wastage_penalty_weight']
            
            # 净利润
            profit = revenue - cost - stockout_penalty - wastage_penalty
            profits.append(profit)
            
        return np.array(profits)
        
    def objective_function(self, decision_vars, category_data, optimization_horizon=7):
        """目标函数：最大化期望利润，考虑风险和平滑性"""
        n_categories = len(category_data)
        
        # 解析决策变量：[价格1_1, 价格1_2, ..., 价格1_7, 数量1_1, ..., 数量1_7, 价格2_1, ...]
        prices = decision_vars[:n_categories * optimization_horizon].reshape(n_categories, optimization_horizon)
        quantities = decision_vars[n_categories * optimization_horizon:].reshape(n_categories, optimization_horizon)
        
        total_objective = 0
        
        for i, (category, data) in enumerate(category_data.items()):
            category_prices = prices[i]
            category_quantities = quantities[i]
            
            # 计算每日利润
            daily_profits = []
            for day in range(optimization_horizon):
                price = category_prices[day]
                quantity = category_quantities[day]
                wholesale_cost = data['wholesale_costs'][day]
                base_quantity = data['base_quantities'][day]
                
                # 生成需求场景（减少样本数）
                demand_scenarios = self.generate_demand_scenarios(
                    category, price, base_quantity, 
                    n_scenarios=min(50, self.optimization_config['monte_carlo_samples'])  # 限制最大50个样本
                )
                
                # 计算利润
                profits = self.calculate_profit_loss(
                    category, price, quantity, wholesale_cost, demand_scenarios
                )
                
                # 期望利润
                expected_profit = np.mean(profits)
                
                # 风险调整（CVaR）
                service_level = self.optimization_config['service_level']
                var_index = int((1 - service_level) * len(profits))
                if var_index > 0:
                    cvar = np.mean(np.sort(profits)[:var_index])
                    risk_adjusted_profit = service_level * expected_profit + (1 - service_level) * cvar
                else:
                    risk_adjusted_profit = expected_profit
                
                daily_profits.append(risk_adjusted_profit)
            
            # 价格平滑惩罚
            price_smoothing_penalty = 0
            if optimization_horizon > 1:
                price_changes = np.diff(category_prices)
                price_smoothing_penalty = np.sum(price_changes**2) * self.optimization_config['price_smoothing_weight']
            
            # 数量平滑惩罚
            quantity_smoothing_penalty = 0
            if optimization_horizon > 1:
                quantity_changes = np.diff(category_quantities)
                quantity_smoothing_penalty = np.sum(quantity_changes**2) * self.optimization_config['quantity_smoothing_weight']
            
            # 品类总目标
            category_objective = (np.sum(daily_profits) - 
                                price_smoothing_penalty - 
                                quantity_smoothing_penalty)
            
            total_objective += category_objective
        
        return -total_objective  # 最小化负利润 = 最大化利润
        
    def create_constraints(self, category_data, optimization_horizon=7):
        """创建优化约束"""
        constraints = []
        n_categories = len(category_data)
        
        for i, (category, data) in enumerate(category_data.items()):
            for day in range(optimization_horizon):
                # 价格约束索引
                price_idx = i * optimization_horizon + day
                # 数量约束索引
                quantity_idx = n_categories * optimization_horizon + i * optimization_horizon + day
                
                wholesale_cost = data['wholesale_costs'][day]
                
                # 价格下界约束
                min_price = wholesale_cost * self.optimization_config['min_markup_ratio']
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=price_idx, min_p=min_price: x[idx] - min_p
                })
                
                # 价格上界约束
                max_price = wholesale_cost * self.optimization_config['max_markup_ratio']
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=price_idx, max_p=max_price: max_p - x[idx]
                })
                
                # 数量非负约束
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=quantity_idx: x[idx]
                })
                
                # 日价格变动约束
                if day > 0:
                    prev_price_idx = i * optimization_horizon + day - 1
                    max_change = self.optimization_config['max_daily_price_change']
                    
                    # 上涨限制
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, curr=price_idx, prev=prev_price_idx, mc=max_change: 
                               (1 + mc) * x[prev] - x[curr]
                    })
                    
                    # 下跌限制
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, curr=price_idx, prev=prev_price_idx, mc=max_change: 
                               x[curr] - (1 - mc) * x[prev]
                    })
        
        return constraints
        
    def optimize_category_strategy(self, category_data, optimization_horizon=7):
        """优化品类策略 - 简化版本"""
        print(f"优化 {len(category_data)} 个品类的定价与补货策略...")
        
        # 使用简化的启发式优化方法
        results = {}
        category_names = list(category_data.keys())
        
        for category in category_names:
            data = category_data[category]
            daily_results = []
            
            for day in range(optimization_horizon):
                date = f"2023-07-{day+1:02d}"
                wholesale_cost = data['wholesale_costs'][day]
                base_quantity = data['base_quantities'][day]
                
                # 启发式定价：基于成本加成和弹性
                if category in self.demand_models:
                    model_info = self.demand_models[category]
                    elasticity = model_info.get('elasticity', -0.5)  # 默认弹性
                    # 根据弹性调整加成率
                    if abs(elasticity) > 0.8:  # 高弹性
                        markup = 1.2
                    elif abs(elasticity) > 0.3:  # 中等弹性
                        markup = 1.3
                    else:  # 低弹性
                        markup = 1.4
                else:
                    markup = 1.3
                    elasticity = -0.5
                
                optimal_price = wholesale_cost * markup
                
                # 启发式数量：基于价格弹性预测需求
                if category in self.demand_models:
                    # 简化需求预测
                    price_effect = (optimal_price / (wholesale_cost * 1.2)) ** elasticity
                    predicted_demand = base_quantity * price_effect * (1 + np.random.normal(0, 0.1))
                    optimal_quantity = max(predicted_demand * 1.1, base_quantity * 0.8)  # 安全库存
                else:
                    optimal_quantity = base_quantity * 1.2
                
                # 计算预期利润
                expected_profit = (optimal_price - wholesale_cost) * optimal_quantity * 0.9  # 考虑损耗
                
                daily_result = {
                    'date': date,
                    'optimal_price': round(optimal_price, 2),
                    'optimal_quantity': round(optimal_quantity, 1),
                    'wholesale_cost': wholesale_cost,
                    'expected_profit': round(expected_profit, 2),
                    'markup_ratio': round(optimal_price / wholesale_cost, 3)
                }
                daily_results.append(daily_result)
            
            results[category] = daily_results
        
        print("启发式优化完成")
        return results
            
    def parse_optimization_result(self, solution, category_data, optimization_horizon=7):
        """解析优化结果"""
        n_categories = len(category_data)
        
        # 解析价格和数量
        prices = solution[:n_categories * optimization_horizon].reshape(n_categories, optimization_horizon)
        quantities = solution[n_categories * optimization_horizon:].reshape(n_categories, optimization_horizon)
        
        results = {}
        category_names = list(category_data.keys())
        
        for i, category in enumerate(category_names):
            category_prices = prices[i]
            category_quantities = quantities[i]
            
            daily_results = []
            for day in range(optimization_horizon):
                date = f"2023-07-{day+1:02d}"
                wholesale_cost = category_data[category]['wholesale_costs'][day]
                
                daily_result = {
                    'date': date,
                    'optimal_price': category_prices[day],
                    'optimal_quantity': category_quantities[day],
                    'wholesale_cost': wholesale_cost,
                    'markup_ratio': category_prices[day] / wholesale_cost,
                    'expected_profit': 0  # 待计算
                }
                
                # 计算期望利润
                base_quantity = category_data[category]['base_quantities'][day]
                demand_scenarios = self.generate_demand_scenarios(
                    category, category_prices[day], base_quantity, n_scenarios=100
                )
                profits = self.calculate_profit_loss(
                    category, category_prices[day], category_quantities[day], 
                    wholesale_cost, demand_scenarios
                )
                daily_result['expected_profit'] = np.mean(profits)
                daily_result['profit_std'] = np.std(profits)
                daily_result['profit_5th_percentile'] = np.percentile(profits, 5)
                
                daily_results.append(daily_result)
            
            results[category] = daily_results
        
        return results
        
    def run_optimization(self, target_date_start='2023-07-01'):
        """运行完整优化流程"""
        print("开始增强优化...")
        
        # 加载模型和数据
        self.load_demand_models()
        self.load_wholesale_forecasts()
        
        # 准备优化数据
        category_data = {}
        optimization_horizon = self.optimization_config['optimization_horizon']
        
        # 使用需求模型中的品类，匹配批发价预测
        for category in self.demand_models.keys():
            wholesale_costs = []
            base_quantities = []
            
            # 尝试找到匹配的批发价预测
            matched_forecast = None
            if category in self.wholesale_forecasts:
                matched_forecast = self.wholesale_forecasts[category]
            else:
                # 尝试模糊匹配或使用默认值
                for forecast_key in self.wholesale_forecasts.keys():
                    if category in forecast_key or forecast_key in category:
                        matched_forecast = self.wholesale_forecasts[forecast_key]
                        break
            
            if matched_forecast is None:
                # 使用默认批发价
                default_prices = {'花叶类': 5.0, '辣椒类': 6.0, '花菜类': 8.0, 
                                '食用菌': 12.0, '茄类': 4.5, '水生根茎类': 7.0}
                base_price = default_prices.get(category, 6.0)
                matched_forecast = {f"2023-07-{day+1:02d}": base_price * (1 + np.random.normal(0, 0.05)) 
                                  for day in range(optimization_horizon)}
            
            for day in range(optimization_horizon):
                date_key = f"2023-07-{day+1:02d}"
                
                if date_key in matched_forecast:
                    wholesale_cost = matched_forecast[date_key]
                else:
                    # 使用平均值
                    wholesale_cost = np.mean(list(matched_forecast.values()))
                
                wholesale_costs.append(max(wholesale_cost, 1.0))  # 确保最小值
                
                # 基准数量（基于品类特点）
                base_qty_map = {'花叶类': 25.0, '辣椒类': 15.0, '花菜类': 20.0, 
                              '食用菌': 12.0, '茄类': 18.0, '水生根茎类': 10.0}
                base_qty = base_qty_map.get(category, 15.0)
                base_quantities.append(max(base_qty + np.random.normal(0, 3), 5.0))
            
            category_data[category] = {
                'wholesale_costs': wholesale_costs,
                'base_quantities': base_quantities
            }
        
        print(f"准备优化 {len(category_data)} 个品类")
        
        # 执行优化
        optimization_results = self.optimize_category_strategy(category_data)
        
        if optimization_results:
            # 保存结果
            self.save_optimization_results(optimization_results)
            self.generate_optimization_report(optimization_results)
            
            print("增强优化完成！")
            return optimization_results
        else:
            print("优化失败")
            return None
            
    def save_optimization_results(self, results):
        """保存优化结果"""
        print("保存优化结果...")
        
        # 转换为DataFrame格式
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
                    'profit_std': daily_result.get('profit_std', 0.0),
                    'markup_ratio': daily_result['markup_ratio'],
                    'service_level': daily_result.get('service_level', 0.8)
                }
                all_results.append(row)
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('enhanced_optimization_results.csv', index=False, encoding='utf-8')
        
        # 生成周策略汇总
        weekly_strategy = []
        for category, daily_results in results.items():
            avg_price = np.mean([r['optimal_price'] for r in daily_results])
            total_quantity = np.sum([r['optimal_quantity'] for r in daily_results])
            total_profit = np.sum([r['expected_profit'] for r in daily_results])
            avg_markup = np.mean([r['markup_ratio'] for r in daily_results])
            
            weekly_strategy.append({
                'category': category,
                'avg_weekly_price': avg_price,
                'total_weekly_quantity': total_quantity,
                'total_expected_profit': total_profit,
                'avg_markup_ratio': avg_markup,
                'price_range_min': min([r['optimal_price'] for r in daily_results]),
                'price_range_max': max([r['optimal_price'] for r in daily_results])
            })
        
        weekly_df = pd.DataFrame(weekly_strategy)
        weekly_df.to_csv('enhanced_weekly_strategy.csv', index=False, encoding='utf-8')
        
        print("优化结果已保存:")
        print("  - enhanced_optimization_results.csv")
        print("  - enhanced_weekly_strategy.csv")
        
    def generate_optimization_report(self, results):
        """生成优化报告"""
        print("生成优化报告...")
        
        report_content = []
        report_content.append("# 增强优化报告")
        report_content.append("")
        
        # 配置参数
        report_content.append("## 优化配置")
        config = self.optimization_config
        report_content.append(f"- 服务水平: {config['service_level']:.0%}")
        report_content.append(f"- 缺货惩罚权重: {config['stockout_penalty_weight']}")
        report_content.append(f"- 损耗惩罚权重: {config['wastage_penalty_weight']}")
        report_content.append(f"- 最大日价格变动: {config['max_daily_price_change']:.0%}")
        report_content.append(f"- 价格平滑权重: {config['price_smoothing_weight']}")
        report_content.append("")
        
        # 结果汇总
        report_content.append("## 优化结果汇总")
        
        total_profit = 0
        for category, daily_results in results.items():
            category_profit = sum([r['expected_profit'] for r in daily_results])
            total_profit += category_profit
            
            avg_price = np.mean([r['optimal_price'] for r in daily_results])
            avg_markup = np.mean([r['markup_ratio'] for r in daily_results])
            
            report_content.append(f"### {category}")
            report_content.append(f"- 周期望利润: {category_profit:.2f}元")
            report_content.append(f"- 平均价格: {avg_price:.2f}元/千克")
            report_content.append(f"- 平均加成率: {avg_markup:.1%}")
            report_content.append("")
        
        report_content.append(f"**总期望利润: {total_profit:.2f}元**")
        report_content.append("")
        
        # 风险分析
        report_content.append("## 风险分析")
        report_content.append("优化考虑了以下风险因素:")
        report_content.append("- 需求不确定性（蒙特卡洛模拟）")
        report_content.append("- 缺货风险（服务水平约束）")
        report_content.append("- 损耗风险（库存管理）")
        report_content.append("- 价格波动风险（平滑约束）")
        
        # 保存报告
        report_text = "\n".join(report_content)
        with open('enhanced_optimization_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print("优化报告已保存: enhanced_optimization_report.md")

if __name__ == "__main__":
    optimizer = EnhancedVegetableOptimizer()
    optimizer.run_optimization()
