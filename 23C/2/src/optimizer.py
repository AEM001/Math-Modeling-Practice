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
# from font_config import setup_chinese_font # 为了兼容性暂时禁用

class VegetableOptimizer:
    """蔬菜定价与补货优化器"""
    
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
        # 订货与仿真的全局默认损耗率
        self.default_wastage_rate = float(self.opt_config.get('wastage_rate', 0.05))
        # 保存训练数据中的起始日期以计算时间趋势
        self.time_series_start_date = pd.to_datetime(self.config['time_series_start_date'])
        # 用于归一化 time_trend 的分母，与训练数据跨度对齐
        self.time_trend_denominator = 365
        # 尝试将时间趋势的缩放与特征工程输出保持一致
        self._init_time_trend_scaler()
        # 从训练特征中初始化各品类的基准与需求上限
        self._init_category_baselines_and_caps()

    def _init_time_trend_scaler(self):
        """若可用，则使用训练特征文件来初始化时间趋势的归一化。"""
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
        """基于训练特征，计算各品类的 ln_price 基准与需求上限。"""
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
            # 计算 ln_price 的参考值（各品类均值）
            grp = df.groupby('分类名称')['ln_price']
            self.category_ln_price_ref = grp.mean().to_dict()
            # 聚合为“品类-日期”的销量（对 ln_quantity 取指数后求和）
            if 'ln_quantity' in df.columns:
                df['quantity'] = np.exp(df['ln_quantity']).clip(lower=1e-6)
                q = df.groupby(['分类名称', '销售日期'], as_index=False)['quantity'].sum()
                qtl = float(self.opt_config.get('demand_cap_quantile', 0.95))
                for cat, sub in q.groupby('分类名称'):
                    self.category_demand_cap[cat] = float(sub['quantity'].quantile(qtl))
        except Exception as e:
            print(f"Warning: failed to initialize category baselines/caps: {e}")
    
    def load_demand_models(self):
        """加载已训练的随机森林（或其他）模型及其元数据。"""
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
                    # 若结果中包含，则解析模型实际使用的特征列表
                    features = default_features
                    if 'features' in row and pd.notna(row['features']):
                        raw = str(row['features'])
                        try:
                            import json as _json
                            features = _json.loads(raw)
                        except Exception:
                            # 退化处理：去除括号/引号并按逗号分割
                            raw2 = raw.strip().strip('[]')
                            parts = [p.strip().strip("'\"") for p in raw2.split(',') if p.strip()]
                            if parts:
                                features = parts
                    ln_resid_std = row['ln_resid_std'] if 'ln_resid_std' in row and pd.notna(row['ln_resid_std']) else None
                    self.demand_models[category] = {
                        'model_object': model_obj,
                        'price_elasticity': row['price_elasticity'] if 'price_elasticity' in row else np.nan,
                        'exog_features': features, # 存储训练时实际使用的特征列表
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
        """使用已加载的模型预测需求（随机森林以 ln_quantity 为目标）。
        为与训练特征对齐，将构造关键特征：
        - ln_price、ln_wholesale、ln_markup_ratio
        - time_trend、is_weekend、weekday 独热编码
        其他基于历史的特征（滞后/滚动）设为中性默认值。
        """
        if category not in self.demand_models:
            # 若无可用模型，则返回默认基线需求与较高不确定性
            return 15.0, 15.0 * 0.5 

        model_info = self.demand_models[category]
        model = model_info['model_object']
        features = model_info['exog_features']

        # 1. 构造外生特征向量（X）
        date = pd.to_datetime(date)
        exog_data = pd.DataFrame(index=[0])
        
        # 价格相关特征
        if 'ln_price' in features:
            exog_data['ln_price'] = float(np.log(max(price, 1e-6)))
        if 'ln_wholesale' in features:
            if wholesale_cost is None:
                # 若未提供，则用价格与假定加价率进行近似
                approx_wholesale = max(price / max(self.opt_config.get('min_markup_ratio', 1.6), 1e-6), 1e-6)
                exog_data['ln_wholesale'] = float(np.log(approx_wholesale))
            else:
                exog_data['ln_wholesale'] = float(np.log(max(wholesale_cost, 1e-6)))
        if 'ln_markup_ratio' in features:
            w = wholesale_cost if wholesale_cost is not None else max(price / max(self.opt_config.get('min_markup_ratio', 1.6), 1e-6), 1e-6)
            exog_data['ln_markup_ratio'] = float(np.log(max(price / max(w, 1e-6), 1e-6)))
        if 'ln_relative_price' in features:
            # 在缺少同类对比时，以加价率作为相对价格的替代
            w = wholesale_cost if wholesale_cost is not None else max(price / max(self.opt_config.get('min_markup_ratio', 1.6), 1e-6), 1e-6)
            exog_data['ln_relative_price'] = float(np.log(max(price / max(w, 1e-6), 1e-6)))

        # 日历类特征
        if 'is_weekend' in features:
            exog_data['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        if 'time_trend' in features:
            # 计算自（训练）起始日期以来的天数，并按与训练相同的跨度进行归一化
            days_since_start = max(0, (date - self.time_series_start_date).days)
            denom = max(1, getattr(self, 'time_trend_denominator', 365))
            exog_data['time_trend'] = min(days_since_start / denom, 1.0)
        # 工作日独热编码（weekday_0..weekday_6）
        for wd in range(7):
            col = f'weekday_{wd}'
            if col in features:
                exog_data[col] = 1 if date.weekday() == wd else 0

        # 确保所需特征齐全；缺失项以 0.0 作为安全默认值
        missing_cols = [col for col in features if col not in exog_data.columns]
        for col in missing_cols:
            exog_data[col] = 0.0
        # 确保列顺序与训练时一致
        exog_data = exog_data[features]
        # 确保为数值类型
        exog_data = exog_data.astype(float)

        # 2. 使用模型进行预测（随机森林输出 ln_quantity 的预测值）
        try:
            ln_pred_val = float(model.predict(exog_data)[0])
            if not np.isfinite(ln_pred_val):
                raise ValueError(f"ln_pred not finite: {ln_pred_val}")
            # 融合一个负的价格弹性先验，以加强与价格的单调关系
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

        # 3. 估计不确定性
        demand_std = None
        ln_sigma = model_info.get('ln_resid_std', None)
        if ln_sigma is not None and np.isfinite(ln_sigma) and ln_sigma > 0:
            demand_std = max(predicted_demand * float(ln_sigma), predicted_demand * 0.05)

        if demand_std is None or not np.isfinite(demand_std) or demand_std <= 0:
            # 退化处理：根据测试集 R^2 得到保守的标准差
            uncertainty_factor = np.sqrt(1 - max(0, model_info.get('test_r2', 0.5)))
            demand_std = max(predicted_demand * uncertainty_factor * 0.5, predicted_demand * 0.05)
        
        # 将需求上限限定在合理的历史分位数以避免不现实的尖峰
        cap = self.category_demand_cap.get(category)
        if cap is not None and np.isfinite(cap) and cap > 0:
            predicted_demand = float(min(predicted_demand, cap * 1.1))
        return predicted_demand, demand_std
    
    def calculate_profit_scenarios(self, category, price, date, quantity, wholesale_cost, 
                                 n_scenarios=20, wastage_rate=None):
        """基于模型的需求预测来计算利润场景。
        改进点：
        - 在需求预测中使用 wholesale_cost，以保持特征一致性。
        - 明确建模缺货惩罚与损耗（剩余）惩罚。
        - 不预先用损耗减少可用库存；损耗只作用于剩余部分。
        返回每个场景的利润数组。
        """
        if wastage_rate is None:
            wastage_rate = self.default_wastage_rate
        mean_demand, std_demand = self.predict_demand(category, price, date, wholesale_cost=wholesale_cost)
        
        if std_demand <= 0:
            std_demand = mean_demand * 0.1 # 为不确定性设置下限

        np.random.seed(hash(date) % (2**32 - 1)) # 使用日期作为随机种子以保持一致性
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
        """模拟需求以计算用于报告与优化的期望指标。"""
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
        # 汇总指标
        eps = 1e-9
        expected = {
            'expected_profit': float(np.mean(profit)),
            'profit_std': float(np.std(profit)),
            'expected_revenue': float(np.mean(revenue)),
            'expected_cost': float(cost),  # 此处不同场景下成本不变
            'expected_sales': float(np.mean(actual_sales)),
            'expected_leftover': float(np.mean(leftover)),
            'expected_stockout': float(np.mean(stockout)),
            'service_rate': float(np.mean(np.where(demand > eps, actual_sales / np.maximum(demand, eps), 1.0)))
        }
        return expected

    def evaluate_markup_profit(self, category, wholesale_cost, date, markup_ratio):
        """通过预测需求来评估给定加价率下的利润。"""
        price = wholesale_cost * markup_ratio
        
        # 预测需求并基于安全库存计算订货量
        predicted_demand, std_demand = self.predict_demand(category, price, date, wholesale_cost=wholesale_cost)
        order_quantity = self.calculate_order_quantity(predicted_demand, std_demand)

        # 使用蒙特卡洛模拟计算期望利润
        metrics = self.simulate_metrics(
            category, price, date, order_quantity, wholesale_cost,
            n_scenarios=self.opt_config['monte_carlo_samples']
        )
        
        expected_profit = metrics['expected_profit']
        profit_std = metrics['profit_std']
        
        # 风险调整后的利润（对波动进行惩罚）
        risk_penalty = self.opt_config.get('risk_aversion_factor', 0.1) * profit_std
        return expected_profit - risk_penalty

    def golden_section_search(self, category, wholesale_cost, date, a, b, tol=1e-3):
        """使用黄金分割搜索最优加价率。"""
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi
        
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        
        # 注意：我们在最大化利润，因此不取函数值的相反数
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
        """基于服务水平计算订货量，包含安全库存。
        通过考虑损耗来保证 R*(1-W) \u003e= Q。
        """
        if wastage_rate is None:
            wastage_rate = self.default_wastage_rate
        service_level = self.opt_config['service_level']
        # 标准正态分布的 Z 分数
        z_score_map = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.98: 2.05}
        safety_factor = z_score_map.get(service_level, 1.65) # 默认按 95%
        
        safety_stock = safety_factor * std_demand
        # 放大订货量以确保扣除损耗后的净可用量满足期望需求
        net_required = predicted_demand + safety_stock
        order_quantity = net_required / max(1e-6, (1 - max(wastage_rate, 0.0)))
        return max(0.0, order_quantity) # 保证非负

    def optimize_category(self, category, wholesale_cost, date):
        """在特定日期为某个品类优化价格与订货量。"""
        if category not in self.demand_models:
            # 若无模型可用，采用默认策略
            optimal_markup = np.mean([self.opt_config['min_markup_ratio'], self.opt_config['max_markup_ratio']])
            optimal_price = wholesale_cost * optimal_markup
            # 无法预测需求，返回占位订货量
            return optimal_price, 10.0

        # 从配置中读取加价率搜索边界
        min_markup = self.opt_config['min_markup_ratio']
        max_markup = self.opt_config['max_markup_ratio']

        # 使用黄金分割搜索得到最优加价率
        optimal_markup = self.golden_section_search(
            category, wholesale_cost, date, min_markup, max_markup
        )
        optimal_price = wholesale_cost * optimal_markup

        predicted_demand, std_demand = self.predict_demand(category, optimal_price, date, wholesale_cost=wholesale_cost)
        optimal_quantity = self.calculate_order_quantity(predicted_demand, std_demand)
        
        return optimal_price, optimal_quantity

    def generate_wholesale_forecasts(self, categories):
        """为给定品类生成简化的批发价预测。"""
        forecasts = {}
        horizon = self.opt_config['optimization_horizon']
        base_prices = self.config['base_wholesale_prices']

        # 确定性的伪预测以保证可复现性
        rng = np.random.default_rng(20230701)
        for category in categories:
            base_price = base_prices.get(category, 5.0) # 若配置缺失则使用默认价格
            days = np.arange(horizon)
            seasonal_effect = 0.1 * np.sin(2 * np.pi * days / 7) # 周期为一周的季节性
            noise = rng.normal(0, 0.02, horizon)
            price_multipliers = 1 + seasonal_effect + noise
            forecasts[category] = base_price * price_multipliers
        return forecasts

    def run_daily_optimization(self):
        """对所有拥有模型的品类执行每日优化循环。"""
        print("\nStarting daily optimization...")
        categories_with_models = list(self.demand_models.keys())
        if not categories_with_models:
            print("No models loaded, cannot run optimization.")
            return pd.DataFrame()
            
        wholesale_forecasts = self.generate_wholesale_forecasts(categories_with_models)
        
        results = []
        horizon = self.opt_config['optimization_horizon']
        # 按 23C 要求使用固定的目标周起始日期（2023-07-01），可通过配置覆盖
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
                
                # 重新计算用于报告的期望利润与扩展指标
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
        """将优化结果保存为 CSV 文件。"""
        if not os.path.exists(self.output_paths['results_dir']):
            os.makedirs(self.output_paths['results_dir'])
        path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
        results_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\nOptimization results saved to {path}")

    def plot_price_quantity_profit(self, category, wholesale_cost, date):
        """绘制给定日期下的价格-需求-利润关系曲线。"""
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
                # 在绘图前确保该品类存在于结果中
                if cat in optimization_results['category'].unique():
                    first_day_cost = optimization_results[optimization_results['category'] == cat]['wholesale_cost'].iloc[0]
                    optimizer.plot_price_quantity_profit(cat, first_day_cost, today)
    else:
        print("No demand models were loaded. Cannot run optimization.")
