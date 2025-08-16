#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蔬菜类商品自动定价与补货决策 - 核心分析引擎
整合了数据预处理、需求建模和优化求解的完整功能
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class VegetableOptimizer:
    """蔬菜定价与补货优化器"""
    
    def __init__(self, data_file='单品级每日汇总表.csv'):
        """初始化优化器"""
        self.data_file = data_file
        self.df_original = None
        self.train_df = None
        self.test_df = None
        self.demand_models = {}
        self.wholesale_forecasts = {}
        self.optimization_results = []
        
    def load_and_prepare_data(self):
        """加载并预处理数据"""
        print("正在加载和预处理数据...")
        
        try:
            self.df_original = pd.read_csv(self.data_file)
        except FileNotFoundError:
            print(f"错误：找不到文件 {self.data_file}")
            return False
            
        # 数据清洗
        df_normal_sales = self.df_original[self.df_original['打折销量(千克)'] == 0].copy()
        
        # 处理缺失值和无效值
        key_columns = ['正常销量(千克)', '正常销售单价(元/千克)', '批发价格(元/千克)']
        df_normal_sales.dropna(subset=key_columns, inplace=True)
        
        # 过滤正数
        df_normal_sales = df_normal_sales[
            (df_normal_sales['正常销量(千克)'] > 0) &
            (df_normal_sales['正常销售单价(元/千克)'] > 0) &
            (df_normal_sales['批发价格(元/千克)'] > 0)
        ]
        
        # 日期处理
        df_normal_sales['销售日期'] = pd.to_datetime(df_normal_sales['销售日期'])
        df_normal_sales.sort_values(by='销售日期', inplace=True)
        
        # 数据集划分
        train_size = int(0.7 * len(df_normal_sales))
        self.train_df = df_normal_sales.iloc[:train_size]
        self.test_df = df_normal_sales.iloc[train_size:]
        
        print(f"数据加载完成：训练集 {len(self.train_df)} 条，测试集 {len(self.test_df)} 条")
        return True
        
    def fit_demand_model(self, group_df):
        """拟合单品的双对数需求模型"""
        if len(group_df) < 5:
            return None
        
        Q = group_df['正常销量(千克)'].values
        P = group_df['正常销售单价(元/千克)'].values
        
        ln_Q = np.log(Q)
        ln_P = np.log(P)
        
        X = sm.add_constant(ln_P)
        y = ln_Q
        
        try:
            model = sm.OLS(y, X).fit()
            
            alpha = model.params[0]
            beta = model.params[1]
            r_squared = model.rsquared
            p_beta = model.pvalues[1]
            
            return {
                'alpha': alpha,
                'beta': beta,
                'r_squared': r_squared,
                'p_beta': p_beta,
                'significant': p_beta < 0.05,
                'valid': (beta < 0) and (r_squared > 0.1),
                'samples': len(group_df),
                'model': model
            }
        except Exception as e:
            print(f"模型拟合错误: {e}")
            return None
    
    def train_demand_models(self):
        """训练所有单品的需求模型"""
        print("正在训练需求模型...")
        
        models = {}
        model_summary = []
        
        for item_code in self.train_df['单品编码'].unique():
            item_data = self.train_df[self.train_df['单品编码'] == item_code]
            
            if len(item_data) >= 5:
                model_result = self.fit_demand_model(item_data)
                
                if model_result and model_result['valid']:
                    models[item_code] = model_result
                    item_name = item_data['单品名称'].iloc[0]
                    category = item_data['分类名称'].iloc[0]
                    loss_rate = float(item_data['损耗率(%)'].iloc[0]) / 100
                    
                    model_summary.append({
                        '单品编码': item_code,
                        '单品名称': item_name,
                        '分类名称': category,
                        'alpha': model_result['alpha'],
                        'beta': model_result['beta'],
                        'r_squared': model_result['r_squared'],
                        'significant': model_result['significant'],
                        'samples': model_result['samples']
                    })
                    
                    # 保存模型参数
                    self.demand_models[str(item_code)] = {
                        'alpha': model_result['alpha'],
                        'beta': model_result['beta'],
                        'r_squared': model_result['r_squared'],
                        'item_name': item_name,
                        'category': category,
                        'loss_rate': loss_rate
                    }
        
        print(f"成功训练 {len(models)} 个有效模型")
        
        # 保存结果
        model_summary_df = pd.DataFrame(model_summary)
        model_summary_df.to_csv('demand_model_results.csv', index=False)
        
        with open('demand_models.json', 'w', encoding='utf-8') as f:
            json.dump(self.demand_models, f, ensure_ascii=False, indent=2)
            
        return len(models) > 0
    
    def validate_models(self):
        """验证模型性能"""
        print("正在验证模型性能...")
        
        validation_results = []
        
        def predict_demand(row, model_params):
            if pd.isna(row['正常销售单价(元/千克)']) or row['正常销售单价(元/千克)'] <= 0:
                return np.nan
            
            P = row['正常销售单价(元/千克)']
            alpha = model_params['alpha']
            beta = model_params['beta']
            
            ln_Q_pred = alpha + beta * np.log(P)
            return np.exp(ln_Q_pred)
        
        for item_code in self.demand_models.keys():
            item_code_int = int(item_code)
            if item_code_int in self.test_df['单品编码'].values:
                test_item = self.test_df[self.test_df['单品编码'] == item_code_int]
                
                if len(test_item) > 0:
                    model_params = self.demand_models[item_code]
                    
                    y_true = test_item['正常销量(千克)'].values
                    y_pred = [predict_demand(row, model_params) for _, row in test_item.iterrows()]
                    
                    valid_mask = ~np.isnan(y_pred)
                    if np.sum(valid_mask) > 0:
                        y_true_valid = y_true[valid_mask]
                        y_pred_valid = np.array(y_pred)[valid_mask]
                        
                        if len(y_true_valid) > 0:
                            mse = mean_squared_error(y_true_valid, y_pred_valid)
                            rmse = np.sqrt(mse)
                            mape = mean_absolute_percentage_error(y_true_valid, y_pred_valid)
                            
                            validation_results.append({
                                '单品编码': item_code,
                                'test_samples': len(y_true_valid),
                                'rmse': rmse,
                                'mape': mape,
                                'y_true_mean': np.mean(y_true_valid),
                                'y_pred_mean': np.mean(y_pred_valid)
                            })
        
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv('validation_results.csv', index=False)
        
        print(f"模型验证完成，验证了 {len(validation_results)} 个模型")
        return True
    
    def forecast_wholesale_prices(self, forecast_days=7):
        """预测批发价格"""
        print("正在预测批发价格...")
        
        df_normal = self.df_original[
            (self.df_original['打折销量(千克)'] == 0) &
            (self.df_original['正常销量(千克)'] > 0) &
            (self.df_original['批发价格(元/千克)'] > 0)
        ].copy()
        
        def forecast_single_item_price(item_df, forecast_days=7):
            item_df = item_df.sort_values('销售日期')
            prices = item_df['批发价格(元/千克)'].values
            
            if len(prices) < 5:
                return np.mean(prices) * np.ones(forecast_days)
            else:
                try:
                    model = ExponentialSmoothing(prices, trend='add', seasonal=None)
                    fitted_model = model.fit()
                    return fitted_model.forecast(steps=forecast_days)
                except:
                    return np.mean(prices) * np.ones(forecast_days)
        
        wholesale_forecasts = {}
        
        for item_code in self.demand_models.keys():
            item_data = df_normal[df_normal['单品编码'] == int(item_code)].copy()
            
            if len(item_data) >= 3:
                forecasts = forecast_single_item_price(item_data, forecast_days)
            else:
                forecasts = np.array([df_normal['批发价格(元/千克)'].mean()] * forecast_days)
            
            wholesale_forecasts[item_code] = forecasts.tolist()
        
        self.wholesale_forecasts = wholesale_forecasts
        
        with open('wholesale_forecasts.json', 'w', encoding='utf-8') as f:
            json.dump(wholesale_forecasts, f, ensure_ascii=False, indent=2)
            
        print("批发价格预测完成")
        return True
    
    def optimize_single_item_daily(self, alpha, beta, cost, loss_rate):
        """单品单日优化"""
        min_price = cost * 1.001
        max_price = cost * 2.0
        
        if min_price >= max_price:
            max_price = cost * 2.5
        
        def calculate_profit(P):
            if P <= 0:
                return -np.inf
            
            ln_Q = alpha + beta * np.log(P)
            
            if ln_Q > 50 or ln_Q < -50:
                return -np.inf
                
            Q = np.exp(ln_Q)
            
            if Q > 1e6 or Q <= 0 or not np.isfinite(Q):
                return -np.inf
            
            if loss_rate >= 1:
                return -np.inf
            R = Q / (1 - loss_rate)
            
            profit = Q * P - R * cost
            
            if not np.isfinite(profit):
                return -np.inf
                
            return profit
        
        price_range = np.linspace(min_price, max_price, 100)
        profits = [calculate_profit(p) for p in price_range]
        
        max_profit_index = np.argmax(profits)
        optimal_price = price_range[max_profit_index]
        max_profit = profits[max_profit_index]
        
        optimal_q = np.exp(alpha + beta * np.log(optimal_price))
        optimal_r = optimal_q / (1 - loss_rate)
        
        return {
            'optimal_price': float(optimal_price),
            'optimal_quantity': float(optimal_q),
            'optimal_replenishment': float(optimal_r),
            'max_profit': float(max_profit),
            'cost': float(cost),
            'loss_rate': float(loss_rate),
            'margin': float((optimal_price - cost) / optimal_price)
        }
    
    def optimize_pricing_strategy(self):
        """优化定价策略"""
        print("正在优化定价策略...")
        
        target_start_date = datetime(2023, 7, 1)
        target_dates = [target_start_date + timedelta(days=i) for i in range(7)]
        
        optimization_results = []
        
        for item_code in self.demand_models.keys():
            if item_code not in self.wholesale_forecasts:
                continue
            
            model = self.demand_models[item_code]
            
            if model['beta'] < -4 or model['beta'] > -0.1:
                continue
                
            forecasts = self.wholesale_forecasts[item_code]
            
            for day_idx in range(7):
                date = target_dates[day_idx]
                wholesale_price = forecasts[day_idx] if len(forecasts) > day_idx else forecasts[0]
                
                result = self.optimize_single_item_daily(
                    alpha=model['alpha'],
                    beta=model['beta'],
                    cost=wholesale_price,
                    loss_rate=model['loss_rate']
                )
                
                optimization_results.append({
                    '单品编码': item_code,
                    '单品名称': model['item_name'],
                    '分类名称': model['category'],
                    '日期': date.strftime('%Y-%m-%d'),
                    '日序号': day_idx + 1,
                    '批发价格(元/千克)': result['cost'],
                    '最优售价(元/千克)': result['optimal_price'],
                    '最优销量(千克)': result['optimal_quantity'],
                    '最优补货量(千克)': result['optimal_replenishment'],
                    '最大利润(元)': result['max_profit'],
                    '加成率': result['margin'],
                    '损耗率': result['loss_rate']
                })
        
        self.optimization_results = optimization_results
        
        # 保存单品级结果
        optimization_df = pd.DataFrame(optimization_results)
        optimization_df.to_csv('daily_optimization_results.csv', index=False)
        
        # 生成品类级汇总
        def calculate_weighted_price(group):
            weights = group['最优销量(千克)']
            prices = group['最优售价(元/千克)']
            if weights.sum() == 0 or len(prices) == 0:
                return prices.mean() if len(prices) > 0 else 0
            return (weights * prices).sum() / weights.sum()
        
        category_summary = optimization_df.groupby(['分类名称', '日期']).agg({
            '最优补货量(千克)': 'sum',
            '最优销量(千克)': 'sum',
            '最大利润(元)': 'sum',
            '批发价格(元/千克)': 'mean'
        }).reset_index()
        
        weighted_prices = optimization_df.groupby(['分类名称', '日期']).apply(calculate_weighted_price).reset_index()
        weighted_prices.columns = ['分类名称', '日期', '品类别加权平均售价(元/千克)']
        
        category_summary = category_summary.merge(weighted_prices, on=['分类名称', '日期'])
        
        category_summary = category_summary.rename(columns={
            '最优补货量(千克)': '品类补货总量(千克)',
            '最优销量(千克)': '品类销量总量(千克)',
            '最大利润(元)': '品类总利润(元)',
            '批发价格(元/千克)': '品类平均批发价(元/千克)'
        })
        
        category_summary.to_csv('weekly_category_strategy.csv', index=False)
        
        print(f"优化完成：{len(optimization_results)} 个单品-日优化结果")
        print(f"总预期利润：¥{optimization_df['最大利润(元)'].sum():.2f}")
        
        return True
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("="*60)
        print("开始蔬菜定价与补货优化分析")
        print("="*60)
        
        steps = [
            ("数据加载与预处理", self.load_and_prepare_data),
            ("训练需求模型", self.train_demand_models),
            ("验证模型性能", self.validate_models),
            ("预测批发价格", self.forecast_wholesale_prices),
            ("优化定价策略", self.optimize_pricing_strategy)
        ]
        
        for step_name, step_func in steps:
            print(f"\n正在执行：{step_name}")
            if not step_func():
                print(f"错误：{step_name} 失败")
                return False
            print(f"✅ {step_name} 完成")
        
        print("\n" + "="*60)
        print("🎉 分析完成！")
        print("生成的文件：")
        print("- demand_model_results.csv: 需求模型结果")
        print("- demand_models.json: 模型参数")
        print("- validation_results.csv: 模型验证结果")
        print("- wholesale_forecasts.json: 批发价预测")
        print("- daily_optimization_results.csv: 日优化结果")
        print("- weekly_category_strategy.csv: 品类策略")
        print("="*60)
        
        return True


def main():
    """主函数"""
    optimizer = VegetableOptimizer()
    success = optimizer.run_full_analysis()
    
    if success:
        print("\n分析成功完成！请查看生成的结果文件。")
    else:
        print("\n分析过程中出现错误，请检查数据文件和配置。")


if __name__ == "__main__":
    main()