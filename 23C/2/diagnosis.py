# -*- coding: utf-8 -*-
"""
诊断脚本：分析利润-加价曲线，找出为什么总是选择上限
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimizer import VegetableOptimizer
from font_config import setup_chinese_font

def diagnose_profit_curves():
    """诊断各品类的利润-加价曲线"""
    setup_chinese_font()
    
    # 创建优化器实例
    optimizer = VegetableOptimizer()
    optimizer.load_demand_models()
    
    # 测试参数
    base_quantities = {
        '花叶类': 25.0, '辣椒类': 15.0, '花菜类': 20.0,
        '食用菌': 12.0, '茄类': 18.0, '水生根茎类': 10.0
    }
    
    base_wholesale_prices = {
        '花叶类': 5.5, '辣椒类': 6.5, '花菜类': 9.0, 
        '食用菌': 13.0, '茄类': 4.8, '水生根茎类': 7.5
    }
    
    # 创建诊断图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('利润-加价率诊断分析（扩展区间1.2-2.2）', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    results = []
    
    for i, category in enumerate(optimizer.demand_models.keys()):
        ax = axes[i]
        
        wholesale_cost = base_wholesale_prices.get(category, 8.0)
        base_quantity = base_quantities.get(category, 15.0)
        elasticity = optimizer.demand_models[category]['price_elasticity']
        
        # 扩展的加价率区间（1.2到2.2）
        markup_ratios = np.linspace(1.2, 2.2, 50)
        profits = []
        demands = []
        
        print(f"\n=== {category} ===")
        print(f"弹性: {elasticity:.3f}")
        print(f"批发价: {wholesale_cost:.2f}")
        
        for markup in markup_ratios:
            profit = optimizer.evaluate_markup_profit(category, wholesale_cost, base_quantity, markup)
            profits.append(profit if profit != -float('inf') else np.nan)
            
            # 同时记录需求量
            price = wholesale_cost * markup
            demand, _ = optimizer.predict_demand(category, price, base_quantity)
            demands.append(demand)
        
        # 过滤有效的利润值
        valid_profits = [p for p in profits if not np.isnan(p) and p != -float('inf')]
        valid_markups = [markup_ratios[i] for i, p in enumerate(profits) if not np.isnan(p) and p != -float('inf')]
        
        if valid_profits:
            max_profit_idx = np.argmax(valid_profits)
            optimal_markup = valid_markups[max_profit_idx]
            optimal_profit = valid_profits[max_profit_idx]
            
            print(f"最优加价率: {optimal_markup:.3f}")
            print(f"最大利润: {optimal_profit:.2f}")
            
            # 记录结果
            results.append({
                'category': category,
                'elasticity': elasticity,
                'optimal_markup': optimal_markup,
                'optimal_profit': optimal_profit,
                'theoretical_markup': optimizer.calculate_theoretical_optimal_markup(elasticity)
            })
        
        # 绘制利润曲线
        ax.plot(markup_ratios, profits, 'b-', linewidth=2, label='利润曲线')
        if valid_profits:
            ax.scatter([optimal_markup], [optimal_profit], color='red', s=100, zorder=5, label='最优点')
        
        # 标记1.6-1.8区间
        ax.axvspan(1.6, 1.8, alpha=0.2, color='yellow', label='当前搜索区间')
        
        ax.set_xlabel('加价率')
        ax.set_ylabel('风险调整利润')
        ax.set_title(f'{category} (弹性={elasticity:.3f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加理论最优线
        theoretical_markup = optimizer.calculate_theoretical_optimal_markup(elasticity)
        ax.axvline(theoretical_markup, color='green', linestyle='--', alpha=0.7, label='理论最优')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/profit_markup_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出诊断结果
    print("\n=== 诊断结果汇总 ===")
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # 保存诊断结果
    results_df.to_csv('outputs/results/diagnosis_results.csv', index=False)
    
    print(f"\n诊断图已保存: outputs/figures/profit_markup_diagnosis.png")
    print(f"诊断数据已保存: outputs/results/diagnosis_results.csv")

if __name__ == "__main__":
    diagnose_profit_curves()
