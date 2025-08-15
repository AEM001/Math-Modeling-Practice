import pandas as pd
import numpy as np

# Load all results files
daily_results = pd.read_csv('daily_optimization_results.csv')
category_results = pd.read_csv('weekly_category_strategy.csv')
demand_models = pd.read_csv('demand_model_results.csv')
validation_results = pd.read_csv('validation_results.csv')

print("="*80)
print("蔬菜类商品自动定价与补货决策 - 问题二最终结果汇总")
print("="*80)

print("\n【一、需求建模结果摘要】")
print(f"1. 成功建模单品数量：{len(demand_models)} 个")
print(f"2. 统计显著模型数量：{len(demand_models[demand_models['significant'] == True])} 个")
print(f"3. 平均拟合优度（R²）：{demand_models['r_squared'].mean():.3f}")
print(f"4. 价格弹性系数范围：[{demand_models['beta'].min():.3f}, {demand_models['beta'].max():.3f}]")

if len(validation_results) > 0:
    print(f"5. 验证集表现：")
    print(f"   - 平均RMSE：{validation_results['rmse'].mean():.2f}")
    print(f"   - 平均MAPE：{validation_results['mape'].mean():.2f}")

print("\n【二、品类级优化策略概览】")
# Calculate overall statistics
total_daily_replenishment = category_results.groupby('日期')['品类补货总量(千克)'].sum()
total_daily_profit = category_results.groupby('日期')['品类总利润(元)'].sum()

print("每日汇总统计：")
for i, date in enumerate(['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', 
                         '2023-07-05', '2023-07-06', '2023-07-07']):
    day_data = category_results[category_results['日期'] == date]
    print(f"  {date}: 补货总量 {day_data['品类补货总量(千克)'].sum():.1f}kg, "
          f"预期利润 ¥{day_data['品类总利润(元)'].sum():.2f}")

print(f"\n一周总计：")
print(f"  - 补货总量：{category_results['品类补货总量(千克)'].sum():.1f} kg")
print(f"  - 预期总利润：¥{category_results['品类总利润(元)'].sum():.2f}")
print(f"  - 平均日利润：¥{category_results['品类总利润(元)'].sum()/7:.2f}")

print("\n【三、各品类每日补货总量和定价策略】")
print("注：售价为加权平均售价，反映各品类整体定价水平")
print("-"*80)

categories = category_results['分类名称'].unique()
for category in categories:
    print(f"\n■ {category}")
    cat_data = category_results[category_results['分类名称'] == category].sort_values('日期')
    print("日期       | 补货总量(kg) | 加权平均售价(元/kg) | 预期利润(元)")
    print("-"*65)
    for _, row in cat_data.iterrows():
        print(f"{row['日期']} |  {row['品类补货总量(千克)']:>10.1f} |  {row['品类别加权平均售价(元/千克)']:>15.2f} | {row['品类总利润(元)']:>10.2f}")
    
    # Category summary
    avg_replenishment = cat_data['品类补货总量(千克)'].mean()
    avg_price = (cat_data['品类别加权平均售价(元/千克)'] * cat_data['品类销量总量(千克)']).sum() / cat_data['品类销量总量(千克)'].sum()
    total_profit = cat_data['品类总利润(元)'].sum()
    
    print(f"  小计：周平均补货量 {avg_replenishment:.1f}kg, 周平均售价 ¥{avg_price:.2f}/kg, 周总利润 ¥{total_profit:.2f}")

print("\n【四、定价策略分析】")
# Analysis of pricing patterns
category_pricing = category_results.groupby('分类名称').agg({
    '品类别加权平均售价(元/千克)': 'mean',
    '品类平均批发价(元/千克)': 'mean',
    '品类总利润(元)': 'sum'
})

category_pricing['成本加成率'] = (category_pricing['品类别加权平均售价(元/千克)'] / 
                               category_pricing['品类平均批发价(元/千克)']) - 1

print("各品类定价策略特征：")
print("品类       | 平均售价(元/kg) | 平均批发价(元/kg) | 成本加成率 | 周总利润(元)")
print("-"*78)
for category, row in category_pricing.iterrows():
    print(f"{category:<10} | {row['品类别加权平均售价(元/千克)']:>13.2f} | {row['品类平均批发价(元/千克)']:>15.2f} | "
          f"{row['成本加成率']:>8.1%} | {row['品类总利润(元)']:>9.2f}")

print("\n【五、模型特点与应用建议】")
print("1. 模型优势：")
print("   - 基于历史数据的双对数需求模型，能准确捕捉价格弹性")
print("   - 考虑了损耗率，更贴近实际经营情况") 
print("   - 实现了单品级精细化建模，然后聚合到品类级决策")
print("   - 结合批发价预测，提供动态定价策略")

print("\n2. 应用建议：")
print("   - 建议重点关注高利润品类的库存管理")
print("   - 价格弹性较大的品类适合采用促销策略")
print("   - 定期更新模型参数以适应市场变化")
print("   - 结合天气、节假日等因素进一步优化预测")

print("\n" + "="*80)
print("报告完成 - 已为2023年7月1-7日提供最优补货与定价策略")
print("="*80)