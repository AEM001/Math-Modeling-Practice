import pandas as pd

# Load category results
category_results = pd.read_csv('weekly_category_strategy.csv')

print("="*70)
print("问题二：蔬菜类商品未来一周的日补货总量和定价策略")
print("="*70)

print("\n【最终答案表格】")
print("以下为各蔬菜品类在2023年7月1-7日的每日最优补货总量和定价策略：")
print("\n表1：各品类每日补货总量（千克）")
print("-"*65)
print("品类名称    | 7月1日 | 7月2日 | 7月3日 | 7月4日 | 7月5日 | 7月6日 | 7月7日")
print("-"*65)

categories = ['花叶类', '水生根茎类', '茄类', '辣椒类', '食用菌']
dates = ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05', '2023-07-06', '2023-07-07']

for category in categories:
    cat_data = category_results[category_results['分类名称'] == category].sort_values('日期')
    row_data = []
    for date in dates:
        value = cat_data[cat_data['日期'] == date]['品类补货总量(千克)'].values
        if len(value) > 0:
            row_data.append(f"{value[0]:>6.1f}")
        else:
            row_data.append("   N/A")
    
    print(f"{category:<10} | {' | '.join(row_data)}")

# Total row
print("-"*65)
totals = []
for date in dates:
    daily_total = category_results[category_results['日期'] == date]['品类补货总量(千克)'].sum()
    totals.append(f"{daily_total:>6.1f}")
print(f"{'合计':<10} | {' | '.join(totals)}")

print("\n表2：各品类每日定价策略（加权平均售价，元/千克）")
print("-"*70)
print("品类名称    | 7月1日 | 7月2日 | 7月3日 | 7月4日 | 7月5日 | 7月6日 | 7月7日")
print("-"*70)

for category in categories:
    cat_data = category_results[category_results['分类名称'] == category].sort_values('日期')
    row_data = []
    for date in dates:
        value = cat_data[cat_data['日期'] == date]['品类别加权平均售价(元/千克)'].values
        if len(value) > 0:
            row_data.append(f"{value[0]:>6.2f}")
        else:
            row_data.append("  N/A")
    
    print(f"{category:<10} | {' | '.join(row_data)}")

print("\n表3：各品类每日预期收益（元）")
print("-"*65)
print("品类名称    | 7月1日 | 7月2日 | 7月3日 | 7月4日 | 7月5日 | 7月6日 | 7月7日")
print("-"*65)

for category in categories:
    cat_data = category_results[category_results['分类名称'] == category].sort_values('日期')
    row_data = []
    for date in dates:
        value = cat_data[cat_data['日期'] == date]['品类总利润(元)'].values
        if len(value) > 0:
            row_data.append(f"{value[0]:>6.2f}")
        else:
            row_data.append("  N/A")
    
    print(f"{category:<10} | {' | '.join(row_data)}")

# Total row
print("-"*65)
profit_totals = []
for date in dates:
    daily_profit = category_results[category_results['日期'] == date]['品类总利润(元)'].sum()
    profit_totals.append(f"{daily_profit:>6.2f}")
print(f"{'合计':<10} | {' | '.join(profit_totals)}")

print("\n【策略说明】")
print("1. 补货策略：基于需求预测模型，考虑损耗率，确定最优补货量")
print("2. 定价策略：通过价格弹性优化，在成本约束下实现利润最大化")
print("3. 总体收益：预期一周总利润¥3,225.24，日均利润¥460.75")
print("4. 重点品类：花叶类贡献最大利润，辣椒类价格弹性最高")

print(f"\n一周汇总：")
total_replenishment = category_results['品类补货总量(千克)'].sum()
total_profit = category_results['品类总利润(元)'].sum()
print(f"- 总补货量：{total_replenishment:.1f} 千克")
print(f"- 总预期收益：¥{total_profit:.2f}")
print(f"- 平均毛利率：{((total_profit / (total_replenishment * category_results['品类平均批发价(元/千克)'].mean())) * 100):>5.1f}%")

print("\n" + "="*70)