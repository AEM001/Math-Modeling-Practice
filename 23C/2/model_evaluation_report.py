import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_model_results():
    """Load and analyze model results"""
    print("="*80)
    print("蔬菜类商品自动定价与补货决策 - 模型效果评估报告")
    print("="*80)
    
    # Load data
    try:
        demand_models = pd.read_csv('demand_model_results.csv')
        validation_results = pd.read_csv('validation_results.csv')
        daily_results = pd.read_csv('daily_optimization_results.csv')
        category_results = pd.read_csv('weekly_category_strategy.csv')
        
        print("✅ 所有结果文件加载成功")
    except FileNotFoundError as e:
        print(f"❌ 文件加载失败: {e}")
        return
    
    print(f"\n【数据集基础信息】")
    print(f"- 建模单品总数：{len(demand_models)} 个")
    print(f"- 成功验证单品数：{len(validation_results)} 个") 
    print(f"- 优化结果记录数：{len(daily_results)} 条")
    print(f"- 品类策略记录数：{len(category_results)} 条")
    
    return demand_models, validation_results, daily_results, category_results

def evaluate_demand_models(demand_models, validation_results):
    """Evaluate demand model performance"""
    print(f"\n【需求建模效果评估】")
    
    # Model fit statistics
    print(f"\n1. 模型拟合质量：")
    print(f"   - 平均R²：{demand_models['r_squared'].mean():.3f}")
    print(f"   - R²中位数：{demand_models['r_squared'].median():.3f}")
    print(f"   - R²标准差：{demand_models['r_squared'].std():.3f}")
    print(f"   - R² > 0.3的模型：{len(demand_models[demand_models['r_squared'] > 0.3])} 个 ({len(demand_models[demand_models['r_squared'] > 0.3])/len(demand_models)*100:.1f}%)")
    print(f"   - R² > 0.5的模型：{len(demand_models[demand_models['r_squared'] > 0.5])} 个 ({len(demand_models[demand_models['r_squared'] > 0.5])/len(demand_models)*100:.1f}%)")
    
    # Statistical significance
    significant_models = demand_models[demand_models['significant'] == True]
    print(f"\n2. 统计显著性：")
    print(f"   - 统计显著的模型：{len(significant_models)} 个 ({len(significant_models)/len(demand_models)*100:.1f}%)")
    print(f"   - 显著模型的平均R²：{significant_models['r_squared'].mean():.3f}")
    
    # Price elasticity analysis
    print(f"\n3. 价格弹性分析：")
    print(f"   - 弹性系数范围：[{demand_models['beta'].min():.3f}, {demand_models['beta'].max():.3f}]")
    print(f"   - 弹性系数均值：{demand_models['beta'].mean():.3f}")
    print(f"   - 弹性系数中位数：{demand_models['beta'].median():.3f}")
    
    # Elasticity categories
    elastic_models = demand_models[demand_models['beta'] < -1]
    inelastic_models = demand_models[demand_models['beta'] >= -1]
    print(f"   - 高弹性商品 (|β| > 1)：{len(elastic_models)} 个 ({len(elastic_models)/len(demand_models)*100:.1f}%)")
    print(f"   - 低弹性商品 (|β| ≤ 1)：{len(inelastic_models)} 个 ({len(inelastic_models)/len(demand_models)*100:.1f}%)")
    
    # Validation performance
    if len(validation_results) > 0:
        print(f"\n4. 模型验证表现：")
        print(f"   - 验证成功率：{len(validation_results)/len(demand_models)*100:.1f}%")
        print(f"   - 平均RMSE：{validation_results['rmse'].mean():.2f}")
        print(f"   - RMSE中位数：{validation_results['rmse'].median():.2f}")
        print(f"   - 平均MAPE：{validation_results['mape'].mean():.2f}")
        print(f"   - MAPE中位数：{validation_results['mape'].median():.2f}")
        
        # Good performance models
        good_models = validation_results[validation_results['mape'] < 1.0]
        print(f"   - MAPE < 1.0的模型：{len(good_models)} 个 ({len(good_models)/len(validation_results)*100:.1f}%)")

def evaluate_category_performance(demand_models, category_results):
    """Evaluate category-level performance"""
    print(f"\n【品类级表现评估】")
    
    # Category model coverage
    category_model_count = demand_models.groupby('分类名称').size()
    print(f"\n1. 各品类建模覆盖：")
    for category, count in category_model_count.items():
        print(f"   - {category}：{count} 个单品")
    
    # Weekly performance by category
    weekly_performance = category_results.groupby('分类名称').agg({
        '品类补货总量(千克)': 'sum',
        '品类销量总量(千克)': 'sum', 
        '品类总利润(元)': 'sum',
        '品类别加权平均售价(元/千克)': 'mean'
    }).round(2)
    
    print(f"\n2. 各品类周度表现汇总：")
    print(f"{'品类名称':<12} | {'周补货量(kg)':<12} | {'周销量(kg)':<12} | {'周利润(元)':<12} | {'平均售价(元/kg)'}")
    print("-" * 80)
    for category, row in weekly_performance.iterrows():
        print(f"{category:<12} | {row['品类补货总量(千克)']:>10.1f} | {row['品类销量总量(千克)']:>10.1f} | {row['品类总利润(元)']:>10.2f} | {row['品类别加权平均售价(元/千克)']:>12.2f}")
    
    # Performance ranking
    weekly_performance_sorted = weekly_performance.sort_values('品类总利润(元)', ascending=False)
    print(f"\n3. 品类利润贡献排名：")
    for i, (category, row) in enumerate(weekly_performance_sorted.iterrows(), 1):
        contribution = row['品类总利润(元)'] / weekly_performance['品类总利润(元)'].sum() * 100
        print(f"   {i}. {category}：¥{row['品类总利润(元)']:.2f} ({contribution:.1f}%)")

def evaluate_optimization_results(daily_results):
    """Evaluate optimization results"""
    print(f"\n【优化算法效果评估】")
    
    # Daily trends
    daily_summary = daily_results.groupby('日序号').agg({
        '最大利润(元)': 'sum',
        '最优销量(千克)': 'sum',
        '最优补货量(千克)': 'sum',
        '加成率': 'mean'
    }).round(2)
    
    print(f"\n1. 每日优化结果趋势：")
    print(f"{'日期':<12} | {'利润(元)':<10} | {'销量(kg)':<10} | {'补货量(kg)':<12} | {'平均加成率'}")
    print("-" * 65)
    dates = ['7月1日', '7月2日', '7月3日', '7月4日', '7月5日', '7月6日', '7月7日']
    for i, (day, row) in enumerate(daily_summary.iterrows()):
        print(f"{dates[i]:<12} | {row['最大利润(元)']:>8.2f} | {row['最优销量(千克)']:>8.1f} | {row['最优补货量(千克)']:>10.1f} | {row['加成率']:>9.1%}")
    
    # Optimization effectiveness
    total_profit = daily_summary['最大利润(元)'].sum()
    avg_daily_profit = total_profit / 7
    profit_growth = (daily_summary['最大利润(元)'].iloc[-1] - daily_summary['最大利润(元)'].iloc[0]) / daily_summary['最大利润(元)'].iloc[0]
    
    print(f"\n2. 优化效果指标：")
    print(f"   - 一周总利润：¥{total_profit:.2f}")
    print(f"   - 日均利润：¥{avg_daily_profit:.2f}")
    print(f"   - 利润增长率：{profit_growth:.1%} (首日vs末日)")
    print(f"   - 总补货量：{daily_summary['最优补货量(千克)'].sum():.1f} kg")
    print(f"   - 日均补货量：{daily_summary['最优补货量(千克)'].mean():.1f} kg")

def model_limitations_and_improvements():
    """Discuss model limitations and potential improvements"""
    print(f"\n【模型局限性与改进建议】")
    
    print(f"\n1. 当前模型局限性：")
    print(f"   - 部分单品样本量较少，模型稳定性有待提升")
    print(f"   - 未考虑季节性、节假日等外部因素影响")
    print(f"   - 批发价预测相对简单，可能影响优化准确性")
    print(f"   - 品类间替代效应未纳入考虑")
    print(f"   - 库存约束和空间限制未充分考虑")
    
    print(f"\n2. 模型改进方向：")
    print(f"   - 增加时间序列特征，考虑季节性和趋势")
    print(f"   - 引入天气、节假日等外部变量")
    print(f"   - 改进批发价预测模型（ARIMA、LSTM等）")
    print(f"   - 考虑品类间的交叉价格弹性")
    print(f"   - 加入库存管理和空间约束")
    print(f"   - 实施动态学习机制，定期更新模型参数")
    
    print(f"\n3. 实际应用建议：")
    print(f"   - 建议每周重新训练模型以适应市场变化")
    print(f"   - 对高价值、高弹性商品应加强监控")
    print(f"   - 结合人工经验对模型结果进行调整")
    print(f"   - 建立模型预警机制，识别异常情况")
    print(f"   - 收集更多外部数据以提升预测精度")

def main():
    """Main evaluation function"""
    try:
        # Load data
        demand_models, validation_results, daily_results, category_results = load_and_analyze_model_results()
        
        # Conduct evaluations
        evaluate_demand_models(demand_models, validation_results)
        evaluate_category_performance(demand_models, category_results)
        evaluate_optimization_results(daily_results)
        model_limitations_and_improvements()
        
        print(f"\n" + "="*80)
        print("📋 模型评估完成")
        print(f"\n总体评价：")
        print(f"✅ 成功建立了基于价格弹性的需求预测模型")
        print(f"✅ 实现了品类级的补货和定价优化")
        print(f"✅ 模型具有一定的预测能力和实际应用价值")
        print(f"⚠️  部分模型拟合度有待提升，需要更多数据和特征工程")
        print(f"🔧 建议持续优化模型并结合实际业务调整策略")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误：{e}")

if __name__ == "__main__":
    main()