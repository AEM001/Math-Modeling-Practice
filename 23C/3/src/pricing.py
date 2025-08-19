import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import OUTPUT_PATHS, TARGET_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_pricing_metrics(solution_df):
    """
    计算定价相关指标
    """
    logger.info("Calculating pricing metrics...")
    
    df = solution_df.copy()
    
    # 基本定价指标
    df['毛利润(元/kg)'] = df['售价(元/kg)'] - df['预测批发价(元/kg)']
    df['毛利率'] = df['毛利润(元/kg)'] / df['售价(元/kg)']
    df['单位利润率'] = df['加成率'] / (1 + df['加成率'])
    
    # 收益相关指标
    df['单位收益(元/kg)'] = df['售价(元/kg)']
    df['总收益(元)'] = df['预测销量(kg)'] * df['售价(元/kg)']
    df['库存周转率'] = df['预测销量(kg)'] / df['进货量(kg)']
    
    # 风险指标
    df['库存风险'] = np.where(df['库存周转率'] < 0.8, '高', 
                          np.where(df['库存周转率'] < 0.95, '中', '低'))
    
    # 价格竞争力（相对于历史平均价格）
    # 这里用预测批发价作为参考
    df['价格竞争力'] = np.where(df['售价(元/kg)'] / df['预测批发价(元/kg)'] <= 1.3, '高',
                           np.where(df['售价(元/kg)'] / df['预测批发价(元/kg)'] <= 1.5, '中', '低'))
    
    logger.info("Pricing metrics calculated successfully")
    return df

def generate_pricing_summary(solution_df):
    """
    生成定价摘要
    """
    logger.info("Generating pricing summary...")
    
    if len(solution_df) == 0:
        return {}
    
    summary = {
        # 基本统计
        'total_products': len(solution_df),
        'total_stock': solution_df['进货量(kg)'].sum(),
        'total_revenue': solution_df['预估收入(元)'].sum(),
        'total_cost': solution_df['进货成本(元)'].sum(),
        'total_profit': solution_df['预估利润(元)'].sum(),
        
        # 定价统计
        'avg_selling_price': solution_df['售价(元/kg)'].mean(),
        'price_range': (solution_df['售价(元/kg)'].min(), solution_df['售价(元/kg)'].max()),
        'avg_markup': solution_df['加成率'].mean(),
        'markup_range': (solution_df['加成率'].min(), solution_df['加成率'].max()),
        
        # 利润指标
        'profit_margin': solution_df['预估利润(元)'].sum() / solution_df['预估收入(元)'].sum(),
        'avg_profit_per_kg': solution_df['预估利润(元)'].sum() / solution_df['预测销量(kg)'].sum(),
        
        # 销量统计
        'total_predicted_sales': solution_df['预测销量(kg)'].sum(),
        'avg_sales_per_product': solution_df['预测销量(kg)'].mean(),
        
        # 品类分布
        'category_count': solution_df['分类名称'].nunique(),
        'category_distribution': solution_df.groupby('分类名称').agg({
            '预估利润(元)': 'sum',
            '预测销量(kg)': 'sum',
            '售价(元/kg)': 'mean'
        }).to_dict(),
        
        # 风险评估
        'high_risk_products': len(solution_df[solution_df.get('库存风险', '') == '高']),
        'avg_inventory_turnover': solution_df.get('库存周转率', pd.Series([1])).mean()
    }
    
    logger.info("Pricing summary generated successfully")
    return summary

def create_final_plan(solution_df):
    """
    创建最终的补货计划
    """
    logger.info("Creating final restocking plan...")
    
    # 按照要求的格式整理数据
    plan_df = solution_df.copy()
    
    # 确保包含所有必需列
    required_columns = [
        '单品编码', '单品名称', '分类编码', '分类名称',
        '进货量(kg)', '加成率', '售价(元/kg)', 
        '预测销量(kg)', '预测批发价(元/kg)', '预估利润(元)'
    ]
    
    # 检查并填补缺失列
    for col in required_columns:
        if col not in plan_df.columns:
            if col == '分类编码':
                # 分类编码缺失时用默认值填补
                plan_df['分类编码'] = '1011010000'
            else:
                plan_df[col] = 0
    
    # 选择最终列
    plan_df = plan_df[required_columns].copy()
    
    # 按预估利润降序排列
    plan_df = plan_df.sort_values('预估利润(元)', ascending=False)
    
    # 添加排名
    plan_df['利润排名'] = range(1, len(plan_df) + 1)
    
    # 数值格式化
    plan_df['进货量(kg)'] = plan_df['进货量(kg)'].round(2)
    plan_df['加成率'] = plan_df['加成率'].round(4)
    plan_df['售价(元/kg)'] = plan_df['售价(元/kg)'].round(2)
    plan_df['预测销量(kg)'] = plan_df['预测销量(kg)'].round(2)
    plan_df['预测批发价(元/kg)'] = plan_df['预测批发价(元/kg)'].round(2)
    plan_df['预估利润(元)'] = plan_df['预估利润(元)'].round(2)
    
    logger.info(f"Final plan created with {len(plan_df)} products")
    
    return plan_df

def export_results(solution_df, summary, output_dir=None):
    """
    导出结果到文件
    """
    if output_dir is None:
        output_dir = OUTPUT_PATHS['results_dir']
    
    logger.info(f"Exporting results to {output_dir}...")
    
    # 确保输出目录存在
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 导出主要计划CSV
    plan_df = create_final_plan(solution_df)
    plan_csv_path = OUTPUT_PATHS['plan_csv']
    plan_df.to_csv(plan_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Main plan exported to {plan_csv_path}")
    
    # 2. 导出详细结果
    detailed_df = calculate_pricing_metrics(solution_df)
    detailed_csv_path = f"{output_dir}/detailed_results_{TARGET_DATE.replace('-', '')}.csv"
    detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Detailed results exported to {detailed_csv_path}")
    
    # 3. 导出摘要报告
    summary_path = f"{output_dir}/summary_report_{TARGET_DATE.replace('-', '')}.txt"
    export_summary_report(summary, plan_df, summary_path)
    
    # 4. 导出品类分析
    category_analysis_path = f"{output_dir}/category_analysis_{TARGET_DATE.replace('-', '')}.csv"
    export_category_analysis(plan_df, category_analysis_path)
    
    logger.info("All results exported successfully")
    
    return {
        'plan_csv': plan_csv_path,
        'detailed_csv': detailed_csv_path,
        'summary_report': summary_path,
        'category_analysis': category_analysis_path
    }

def export_summary_report(summary, plan_df, file_path):
    """
    导出摘要报告
    """
    logger.info(f"Exporting summary report to {file_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("2023-07-01 补货与定价策略优化报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目标日期: {TARGET_DATE}\n\n")
        
        # 基本信息
        f.write("一、基本信息\n")
        f.write("-" * 20 + "\n")
        f.write(f"选中产品数量: {summary['total_products']} 个\n")
        f.write(f"涉及品类数量: {summary['category_count']} 个\n")
        f.write(f"总进货量: {summary['total_stock']:.2f} kg\n")
        f.write(f"预计总销量: {summary['total_predicted_sales']:.2f} kg\n\n")
        
        # 财务指标
        f.write("二、财务指标\n")
        f.write("-" * 20 + "\n")
        f.write(f"预估总收入: {summary['total_revenue']:.2f} 元\n")
        f.write(f"总进货成本: {summary['total_cost']:.2f} 元\n")
        f.write(f"预估总利润: {summary['total_profit']:.2f} 元\n")
        f.write(f"利润率: {summary['profit_margin']:.2%}\n")
        f.write(f"平均每公斤利润: {summary['avg_profit_per_kg']:.2f} 元/kg\n\n")
        
        # 定价策略
        f.write("三、定价策略\n")
        f.write("-" * 20 + "\n")
        f.write(f"平均售价: {summary['avg_selling_price']:.2f} 元/kg\n")
        f.write(f"售价区间: {summary['price_range'][0]:.2f} - {summary['price_range'][1]:.2f} 元/kg\n")
        f.write(f"平均加成率: {summary['avg_markup']:.2%}\n")
        f.write(f"加成率区间: {summary['markup_range'][0]:.2%} - {summary['markup_range'][1]:.2%}\n\n")
        
        # 品类分析
        f.write("四、品类分析\n")
        f.write("-" * 20 + "\n")
        category_profits = {}
        for category in plan_df['分类名称'].unique():
            cat_data = plan_df[plan_df['分类名称'] == category]
            category_profits[category] = {
                'count': len(cat_data),
                'profit': cat_data['预估利润(元)'].sum(),
                'avg_price': cat_data['售价(元/kg)'].mean()
            }
        
        for category, data in sorted(category_profits.items(), 
                                   key=lambda x: x[1]['profit'], reverse=True):
            f.write(f"{category}: {data['count']}个产品, "
                   f"预估利润{data['profit']:.2f}元, "
                   f"平均售价{data['avg_price']:.2f}元/kg\n")
        
        f.write("\n")
        
        # 风险评估
        f.write("五、风险评估\n")
        f.write("-" * 20 + "\n")
        f.write(f"库存周转率: {summary['avg_inventory_turnover']:.2f}\n")
        if 'high_risk_products' in summary:
            f.write(f"高风险产品: {summary['high_risk_products']} 个\n")
        
        # 前10大利润产品
        f.write("\n六、前10大利润产品\n")
        f.write("-" * 20 + "\n")
        top10 = plan_df.head(10)
        for idx, row in top10.iterrows():
            f.write(f"{row['利润排名']:2d}. {row['单品名称']:20s} "
                   f"利润: {row['预估利润(元)']:8.2f}元 "
                   f"售价: {row['售价(元/kg)']:6.2f}元/kg\n")
    
    logger.info("Summary report exported successfully")

def export_category_analysis(plan_df, file_path):
    """
    导出品类分析
    """
    logger.info(f"Exporting category analysis to {file_path}")
    
    # 计算缺失的列
    if '预估收入(元)' not in plan_df.columns:
        plan_df['预估收入(元)'] = plan_df['预测销量(kg)'] * plan_df['售价(元/kg)']
    
    if '进货成本(元)' not in plan_df.columns:
        plan_df['进货成本(元)'] = plan_df['进货量(kg)'] * plan_df['预测批发价(元/kg)']
    
    # 按品类汇总
    category_analysis = plan_df.groupby(['分类编码', '分类名称']).agg({
        '单品编码': 'count',
        '进货量(kg)': 'sum',
        '预测销量(kg)': 'sum',
        '预估收入(元)': 'sum',
        '进货成本(元)': 'sum',
        '预估利润(元)': 'sum',
        '售价(元/kg)': 'mean',
        '加成率': 'mean'
    }).reset_index()
    
    # 重命名列
    category_analysis.rename(columns={
        '单品编码': '产品数量',
        '售价(元/kg)': '平均售价(元/kg)',
        '加成率': '平均加成率'
    }, inplace=True)
    
    # 计算品类利润率
    category_analysis['利润率'] = (
        category_analysis['预估利润(元)'] / category_analysis['预估收入(元)']
    )
    
    # 按利润排序
    category_analysis = category_analysis.sort_values('预估利润(元)', ascending=False)
    
    # 格式化数值
    numeric_cols = ['进货量(kg)', '预测销量(kg)', '预估收入(元)', '进货成本(元)', 
                   '预估利润(元)', '平均售价(元/kg)']
    for col in numeric_cols:
        category_analysis[col] = category_analysis[col].round(2)
    
    category_analysis['平均加成率'] = category_analysis['平均加成率'].round(4)
    category_analysis['利润率'] = category_analysis['利润率'].round(4)
    
    # 导出CSV
    category_analysis.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    logger.info("Category analysis exported successfully")

from config import MIN_SHELF_COUNT, MAX_SHELF_COUNT

def validate_plan(plan_df):
    """
    验证计划的合理性
    """
    logger.info("Validating final plan...")
    
    issues = []
    
    # 检查基本约束（自适应候选数量：以计划行数作为可选上限）
    effective_min = min(MIN_SHELF_COUNT, len(plan_df))
    effective_max = min(MAX_SHELF_COUNT, len(plan_df))
    if len(plan_df) < effective_min:
        issues.append(f"产品数量不足: {len(plan_df)} < {effective_min}")
    
    if len(plan_df) > effective_max:
        issues.append(f"产品数量超限: {len(plan_df)} > {effective_max}")
    
    # 检查进货量
    min_stock = plan_df['进货量(kg)'].min()
    if min_stock < 2.5:
        issues.append(f"存在进货量不足的产品: {min_stock} < 2.5 kg")
    
    # 检查加成率
    min_markup = plan_df['加成率'].min()
    max_markup = plan_df['加成率'].max()
    
    if min_markup < 0.1:
        issues.append(f"存在加成率过低的产品: {min_markup} < 0.1")
    
    if max_markup > 0.6:
        issues.append(f"存在加成率过高的产品: {max_markup} > 0.6")
    
    # 检查数据完整性
    required_cols = ['单品编码', '单品名称', '分类名称', '进货量(kg)', 
                    '加成率', '售价(元/kg)', '预测销量(kg)', '预测批发价(元/kg)', '预估利润(元)']
    
    for col in required_cols:
        if col not in plan_df.columns:
            issues.append(f"缺少必要列: {col}")
        elif plan_df[col].isnull().any():
            issues.append(f"列 {col} 存在空值")
    
    if issues:
        logger.warning("Plan validation failed with issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues
    else:
        logger.info("Plan validation passed successfully")
        return True, []

def integrate_results(solution_df, optimization_summary):
    """
    结果整合主函数
    """
    logger.info("Starting result integration...")
    
    # 计算定价指标
    enhanced_solution = calculate_pricing_metrics(solution_df)
    
    # 生成定价摘要
    pricing_summary = generate_pricing_summary(enhanced_solution)
    
    # 合并摘要信息
    combined_summary = {**optimization_summary, **pricing_summary}
    
    # 验证计划
    is_valid, validation_issues = validate_plan(enhanced_solution)
    
    if not is_valid:
        logger.warning("Plan validation failed, but continuing with export...")
    
    # 导出结果
    export_paths = export_results(enhanced_solution, combined_summary)
    
    logger.info("Result integration completed successfully")
    
    return enhanced_solution, combined_summary, export_paths