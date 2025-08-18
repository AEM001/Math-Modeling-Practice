"""
Candidate product screening module.
"""
import pandas as pd
import numpy as np
import logging
from config import MIN_DISPLAY_QTY, N_CANDIDATES_MAX, QUALITY_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def screen_by_sales_threshold(forecast_df, min_sales=MIN_DISPLAY_QTY):
    """
    按预测销量阈值筛选
    """
    logger.info(f"Screening products by minimum sales threshold: {min_sales} kg")
    
    initial_count = len(forecast_df)
    screened_df = forecast_df[forecast_df['pred_Q_p'] >= min_sales].copy()
    final_count = len(screened_df)
    
    logger.info(f"Sales threshold screening: {initial_count} -> {final_count} products")
    
    return screened_df

def screen_by_model_quality(forecast_df, min_quality=QUALITY_THRESHOLDS['min_r2_score']):
    """
    按模型质量筛选
    """
    logger.info(f"Screening products by model quality threshold: {min_quality}")
    
    initial_count = len(forecast_df)
    screened_df = forecast_df[forecast_df['model_quality'] >= min_quality].copy()
    final_count = len(screened_df)
    
    logger.info(f"Model quality screening: {initial_count} -> {final_count} products")
    
    return screened_df

def calculate_product_scores(forecast_df, historical_df):
    """
    计算产品综合评分
    """
    logger.info("Calculating comprehensive product scores...")
    
    df = forecast_df.copy()
    
    # 1. 销量稳定性评分
    stability_scores = {}
    for product_code in df['单品编码']:
        product_hist = historical_df[historical_df['单品编码'] == product_code]
        
        if len(product_hist) >= 7:
            sales_cv = product_hist['总销量(千克)'].std() / (product_hist['总销量(千克)'].mean() + 1e-8)
            stability_score = np.exp(-sales_cv)  # 变异系数越小，稳定性评分越高
        else:
            stability_score = 0.5
        
        stability_scores[product_code] = stability_score
    
    df['stability_score'] = df['单品编码'].map(stability_scores)
    
    # 2. 利润潜力评分
    df['profit_potential'] = df['pred_Q_p'] * df['pred_C'] * 0.3  # 假设平均加成率30%
    
    # 3. 综合评分
    # 权重：模型质量 40%，稳定性 30%，利润潜力 30%
    df['profit_potential_norm'] = df['profit_potential'] / df['profit_potential'].max()
    
    df['composite_score'] = (
        0.4 * df['model_quality'] +
        0.3 * df['stability_score'] +
        0.3 * df['profit_potential_norm']
    )
    
    logger.info("Product scoring completed")
    
    return df

def ensure_category_diversity(screened_df, min_per_category=1):
    """
    确保品类多样性
    """
    logger.info("Ensuring category diversity...")
    
    # 统计每个品类的产品数量
    category_counts = screened_df['分类编码'].value_counts()
    logger.info(f"Categories represented: {len(category_counts)}")
    
    # 如果某些重要品类缺失，尝试从原始数据中补充
    all_categories = screened_df['分类编码'].unique()
    balanced_products = []
    
    for category in all_categories:
        category_products = screened_df[
            screened_df['分类编码'] == category
        ].sort_values('composite_score', ascending=False)
        
        # 每个品类至少保留min_per_category个产品
        n_to_keep = max(min_per_category, len(category_products) // 3)
        n_to_keep = min(n_to_keep, len(category_products))
        
        balanced_products.append(category_products.head(n_to_keep))
    
    balanced_df = pd.concat(balanced_products, ignore_index=True)
    balanced_df = balanced_df.drop_duplicates(subset=['单品编码'])
    
    logger.info(f"Category balancing: {len(screened_df)} -> {len(balanced_df)} products")
    
    return balanced_df

def select_top_candidates(screened_df, max_candidates=N_CANDIDATES_MAX):
    """
    选择最优候选产品
    """
    logger.info(f"Selecting top {max_candidates} candidates...")
    
    # 按综合评分排序
    sorted_df = screened_df.sort_values('composite_score', ascending=False)
    
    # 选择前N个
    top_candidates = sorted_df.head(max_candidates).copy()
    
    logger.info(f"Selected {len(top_candidates)} top candidates")
    
    # 统计选中产品的品类分布
    category_dist = top_candidates['分类名称'].value_counts()
    logger.info("Category distribution in selected candidates:")
    for category, count in category_dist.items():
        logger.info(f"  {category}: {count} products")
    
    return top_candidates

def screen_candidates(forecast_df, historical_df, config=None):
    """
    候选产品筛选主函数
    """
    logger.info("Starting candidate product screening...")
    
    if config is None:
        config = {
            'min_sales': MIN_DISPLAY_QTY,
            'max_candidates': N_CANDIDATES_MAX,
            'min_quality': QUALITY_THRESHOLDS['min_r2_score'],
            'ensure_diversity': True
        }
    
    # 第一步：销量阈值筛选
    step1_df = screen_by_sales_threshold(forecast_df, config['min_sales'])
    
    if len(step1_df) == 0:
        logger.warning("No products passed sales threshold screening!")
        return pd.DataFrame()
    
    # 第二步：模型质量筛选
    step2_df = screen_by_model_quality(step1_df, config['min_quality'])
    
    if len(step2_df) == 0:
        logger.warning("No products passed model quality screening! Relaxing criteria...")
        # 放宽质量要求
        step2_df = screen_by_model_quality(step1_df, 0.0)
    
    # 第三步：计算综合评分
    step3_df = calculate_product_scores(step2_df, historical_df)
    
    # 第四步：品类多样性平衡（可选）
    if config.get('ensure_diversity', True):
        step4_df = ensure_category_diversity(step3_df)
    else:
        step4_df = step3_df
    
    # 第五步：选择最优候选
    final_candidates = select_top_candidates(step4_df, config['max_candidates'])
    
    # 输出筛选摘要
    logger.info("\n=== Screening Summary ===")
    logger.info(f"Initial products: {len(forecast_df)}")
    logger.info(f"After sales threshold: {len(step1_df)}")
    logger.info(f"After quality screening: {len(step2_df)}")
    logger.info(f"After diversity balancing: {len(step4_df)}")
    logger.info(f"Final candidates: {len(final_candidates)}")
    
    # 输出评分统计
    if len(final_candidates) > 0:
        logger.info(f"\nCandidate Statistics:")
        logger.info(f"Average predicted sales: {final_candidates['pred_Q_p'].mean():.2f} ± {final_candidates['pred_Q_p'].std():.2f} kg")
        logger.info(f"Average predicted price: {final_candidates['pred_C'].mean():.2f} ± {final_candidates['pred_C'].std():.2f} yuan/kg")
        logger.info(f"Average composite score: {final_candidates['composite_score'].mean():.3f} ± {final_candidates['composite_score'].std():.3f}")
        logger.info(f"Model quality range: {final_candidates['model_quality'].min():.3f} - {final_candidates['model_quality'].max():.3f}")
    
    return final_candidates

def analyze_screening_results(candidates_df, forecast_df):
    """
    分析筛选结果
    """
    logger.info("Analyzing screening results...")
    
    if len(candidates_df) == 0:
        logger.warning("No candidates to analyze!")
        return {}
    
    # 基本统计
    analysis = {
        'total_candidates': len(candidates_df),
        'selection_rate': len(candidates_df) / len(forecast_df),
        
        # 销量统计
        'avg_pred_sales': candidates_df['pred_Q_p'].mean(),
        'min_pred_sales': candidates_df['pred_Q_p'].min(),
        'max_pred_sales': candidates_df['pred_Q_p'].max(),
        'total_pred_sales': candidates_df['pred_Q_p'].sum(),
        
        # 价格统计
        'avg_pred_price': candidates_df['pred_C'].mean(),
        'price_range': (candidates_df['pred_C'].min(), candidates_df['pred_C'].max()),
        
        # 品类分布
        'category_distribution': candidates_df['分类名称'].value_counts().to_dict(),
        'unique_categories': candidates_df['分类编码'].nunique(),
        
        # 质量统计
        'avg_model_quality': candidates_df['model_quality'].mean(),
        'avg_composite_score': candidates_df['composite_score'].mean()
    }
    
    # 预估总利润（粗略估计）
    estimated_profit = (candidates_df['pred_Q_p'] * candidates_df['pred_C'] * 0.3).sum()
    analysis['estimated_total_profit'] = estimated_profit
    
    logger.info("Screening analysis completed")
    
    return analysis

def export_screening_report(candidates_df, analysis, output_path=None):
    """
    导出筛选报告
    """
    if output_path is None:
        output_path = "outputs/results/screening_report.txt"
    
    logger.info(f"Exporting screening report to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("候选产品筛选报告\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"筛选日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"候选产品数量: {analysis['total_candidates']}\n")
        f.write(f"筛选率: {analysis['selection_rate']:.2%}\n\n")
        
        f.write("销量统计:\n")
        f.write(f"  平均预测销量: {analysis['avg_pred_sales']:.2f} kg\n")
        f.write(f"  销量范围: {analysis['min_pred_sales']:.2f} - {analysis['max_pred_sales']:.2f} kg\n")
        f.write(f"  预计总销量: {analysis['total_pred_sales']:.2f} kg\n\n")
        
        f.write("价格统计:\n")
        f.write(f"  平均预测价格: {analysis['avg_pred_price']:.2f} 元/kg\n")
        f.write(f"  价格范围: {analysis['price_range'][0]:.2f} - {analysis['price_range'][1]:.2f} 元/kg\n\n")
        
        f.write("品类分布:\n")
        for category, count in analysis['category_distribution'].items():
            f.write(f"  {category}: {count} 个产品\n")
        f.write(f"涉及品类数: {analysis['unique_categories']}\n\n")
        
        f.write("质量指标:\n")
        f.write(f"  平均模型质量: {analysis['avg_model_quality']:.3f}\n")
        f.write(f"  平均综合评分: {analysis['avg_composite_score']:.3f}\n\n")
        
        f.write(f"预估总利润: {analysis['estimated_total_profit']:.2f} 元\n")
    
    logger.info("Screening report exported successfully")