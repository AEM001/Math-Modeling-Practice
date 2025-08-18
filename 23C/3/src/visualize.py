"""
Visualization module for analysis results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config import OUTPUT_PATHS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_plot_style():
    """
    设置绘图样式
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 设置图形参数
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_sales_distribution(forecast_df, candidates_df, solution_df, save_path=None):
    """
    绘制销量分布对比图
    """
    logger.info("Plotting sales distribution comparison...")
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('销量分布分析', fontsize=16, fontweight='bold')
    
    # 1. 预测销量分布
    axes[0, 0].hist(forecast_df['pred_Q_p'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(forecast_df['pred_Q_p'].mean(), color='red', linestyle='--', 
                      label=f'平均值: {forecast_df["pred_Q_p"].mean():.2f}')
    axes[0, 0].set_title('所有产品预测销量分布')
    axes[0, 0].set_xlabel('预测销量 (kg)')
    axes[0, 0].set_ylabel('产品数量')
    axes[0, 0].legend()
    
    # 2. 候选vs最终选择对比
    axes[0, 1].hist(candidates_df['pred_Q_p'], bins=20, alpha=0.7, color='lightgreen', 
                   label='候选产品', edgecolor='black')
    if len(solution_df) > 0:
        axes[0, 1].hist(solution_df['预测销量(kg)'], bins=15, alpha=0.7, color='orange',
                       label='最终选择', edgecolor='black')
    axes[0, 1].set_title('候选产品vs最终选择销量分布')
    axes[0, 1].set_xlabel('预测销量 (kg)')
    axes[0, 1].set_ylabel('产品数量')
    axes[0, 1].legend()
    
    # 3. 按品类的销量分布
    if len(solution_df) > 0:
        category_sales = solution_df.groupby('分类名称')['预测销量(kg)'].sum().sort_values(ascending=True)
        axes[1, 0].barh(range(len(category_sales)), category_sales.values, color='lightcoral')
        axes[1, 0].set_yticks(range(len(category_sales)))
        axes[1, 0].set_yticklabels(category_sales.index, fontsize=8)
        axes[1, 0].set_title('各品类预测销量汇总')
        axes[1, 0].set_xlabel('预测销量 (kg)')
    
    # 4. 销量vs价格散点图
    if len(solution_df) > 0:
        scatter = axes[1, 1].scatter(solution_df['预测销量(kg)'], solution_df['售价(元/kg)'], 
                                   c=solution_df['预估利润(元)'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_title('销量vs价格关系 (颜色表示利润)')
        axes[1, 1].set_xlabel('预测销量 (kg)')
        axes[1, 1].set_ylabel('售价 (元/kg)')
        plt.colorbar(scatter, ax=axes[1, 1], label='预估利润 (元)')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{OUTPUT_PATHS['figs_dir']}/sales_distribution.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sales distribution plot saved to {save_path}")
    
    return fig

def plot_price_analysis(candidates_df, solution_df, save_path=None):
    """
    绘制价格分析图
    """
    logger.info("Plotting price analysis...")
    
    if len(solution_df) == 0:
        logger.warning("No solution data for price analysis")
        return None
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('价格策略分析', fontsize=16, fontweight='bold')
    
    # 1. 加成率分布
    axes[0, 0].hist(solution_df['加成率'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].axvline(solution_df['加成率'].mean(), color='red', linestyle='--',
                      label=f'平均: {solution_df["加成率"].mean():.2%}')
    axes[0, 0].set_title('加成率分布')
    axes[0, 0].set_xlabel('加成率')
    axes[0, 0].set_ylabel('产品数量')
    axes[0, 0].legend()
    
    # 2. 售价分布
    axes[0, 1].hist(solution_df['售价(元/kg)'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(solution_df['售价(元/kg)'].mean(), color='red', linestyle='--',
                      label=f'平均: {solution_df["售价(元/kg)"].mean():.2f}元/kg')
    axes[0, 1].set_title('售价分布')
    axes[0, 1].set_xlabel('售价 (元/kg)')
    axes[0, 1].set_ylabel('产品数量')
    axes[0, 1].legend()
    
    # 3. 批发价vs售价关系
    axes[1, 0].scatter(solution_df['预测批发价(元/kg)'], solution_df['售价(元/kg)'], alpha=0.6)
    # 绘制参考线
    min_price = min(solution_df['预测批发价(元/kg)'].min(), solution_df['售价(元/kg)'].min())
    max_price = max(solution_df['预测批发价(元/kg)'].max(), solution_df['售价(元/kg)'].max())
    axes[1, 0].plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.5, label='等价线')
    axes[1, 0].plot([min_price, max_price], [min_price*1.3, max_price*1.3], 'g--', alpha=0.5, label='30%加成线')
    axes[1, 0].set_title('批发价vs售价关系')
    axes[1, 0].set_xlabel('批发价 (元/kg)')
    axes[1, 0].set_ylabel('售价 (元/kg)')
    axes[1, 0].legend()
    
    # 4. 品类平均价格
    category_prices = solution_df.groupby('分类名称')['售价(元/kg)'].mean().sort_values(ascending=True)
    if len(category_prices) > 0:
        axes[1, 1].barh(range(len(category_prices)), category_prices.values, color='gold')
        axes[1, 1].set_yticks(range(len(category_prices)))
        axes[1, 1].set_yticklabels(category_prices.index, fontsize=8)
        axes[1, 1].set_title('各品类平均售价')
        axes[1, 1].set_xlabel('平均售价 (元/kg)')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{OUTPUT_PATHS['figs_dir']}/price_analysis.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Price analysis plot saved to {save_path}")
    
    return fig

def plot_profit_analysis(solution_df, save_path=None):
    """
    绘制利润分析图
    """
    logger.info("Plotting profit analysis...")
    
    if len(solution_df) == 0:
        logger.warning("No solution data for profit analysis")
        return None
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('利润分析', fontsize=16, fontweight='bold')
    
    # 1. 利润分布
    axes[0, 0].hist(solution_df['预估利润(元)'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 0].axvline(solution_df['预估利润(元)'].mean(), color='red', linestyle='--',
                      label=f'平均: {solution_df["预估利润(元)"].mean():.2f}元')
    axes[0, 0].set_title('产品利润分布')
    axes[0, 0].set_xlabel('预估利润 (元)')
    axes[0, 0].set_ylabel('产品数量')
    axes[0, 0].legend()
    
    # 2. 帕累托图 - 利润贡献排序
    sorted_profits = solution_df.sort_values('预估利润(元)', ascending=False)
    cumulative_profits = sorted_profits['预估利润(元)'].cumsum()
    total_profit = solution_df['预估利润(元)'].sum()
    cumulative_pct = cumulative_profits / total_profit * 100
    
    ax2_twin = axes[0, 1].twinx()
    bars = axes[0, 1].bar(range(len(sorted_profits)), sorted_profits['预估利润(元)'], 
                         color='steelblue', alpha=0.7)
    line = ax2_twin.plot(range(len(sorted_profits)), cumulative_pct, 'ro-', linewidth=2)
    
    axes[0, 1].set_title('产品利润贡献帕累托分析')
    axes[0, 1].set_xlabel('产品排名')
    axes[0, 1].set_ylabel('预估利润 (元)', color='steelblue')
    ax2_twin.set_ylabel('累积利润占比 (%)', color='red')
    ax2_twin.axhline(80, color='red', linestyle='--', alpha=0.5, label='80%线')
    
    # 3. 品类利润贡献
    category_profits = solution_df.groupby('分类名称')['预估利润(元)'].sum().sort_values(ascending=False)
    if len(category_profits) > 0:
        wedges, texts, autotexts = axes[1, 0].pie(category_profits.values, labels=category_profits.index, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('各品类利润贡献占比')
        
        # 调整文字大小
        for text in texts:
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_fontsize(8)
    
    # 4. 利润率vs销量散点图
    if '毛利率' in solution_df.columns:
        scatter = axes[1, 1].scatter(solution_df['预测销量(kg)'], solution_df['毛利率'], 
                                   c=solution_df['预估利润(元)'], cmap='plasma', alpha=0.6)
        axes[1, 1].set_title('销量vs利润率关系')
        axes[1, 1].set_xlabel('预测销量 (kg)')
        axes[1, 1].set_ylabel('毛利率')
        plt.colorbar(scatter, ax=axes[1, 1], label='预估利润 (元)')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{OUTPUT_PATHS['figs_dir']}/profit_analysis.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Profit analysis plot saved to {save_path}")
    
    return fig

def plot_category_overview(solution_df, save_path=None):
    """
    绘制品类概览图
    """
    logger.info("Plotting category overview...")
    
    if len(solution_df) == 0:
        logger.warning("No solution data for category overview")
        return None
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('品类分析概览', fontsize=16, fontweight='bold')
    
    # 品类汇总统计
    category_stats = solution_df.groupby('分类名称').agg({
        '单品编码': 'count',
        '预测销量(kg)': 'sum',
        '预估利润(元)': 'sum',
        '售价(元/kg)': 'mean'
    }).rename(columns={'单品编码': '产品数量'})
    
    # 1. 各品类产品数量
    category_counts = category_stats['产品数量'].sort_values(ascending=True)
    axes[0, 0].barh(range(len(category_counts)), category_counts.values, color='lightblue')
    axes[0, 0].set_yticks(range(len(category_counts)))
    axes[0, 0].set_yticklabels(category_counts.index, fontsize=8)
    axes[0, 0].set_title('各品类产品数量')
    axes[0, 0].set_xlabel('产品数量')
    
    # 2. 各品类销量汇总
    category_sales = category_stats['预测销量(kg)'].sort_values(ascending=True)
    axes[0, 1].barh(range(len(category_sales)), category_sales.values, color='lightgreen')
    axes[0, 1].set_yticks(range(len(category_sales)))
    axes[0, 1].set_yticklabels(category_sales.index, fontsize=8)
    axes[0, 1].set_title('各品类销量汇总')
    axes[0, 1].set_xlabel('预测销量 (kg)')
    
    # 3. 各品类平均价格
    category_prices = category_stats['售价(元/kg)'].sort_values(ascending=True)
    axes[1, 0].barh(range(len(category_prices)), category_prices.values, color='gold')
    axes[1, 0].set_yticks(range(len(category_prices)))
    axes[1, 0].set_yticklabels(category_prices.index, fontsize=8)
    axes[1, 0].set_title('各品类平均售价')
    axes[1, 0].set_xlabel('平均售价 (元/kg)')
    
    # 4. 品类利润气泡图
    x = category_stats['预测销量(kg)']
    y = category_stats['预估利润(元)']
    sizes = category_stats['产品数量'] * 20  # 调整气泡大小
    
    scatter = axes[1, 1].scatter(x, y, s=sizes, alpha=0.6, c=category_stats['售价(元/kg)'], 
                               cmap='viridis')
    axes[1, 1].set_title('品类综合分析 (气泡大小=产品数量)')
    axes[1, 1].set_xlabel('预测销量 (kg)')
    axes[1, 1].set_ylabel('预估利润 (元)')
    
    # 添加品类标签
    for i, category in enumerate(category_stats.index):
        axes[1, 1].annotate(category, (x.iloc[i], y.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=axes[1, 1], label='平均售价 (元/kg)')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{OUTPUT_PATHS['figs_dir']}/category_overview.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Category overview plot saved to {save_path}")
    
    return fig

def plot_optimization_summary(summary, save_path=None):
    """
    绘制优化结果摘要图
    """
    logger.info("Plotting optimization summary...")
    
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('优化结果摘要', fontsize=16, fontweight='bold')
    
    # 1. 关键指标仪表盘
    metrics = {
        '总利润': summary.get('total_profit', 0),
        '总收入': summary.get('total_revenue', 0),
        '总成本': summary.get('total_cost', 0),
        '利润率': summary.get('profit_margin', 0) * 100
    }
    
    colors = ['gold', 'lightblue', 'lightcoral', 'lightgreen']
    bars = axes[0, 0].bar(metrics.keys(), metrics.values(), color=colors)
    axes[0, 0].set_title('关键财务指标')
    axes[0, 0].set_ylabel('金额 (元) / 百分比 (%)')
    
    # 添加数值标签
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 2. 产品选择统计
    selection_stats = {
        '选中产品': summary.get('selected_count', 0),
        '候选产品': summary.get('total_candidates', summary.get('selected_count', 0) * 2),
        '品类数量': summary.get('category_count', 0)
    }
    
    wedges, texts, autotexts = axes[0, 1].pie([selection_stats['选中产品'], 
                                              selection_stats['候选产品'] - selection_stats['选中产品']], 
                                             labels=['选中', '未选中'], autopct='%1.1f%%',
                                             startangle=90, colors=['lightgreen', 'lightgray'])
    axes[0, 1].set_title(f'产品选择情况 (总候选: {selection_stats["候选产品"]})')
    
    # 3. 库存vs销量对比
    stock_sales_data = {
        '总进货量': summary.get('total_stock_kg', 0),
        '预计销量': summary.get('total_predicted_sales', 0)
    }
    
    x_pos = np.arange(len(stock_sales_data))
    bars = axes[1, 0].bar(x_pos, stock_sales_data.values(), 
                         color=['orange', 'skyblue'], alpha=0.7)
    axes[1, 0].set_title('进货量vs预计销量')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(stock_sales_data.keys())
    axes[1, 0].set_ylabel('重量 (kg)')
    
    # 添加数值标签
    for bar, value in zip(bars, stock_sales_data.values()):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 4. 品类分布（如果有数据）
    if 'category_distribution' in summary and summary['category_distribution']:
        category_dist = summary['category_distribution']
        if isinstance(category_dist, dict):
            # 只显示前6个品类
            top_categories = dict(sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:6])
            
            wedges, texts, autotexts = axes[1, 1].pie(top_categories.values(), 
                                                     labels=list(top_categories.keys()),
                                                     autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('主要品类分布')
            
            # 调整文字大小
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
    else:
        axes[1, 1].text(0.5, 0.5, '品类分布数据不可用', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('品类分布')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{OUTPUT_PATHS['figs_dir']}/optimization_summary.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Optimization summary plot saved to {save_path}")
    
    return fig

def create_all_visualizations(forecast_df, candidates_df, solution_df, summary):
    """
    创建所有可视化图表
    """
    logger.info("Creating all visualizations...")
    
    # 确保输出目录存在
    import os
    os.makedirs(OUTPUT_PATHS['figs_dir'], exist_ok=True)
    
    figures = {}
    
    try:
        # 1. 销量分布分析
        figures['sales_distribution'] = plot_sales_distribution(
            forecast_df, candidates_df, solution_df
        )
        plt.close()  # 关闭图形以节省内存
        
        # 2. 价格分析
        if len(solution_df) > 0:
            figures['price_analysis'] = plot_price_analysis(candidates_df, solution_df)
            plt.close()
            
            # 3. 利润分析
            figures['profit_analysis'] = plot_profit_analysis(solution_df)
            plt.close()
            
            # 4. 品类概览
            figures['category_overview'] = plot_category_overview(solution_df)
            plt.close()
        
        # 5. 优化摘要
        figures['optimization_summary'] = plot_optimization_summary(summary)
        plt.close()
        
        logger.info(f"Successfully created {len(figures)} visualization plots")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return figures

def generate_visualization_report(figures, save_path=None):
    """
    生成可视化报告
    """
    if save_path is None:
        save_path = f"{OUTPUT_PATHS['figs_dir']}/visualization_report.txt"
    
    logger.info(f"Generating visualization report to {save_path}")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("可视化分析报告\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("生成的图表:\n")
        f.write("-" * 20 + "\n")
        
        chart_descriptions = {
            'sales_distribution': '销量分布分析 - 展示预测销量分布、候选vs最终选择对比、品类销量汇总',
            'price_analysis': '价格策略分析 - 展示加成率分布、售价分布、批发价vs售价关系',
            'profit_analysis': '利润分析 - 展示利润分布、帕累托分析、品类利润贡献',
            'category_overview': '品类分析概览 - 展示各品类的产品数量、销量、价格等统计',
            'optimization_summary': '优化结果摘要 - 展示关键财务指标和选择统计'
        }
        
        for figure_name, description in chart_descriptions.items():
            if figure_name in figures:
                f.write(f"✓ {figure_name}.png - {description}\n")
            else:
                f.write(f"✗ {figure_name}.png - 生成失败\n")
        
        f.write(f"\n成功生成 {len(figures)} 个图表\n")
        f.write(f"图表保存路径: {OUTPUT_PATHS['figs_dir']}\n")
    
    logger.info("Visualization report generated successfully")