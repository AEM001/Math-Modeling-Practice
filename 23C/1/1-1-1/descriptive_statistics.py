import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data(csv_path):
    print("正在加载数据...")
    df = pd.read_csv(csv_path)
    
    # 转换日期格式
    df['销售日期'] = pd.to_datetime(df['销售日期'])
    
    return df

def calculate_daily_category_sales(df):
    
    print("正在计算每日品类销售量...")
    
    # 按日期和品类分组，对销量求和
    daily_category_sales = df.groupby(['销售日期', '分类名称'])['单品销量(天)'].sum().reset_index()
    daily_category_sales = daily_category_sales.rename(columns={'单品销量(天)': '品类销量(天)'})
    
    print(f"生成了 {len(daily_category_sales)} 条每日品类销售记录")
    
    return daily_category_sales

def calculate_descriptive_statistics(data, group_col, value_col, name_prefix):


    print(f"正在计算{name_prefix}的描述性统计...")
    
    # 计算每个分组的统计指标
    stats = data.groupby(group_col)[value_col].agg([
        ('均值', 'mean'),
        ('方差', 'var'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max'),
        ('中位数', 'median'),
        ('样本数', 'count')
    ]).reset_index()
    
    # 计算变异系数 CV = σ/μ × 100%
    stats['变异系数(%)'] = (stats['标准差'] / stats['均值']) * 100
    
    # 根据变异系数分类波动程度
    def classify_volatility(cv):
        if cv < 30:
            return '低波动'
        elif cv <= 50:
            return '中等波动'
        else:
            return '高波动'
    
    stats['波动程度'] = stats['变异系数(%)'].apply(classify_volatility)
    
    # 重新排列列顺序
    stats = stats[[group_col, '均值', '方差', '标准差', '变异系数(%)', '波动程度', '最小值', '最大值', '中位数', '样本数']]
    
    print(f"完成{name_prefix}统计，共 {len(stats)} 个分组")
    
    return stats

def analyze_volatility_distribution(stats, name_prefix):
    """
    分析波动程度分布
    """
    print(f"\n=== {name_prefix}波动程度分布 ===")
    volatility_counts = stats['波动程度'].value_counts()
    print(volatility_counts)
    
    # 计算各波动程度的比例
    volatility_percentages = (volatility_counts / len(stats) * 100).round(2)
    print("\n波动程度比例:")
    for level, percentage in volatility_percentages.items():
        print(f"  {level}: {percentage}%")

def create_volatility_visualization(stats, group_col, name_prefix):
    """
    创建波动程度可视化图表
    """
    print(f"正在创建{name_prefix}波动程度可视化...")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{name_prefix}销售波动性分析', fontsize=16, fontweight='bold')
    
    # 1. 变异系数分布直方图
    axes[0, 0].hist(stats['变异系数(%)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=30, color='orange', linestyle='--', label='低波动阈值(30%)')
    axes[0, 0].axvline(x=50, color='red', linestyle='--', label='高波动阈值(50%)')
    axes[0, 0].set_xlabel('变异系数(%)')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('变异系数分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 波动程度饼图
    volatility_counts = stats['波动程度'].value_counts()
    colors = ['lightgreen', 'orange', 'red']
    axes[0, 1].pie(volatility_counts.values, labels=volatility_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 1].set_title('波动程度分布')
    
    # 3. 变异系数与均值散点图
    axes[1, 0].scatter(stats['均值'], stats['变异系数(%)'], alpha=0.6, s=50)
    axes[1, 0].axhline(y=30, color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('平均销量')
    axes[1, 0].set_ylabel('变异系数(%)')
    axes[1, 0].set_title('平均销量 vs 变异系数')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 各波动程度的箱线图
    volatility_data = []
    volatility_labels = []
    for level in ['低波动', '中等波动', '高波动']:
        if level in stats['波动程度'].values:
            volatility_data.append(stats[stats['波动程度'] == level]['变异系数(%)'])
            volatility_labels.append(level)
    
    if volatility_data:
        axes[1, 1].boxplot(volatility_data, labels=volatility_labels)
        axes[1, 1].set_ylabel('变异系数(%)')
        axes[1, 1].set_title('各波动程度的变异系数分布')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{name_prefix}_波动性分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化图表已保存为: {name_prefix}_波动性分析.png")

def save_results(item_stats, category_stats, output_dir='.'):
    """
    保存分析结果
    """
    print("\n正在保存分析结果...")
    
    # 保存单品统计结果
    item_output_file = f'{output_dir}/单品销售统计.csv'
    item_stats.to_csv(item_output_file, index=False, encoding='utf-8-sig')
    print(f"单品统计结果已保存至: {item_output_file}")
    
    # 保存品类统计结果
    category_output_file = f'{output_dir}/品类销售统计.csv'
    category_stats.to_csv(category_output_file, index=False, encoding='utf-8-sig')
    print(f"品类统计结果已保存至: {category_output_file}")
    
    # 创建汇总报告
    report_file = f'{output_dir}/描述性统计分析报告.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('# 销售数据描述性统计分析报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## 1. 单品销售统计\n\n')
        f.write(f'- 总单品种类数: {len(item_stats)}\n')
        f.write('- 波动程度分布:\n')
        for level, count in item_stats['波动程度'].value_counts().items():
            percentage = round((count / len(item_stats) * 100), 2)
            f.write(f'  - {level}: {count}个 ({percentage}%)\n')
        
        f.write('\n## 2. 品类销售统计\n\n')
        f.write(f'- 总品类数: {len(category_stats)}\n')
        f.write('- 波动程度分布:\n')
        for level, count in category_stats['波动程度'].value_counts().items():
            percentage = round((count / len(category_stats) * 100), 2)
            f.write(f'  - {level}: {count}个 ({percentage}%)\n')
        
        f.write('\n## 3. 统计指标说明\n\n')
        f.write('- **均值**: 平均每日销售量\n')
        f.write('- **方差**: 销售量的离散程度\n')
        f.write('- **标准差**: 销售量的波动幅度\n')
        f.write('- **变异系数**: CV = σ/μ × 100%，衡量相对波动程度\n')
        f.write('- **波动程度分类**:\n')
        f.write('  - 低波动: CV < 30% (销售稳定)\n')
        f.write('  - 中等波动: 30% ≤ CV ≤ 50%\n')
        f.write('  - 高波动: CV > 50% (销售不稳定)\n')
    
    print(f"分析报告已保存至: {report_file}")

def main():
    """
    主函数
    """
    print("=== 销售数据描述性统计分析 ===\n")
    
    # 文件路径
    csv_path = 'daily_item_sales_summary.csv'
    
    try:
        # 1. 加载数据
        df = load_and_prepare_data(csv_path)
        
        # 2. 计算每日品类销售量
        daily_category_sales = calculate_daily_category_sales(df)
        
        # 3. 计算单品描述性统计
        item_stats = calculate_descriptive_statistics(
            df, '单品名称', '单品销量(天)', '单品'
        )
        
        # 4. 计算品类描述性统计
        category_stats = calculate_descriptive_statistics(
            daily_category_sales, '分类名称', '品类销量(天)', '品类'
        )
        
        # 5. 分析波动程度分布
        print("\n" + "="*50)
        analyze_volatility_distribution(item_stats, "单品")
        analyze_volatility_distribution(category_stats, "品类")
        
        # 6. 创建可视化
        create_volatility_visualization(item_stats, '单品名称', '单品')
        create_volatility_visualization(category_stats, '分类名称', '品类')
        
        # 7. 保存结果
        save_results(item_stats, category_stats)
        
        # 8. 显示关键统计信息
        print("\n" + "="*50)
        print("=== 关键统计信息 ===")
        print(f"单品总数: {len(item_stats)}")
        print(f"品类总数: {len(category_stats)}")
        
        print("\n单品波动程度分布:")
        item_volatility = item_stats['波动程度'].value_counts()
        for level, count in item_volatility.items():
            percentage = round((count / len(item_stats) * 100), 2)
            print(f"  {level}: {count}个 ({percentage}%)")
        
        print("\n品类波动程度分布:")
        category_volatility = category_stats['波动程度'].value_counts()
        for level, count in category_volatility.items():
            percentage = round((count / len(category_stats) * 100), 2)
            print(f"  {level}: {count}个 ({percentage}%)")
        
        print("\n分析完成！")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        print("请确保 daily_item_sales_summary.csv 文件存在于当前目录")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
