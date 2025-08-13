import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import seaborn as sns

warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """加载并准备数据"""
    print('正在加载数据...')
    df = pd.read_csv(csv_path)
    df['销售日期'] = pd.to_datetime(df['销售日期'])
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['销售日期'].min()} 至 {df['销售日期'].max()}")
    return df


def aggregate_monthly_sales(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """按月度聚合销量数据"""
    print(f'正在按月度聚合{group_col}销量...')
    
    # 添加年月列
    df['年月'] = df['销售日期'].dt.to_period('M')
    
    # 按年月和分组聚合
    monthly_data = df.groupby(['年月', group_col])[value_col].sum().reset_index()
    monthly_data['年月'] = monthly_data['年月'].astype(str)
    monthly_data['日期'] = pd.to_datetime(monthly_data['年月'] + '-01')
    
    print(f'生成了 {len(monthly_data)} 条月度记录')
    return monthly_data


def create_time_series_plot(data: pd.DataFrame, group_col: str, value_col: str, title: str, output_path: str):
    """创建时序图"""
    plt.figure(figsize=(12, 6))
    
    # 为每个分组绘制时序线
    groups = data[group_col].unique()
    for group in groups:
        group_data = data[data[group_col] == group].sort_values('日期')
        plt.plot(group_data['日期'], group_data[value_col], label=str(group), linewidth=2, alpha=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('时间')
    plt.ylabel('月度销量')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'时序图已保存: {output_path}')


def perform_decomposition(series: pd.Series, period: int = 12, title: str = "时间序列分解") -> dict:
    """执行加法模型分解"""
    try:
        # 确保数据长度足够
        if len(series) < period * 2:
            return None
        
        # 执行分解
        decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        
        # 计算各成分的统计指标
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()
        
        # 分析趋势
        trend_slope = np.polyfit(range(len(trend)), trend, 1)[0] if len(trend) > 1 else 0
        trend_direction = "上升" if trend_slope > 0 else "下降" if trend_slope < 0 else "平稳"
        trend_magnitude = abs(trend_slope)
        
        # 分析季节性
        seasonal_strength = np.std(seasonal) / np.std(series.dropna())
        
        # 分析随机项
        residual_std = np.std(residual)
        total_std = np.std(series.dropna())
        residual_ratio = residual_std / total_std if total_std > 0 else 1
        
        result = {
            'decomposition': decomposition,
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'trend_magnitude': trend_magnitude,
            'seasonal_strength': seasonal_strength,
            'residual_std': residual_std,
            'residual_ratio': residual_ratio,
            'series_length': len(series)
        }
        
        return result
        
    except Exception as e:
        print(f"分解失败: {e}")
        return None


def plot_decomposition(decomposition, title: str, output_path: str):
    """绘制分解结果"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 原始数据
    axes[0].plot(decomposition.observed, label='原始数据', color='blue')
    axes[0].set_title('原始时间序列')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 趋势项
    axes[1].plot(decomposition.trend, label='趋势项', color='red')
    axes[1].set_title('趋势项 (T)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 季节性
    axes[2].plot(decomposition.seasonal, label='季节性', color='green')
    axes[2].set_title('季节性 (S)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 随机项
    axes[3].plot(decomposition.resid, label='随机项', color='orange')
    axes[3].set_title('随机项 (R)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'分解图已保存: {output_path}')


def analyze_group_time_series(data: pd.DataFrame, group_col: str, value_col: str, group_name: str):
    """分析单个分组的时间序列"""
    print(f'\n=== 分析 {group_name} ===')
    
    # 创建时序图
    ts_plot_path = os.path.join(OUTPUT_DIR, f'{group_name}_时序图.png')
    create_time_series_plot(data, group_col, value_col, f'{group_name}月度销量时序图', ts_plot_path)
    
    # 对每个分组进行分解分析
    results = []
    groups = data[group_col].unique()
    
    for group in groups:
        group_data = data[data[group_col] == group].sort_values('日期')
        if len(group_data) < 24:  # 至少需要2年数据
            continue
            
        series = pd.Series(group_data[value_col].values, index=group_data['日期'])
        
        # 执行分解
        decomp_result = perform_decomposition(series, period=12, title=f"{group}时间序列分解")
        
        if decomp_result:
            # 绘制分解图
            decomp_plot_path = os.path.join(OUTPUT_DIR, f'{group_name}_{group}_分解图.png')
            plot_decomposition(decomp_result['decomposition'], f'{group}时间序列分解', decomp_plot_path)
            
            # 记录结果
            result = {
                group_col: group,
                '数据长度': decomp_result['series_length'],
                '趋势方向': decomp_result['trend_direction'],
                '趋势斜率': round(decomp_result['trend_slope'], 4),
                '趋势强度': round(decomp_result['trend_magnitude'], 4),
                '季节性强度': round(decomp_result['seasonal_strength'], 4),
                '随机项标准差': round(decomp_result['residual_std'], 4),
                '随机项占比': round(decomp_result['residual_ratio'], 4)
            }
            results.append(result)
    
    return pd.DataFrame(results)


def generate_summary_report(item_results: pd.DataFrame, category_results: pd.DataFrame):
    """生成分析报告"""
    report_path = os.path.join(OUTPUT_DIR, '时间序列分析报告.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 1.1.3 时间序列分析报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## 分析概述\n\n')
        f.write('对单品和品类的月度销量数据进行了时间序列分析，包括：\n')
        f.write('1. 时序图绘制，观察趋势、周期、季节性\n')
        f.write('2. 加法模型分解：销量 = 趋势(T) + 季节性(S) + 随机项(R)\n')
        f.write('3. 各成分特征分析\n\n')
        
        # 单品分析结果
        if not item_results.empty:
            f.write('## 单品分析结果\n\n')
            f.write('### 趋势分析\n')
            trend_counts = item_results['趋势方向'].value_counts()
            for direction, count in trend_counts.items():
                percentage = round((count / len(item_results) * 100), 2)
                f.write(f'- {direction}: {count}个 ({percentage}%)\n')
            
            f.write('\n### 季节性分析\n')
            strong_seasonal = item_results[item_results['季节性强度'] > 0.3]
            f.write(f'- 强季节性(>0.3): {len(strong_seasonal)}个\n')
            f.write(f'- 平均季节性强度: {round(item_results["季节性强度"].mean(), 4)}\n')
            
            f.write('\n### 随机项分析\n')
            low_residual = item_results[item_results['随机项占比'] < 0.5]
            f.write(f'- 低随机性(<0.5): {len(low_residual)}个\n')
            f.write(f'- 平均随机项占比: {round(item_results["随机项占比"].mean(), 4)}\n')
        
        # 品类分析结果
        if not category_results.empty:
            f.write('\n## 品类分析结果\n\n')
            f.write('### 趋势分析\n')
            for _, row in category_results.iterrows():
                f.write(f'- {row["分类名称"]}: {row["趋势方向"]} (斜率: {row["趋势斜率"]})\n')
            
            f.write('\n### 季节性分析\n')
            for _, row in category_results.iterrows():
                strength = "强" if row["季节性强度"] > 0.3 else "中等" if row["季节性强度"] > 0.1 else "弱"
                f.write(f'- {row["分类名称"]}: {strength} (强度: {round(row["季节性强度"], 4)})\n')
            
            f.write('\n### 随机项分析\n')
            for _, row in category_results.iterrows():
                residual_level = "低" if row["随机项占比"] < 0.5 else "中等" if row["随机项占比"] < 0.8 else "高"
                f.write(f'- {row["分类名称"]}: {residual_level} (占比: {round(row["随机项占比"], 4)})\n')
        
        f.write('\n## 主要结论\n\n')
        f.write('### 显著发现\n')
        
        # 找出最显著的趋势
        if not item_results.empty:
            strongest_trend = item_results.loc[item_results['趋势强度'].idxmax()]
            f.write(f'- 趋势最显著的单品: {strongest_trend["单品名称"]} ({strongest_trend["趋势方向"]}, 强度: {round(strongest_trend["趋势强度"], 4)})\n')
        
        if not category_results.empty:
            strongest_cat_trend = category_results.loc[category_results['趋势强度'].idxmax()]
            f.write(f'- 趋势最显著的品类: {strongest_cat_trend["分类名称"]} ({strongest_cat_trend["趋势方向"]}, 强度: {round(strongest_cat_trend["趋势强度"], 4)})\n')
        
        # 季节性最强的
        if not item_results.empty:
            strongest_seasonal = item_results.loc[item_results['季节性强度'].idxmax()]
            f.write(f'- 季节性最强的单品: {strongest_seasonal["单品名称"]} (强度: {round(strongest_seasonal["季节性强度"], 4)})\n')
        
        if not category_results.empty:
            strongest_cat_seasonal = category_results.loc[category_results['季节性强度'].idxmax()]
            f.write(f'- 季节性最强的品类: {strongest_cat_seasonal["分类名称"]} (强度: {round(strongest_cat_seasonal["季节性强度"], 4)})\n')
    
    print(f'分析报告已生成: {report_path}')


def main():
    """主函数"""
    print("=== 时间序列分析 ===\n")
    
    # 文件路径
    csv_path = '../daily_item_sales_summary.csv'
    
    if not os.path.exists(csv_path):
        print(f"错误：未找到数据文件 {csv_path}")
        return
    
    # 加载数据
    df = load_and_prepare_data(csv_path)
    
    # 单品月度聚合
    item_monthly = aggregate_monthly_sales(df, '单品名称', '单品销量(天)')
    
    # 品类月度聚合
    category_monthly = aggregate_monthly_sales(df, '分类名称', '单品销量(天)')
    
    # 单品时间序列分析
    print("\n=== 单品时间序列分析 ===")
    item_results = analyze_group_time_series(item_monthly, '单品名称', '单品销量(天)', '单品')
    
    # 品类时间序列分析
    print("\n=== 品类时间序列分析 ===")
    category_results = analyze_group_time_series(category_monthly, '分类名称', '单品销量(天)', '品类')
    
    # 保存结果
    if not item_results.empty:
        item_csv = os.path.join(OUTPUT_DIR, '单品_时间序列分析结果.csv')
        item_results.to_csv(item_csv, index=False, encoding='utf-8-sig')
        print(f'单品分析结果已保存: {item_csv}')
    
    if not category_results.empty:
        category_csv = os.path.join(OUTPUT_DIR, '品类_时间序列分析结果.csv')
        category_results.to_csv(category_csv, index=False, encoding='utf-8-sig')
        print(f'品类分析结果已保存: {category_csv}')
    
    # 生成报告
    generate_summary_report(item_results, category_results)
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
