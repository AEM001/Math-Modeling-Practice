import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind
import warnings
import numpy as np
import matplotlib.font_manager as fm # 导入字体管理器
import os # 导入os模块

# --- 全局设置 ---
# 忽略警告,使输出更简洁
warnings.filterwarnings('ignore')
# 设置图表样式和调色板
sns.set_theme(style="whitegrid", palette="muted")


# 更健壮的字体配置：直接设置字体列表，让Matplotlib自动查找。
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'sans-serif']
# print("字体尝试设置为: PingFang SC, Heiti TC, Arial Unicode MS, SimHei, Microsoft YaHei (及备用字体)") # 已禁止输出
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# --- CLR变换 ---

def apply_clr_transformation(df, chemical_cols):
    """
    对化学成分数据应用中心对数比（CLR）变换。
    此版本仅处理非零值，在计算几何平均数时忽略零，
    并在转换后的数据中将它们保留为0。
    """
    # 创建一个副本以避免SettingWithCopyWarning
    df_transformed = df.copy()
    
    # 分离出用于变换的化学成分数据
    composition_data = df_transformed[chemical_cols].copy()

    # 将0替换为NaN,以便在计算中忽略它们
    composition_calc = composition_data.replace(0, np.nan)

    # 计算每行的几何平均值,忽略NaN。
    # gmean = exp(mean(log(x)))。我们对对数转换后的数据使用np.nanmean。
    # 这可以正确地计算仅非零值的几何平均值。
    log_data = np.log(composition_calc)
    mean_log_data = np.nanmean(log_data, axis=1)
    geometric_means = np.exp(mean_log_data)

    # 应用CLR变换: log(x_i / g(x))
    # 这将为非NaN值计算CLR。NaN的结果将是NaN。
    clr_data = np.log(composition_calc.div(geometric_means, axis=0))

    # 将NaN结果(来自原始的0)替换回0。
    clr_data.fillna(0, inplace=True)
    
    # 用新的CLR数据更新数据框的化学成分列
    df_transformed[chemical_cols] = clr_data
    
    return df_transformed


# --- 核心分析函数 ---

def analyze_chemical_composition(df, glass_type, report_data, output_dir):
    """
    为特定类型的玻璃分析风化与化学成分之间的统计关系，
    将每一行视为一个独立的样本。

    Args:
        df (pd.DataFrame): 预先筛选和清理过的数据框。
        glass_type (str): 要分析的玻璃类型（'高钾' 或 '铅钡'）。
        report_data (dict): 用于存储最终报告结果的字典。
        output_dir (str): 用于保存文件的输出目录。
    """
    # 已禁止详细的控制台输出
    # print(f"\n{'='*60}")
    # print(f" 开始分析: {glass_type}玻璃的风化与化学成分统计规律")
    # print(f"{'='*60}\n")
    
    # 初始化此玻璃类型的报告数据字典
    report_data[glass_type] = {}
    
    # 筛选特定玻璃类型的数据
    df_type = df[df['类型'] == glass_type].copy()
    
    # 分为风化和未风化组
    weathered_group = df_type[df_type['表面风化'] == '风化']
    unweathered_group = df_type[df_type['表面风化'] == '无风化']

    # 检查两组中是否有足够的样本进行分析
    if weathered_group.empty or unweathered_group.empty:
        print(f"警告：{glass_type}玻璃类型缺少风化或未风化样本，无法进行比较分析。")
        report_data[glass_type]['error'] = "缺少足够样本（风化或未风化组为空）进行比较分析。"
        return

    # 自动识别化学成分列
    # 排除标识符、分类和计算的总和列
    excluded_cols = ['文物采样点', '类型', '表面风化', 'sum']
    chemical_cols = [col for col in df.columns if col not in excluded_cols and df[col].dtype in ['float64', 'int64']]
    # print(f"识别出的化学成分列: {chemical_cols}\n") # 已禁止输出
    
    # 1. 成分含量差异、变异和显著性检验 (T检验)
    # print("--- 1. 成分含量差异、变异及显著性检验 (T-Test) ---") # 已禁止输出
    results = []
    for col in chemical_cols:
        stat_result = {'component': col}
        
        data_w = weathered_group[col].dropna()
        data_u = unweathered_group[col].dropna()

        # A. 描述性统计
        stat_result['mean_weathered'] = data_w.mean()
        stat_result['std_weathered'] = data_w.std()
        stat_result['mean_unweathered'] = data_u.mean()
        stat_result['std_unweathered'] = data_u.std()
        
        # B. T检验的前提条件和执行
        # 确保有足够的数据点进行统计检验
        if len(data_w) > 2 and len(data_u) > 2:
            # 正态性检验 (Shapiro-Wilk)
            _, p_norm_w = shapiro(data_w)
            _, p_norm_u = shapiro(data_u)
            stat_result['normality_p_weathered'] = p_norm_w
            stat_result['normality_p_unweathered'] = p_norm_u
            
            # 方差齐性检验 (Levene)
            _, p_levene = levene(data_w, data_u)
            stat_result['levene_p'] = p_levene
            # 根据Levene检验结果决定是否假设方差相等
            equal_var = p_levene > 0.05
            
            # 独立样本T检验
            t_stat, p_ttest = ttest_ind(data_w, data_u, equal_var=equal_var, nan_policy='omit')
            stat_result['t_stat'] = t_stat
            stat_result['p_value'] = p_ttest
        else:
            # 如果数据不足,则用NaN填充统计数据
            stat_result['t_stat'] = np.nan
            stat_result['p_value'] = np.nan

        results.append(stat_result)
        
    results_df = pd.DataFrame(results).fillna(0) # 填充NaN以便于表格更清晰
    
    # 添加一列以指示统计显著性
    results_df['是否显著'] = np.where((results_df['p_value'] < 0.05) & (results_df['p_value'] != 0), '是', '否')
    
    report_data[glass_type]['stats_table'] = results_df

    # 将统计结果保存到指定输出目录的CSV文件中
    csv_filename = f"{glass_type}分析结果.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        results_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        # print(f"统计分析结果已保存到 '{csv_filepath}'") # 已禁止输出
    except Exception as e:
        print(f"保存CSV文件 '{csv_filepath}' 失败: {e}")

    # 已禁止详细的控制台输出
    # print("T检验结果摘要 (p<0.05表示差异显著):")
    # print(results_df[['component', 'mean_unweathered', 'mean_weathered', 'p_value']].round(4))
    # print("-" * 60)
    
    # 2. 可视化 - 显著成分的组合箱形图
    # print("\n--- 2. 可视化分析 - 成分含量箱形图 ---") # 已禁止输出
    significant_components = results_df[results_df['p_value'] < 0.05]['component'].tolist()
    if not significant_components:
        # 如果没有显著成分,则使用平均含量最高的前4个作为备选
        significant_components = df_type[chemical_cols].mean().sort_values(ascending=False).head(4).index.tolist()

    # 仅在有可绘制的成分时继续
    if significant_components:
        # 将数据框融合为长格式,以便于绘制多个成分
        df_plot = df_type[significant_components + ['表面风化']].melt(id_vars=['表面风化'], 
                                                                 var_name='化学成分', 
                                                                 value_name='CLR变换值')
        
        plt.figure(figsize=(14, 8)) # 调整图形大小以适应多个组件
        sns.boxplot(data=df_plot, x='化学成分', y='CLR变换值', hue='表面风化', 
                    order=significant_components, palette='muted')
        
        plt.title(f'{glass_type}玻璃: 主要化学成分CLR变换值在风化前后对比', fontsize=16, fontweight='bold')
        plt.xlabel('化学成分', fontsize=12)
        plt.ylabel('CLR 变换值', fontsize=12)
        plt.xticks(rotation=45, ha='right') # 旋转x轴标签以提高可读性
        plt.legend(title='表面风化状态')
        plt.tight_layout()
        
        plot_filename_combined_boxplot = f'boxplot_combined_{glass_type}.png'
        plot_filepath_combined_boxplot = os.path.join(output_dir, plot_filename_combined_boxplot)
        plt.savefig(plot_filepath_combined_boxplot)
        plt.close() # 关闭图形以释放内存
        report_data[glass_type]['combined_boxplot_path'] = plot_filename_combined_boxplot
    else:
        print("没有可供可视化的成分。")
    # print("-" * 60) # 已禁止输出

    # 3. 相关性分析
    # print("\n--- 3. 成分相关性规律分析 - 热力图 ---") # 已禁止输出
    corr_unweathered = unweathered_group[chemical_cols].corr(method='spearman')
    corr_weathered = weathered_group[chemical_cols].corr(method='spearman')
    report_data[glass_type]['corr_unweathered'] = corr_unweathered
    report_data[glass_type]['corr_weathered'] = corr_weathered

    # --- 未风化样本热力图 ---
    plt.figure(figsize=(12, 10))
    ax1 = sns.heatmap(corr_unweathered, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8})
    ax1.set_title(f'{glass_type} - 未风化样本成分相关性 (Spearman)', fontsize=16, pad=20)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plot_filename_unweathered = f'heatmap_{glass_type}_unweathered.png'
    plot_filepath_unweathered = os.path.join(output_dir, plot_filename_unweathered)
    plt.savefig(plot_filepath_unweathered)
    plt.close() # 关闭图形以释放内存
    report_data[glass_type]['heatmap_path_unweathered'] = plot_filename_unweathered

    # --- 风化样本热力图 ---
    plt.figure(figsize=(12, 10))
    ax2 = sns.heatmap(corr_weathered, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8})
    ax2.set_title(f'{glass_type} - 风化样本成分相关性 (Spearman)', fontsize=16, pad=20)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plot_filename_weathered = f'heatmap_{glass_type}_weathered.png'
    plot_filepath_weathered = os.path.join(output_dir, plot_filename_weathered)
    plt.savefig(plot_filepath_weathered)
    plt.close() # 关闭图形以释放内存
    report_data[glass_type]['heatmap_path_weathered'] = plot_filename_weathered
    
    # print(f"相关性热力图已分别保存为 '{plot_filename_unweathered}' 和 '{plot_filename_weathered}'") # 已禁止输出


# --- Markdown报告生成 ---

def generate_markdown_report(report_data):
    """生成Markdown格式的综合分析报告。"""
    
    markdown_content = "# 古代玻璃化学成分与风化关系分析报告 (基于CLR变换与Spearman相关性)\n\n"
    markdown_content += f"**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += "本报告对化学成分数据进行了中心对数比（CLR）变换，以消除成分数据固有的约束。所有后续的统计分析（t检验、相关性分析等）都在变换后的数据上进行。相关性分析采用Spearman等级相关系数，该方法不要求数据呈正态分布，适用性更广。这使得分析结果在统计上更为可靠。\n\n---\n\n"
    
    for glass_type, data in report_data.items():
        markdown_content += f"## {glass_type}玻璃分析\n\n"
        
        if 'error' in data:
            markdown_content += f"**分析中止**: {data['error']}\n\n"
            continue
            
        # 1. 成分含量分析
        markdown_content += "### 1. CLR变换值差异与显著性检验\n\n"
        markdown_content += "为探究风化对化学成分的影响，我们对风化和未风化样本在CLR空间中的值进行了独立样本t检验。下表总结了各成分CLR变换后的平均值及检验p值。p值小于0.05表示风化前后的差异在统计学上是显著的。\n\n"
        markdown_content += "| 化学成分 | 未风化CLR均值 | 风化CLR均值 | p值 | 是否显著 |\n"
        markdown_content += "|:---|:---:|:---:|:---:|:---|\n"
        
        stats_df = data['stats_table']
        for _, row in stats_df.iterrows():
            conclusion = f"**{row['是否显著']}**" if row['是否显著'] == '是' else row['是否显著']
            markdown_content += f"| {row['component']} | {row['mean_unweathered']:.4f} | {row['mean_weathered']:.4f} | {row['p_value']:.4f} | {conclusion} |\n"
        
        markdown_content += "\n**规律总结**：\n"
        significant_changes = stats_df[stats_df['是否显著'] == '是']
        if significant_changes.empty:
            markdown_content += f"- 在{glass_type}玻璃中，风化与未风化样本的所有化学成分的CLR变换值均未表现出统计学上的显著差异。\n"
        else:
            for _, row in significant_changes.iterrows():
                direction = "显著增加" if row['mean_weathered'] > row['mean_unweathered'] else "显著减少"
                markdown_content += f"- 风化过程导致 **{row['component']}** 的相对含量（CLR值）发生**{direction}** (p={row['p_value']:.4f})。\n"
        
        # 2. 可视化
        markdown_content += "\n### 2. 主要成分CLR变换值分布可视化\n\n"
        markdown_content += "以下图通过箱形图展示了CLR值差异显著（或含量最高）的化学成分在风化与未风化样本中的分布情况，直观地体现了其CLR值的中位数、范围和离散程度的变化。\n\n"
        
        # boxplot_paths = data.get('boxplot_paths', []) # 旧代码行，不再使用
        combined_boxplot_path = data.get('combined_boxplot_path', '')
        if combined_boxplot_path:
            # for path in boxplot_paths: # 旧循环，不再使用
            markdown_content += f"**{glass_type}玻璃主要成分CLR变换值分布**\n"
            markdown_content += f"![{glass_type}玻璃 - 主要成分CLR值箱形图](./{combined_boxplot_path})\n\n"
        else:
            markdown_content += "未发现CLR值有显著差异的化学成分，或无数据可供可视化。\n\n"

        # 3. 相关性分析
        markdown_content += "### 3. CLR变换后成分相关性变化分析 (Spearman)\n\n"
        markdown_content += "为探究风化过程中各化学元素间的协同变化关系，我们分别计算了风化前后样本在CLR空间中的成分Spearman等级相关性矩阵，并通过热力图进行可视化。CLR变换消除了成分数据的伪相关性，因此这里的相关性更能反映真实的关联关系。\n\n"
        
        markdown_content += f"**未风化样本CLR变换后成分相关性**\n"
        markdown_content += f"![{glass_type}玻璃未风化成分相关性热力图](./{data.get('heatmap_path_unweathered', '')})\n\n"
        
        markdown_content += f"**风化样本CLR变换后成分相关性**\n"
        markdown_content += f"![{glass_type}玻璃风化成分相关性热力图](./{data.get('heatmap_path_weathered', '')})\n\n"

        markdown_content += "**解读**：对比以上两张热力图，可以观察到风化前后，各化学成分在CLR空间中的Spearman相关性（由相关系数的数值和颜色表示）发生了怎样的变化，例如哪些成分间的正/负相关性增强、减弱或逆转。\n\n---\n\n"

    return markdown_content


# --- 主执行块 ---

def main():
    """
    主函数，运行整个分析流程。
    """
    # --- 定义输出目录 ---
    output_dir = '/Users/Mac/Downloads/22C/22/1/1_2'
    # 确保输出目录存在,如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 数据加载 ---
    # 构建CSV文件的绝对路径
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '附件2_处理后.csv')

    try:
        # 每一行都是一个独立的采样点,被视为一个独立样本
        df = pd.read_csv(file_path, encoding='utf-8')
        # print(f"成功从 '{file_path}' 加载数据。共 {len(df)} 个采样点。") # 已禁止输出
    except Exception as e:
        print(f"数据加载失败: {e}")
        print(f"请确保 '{os.path.basename(file_path)}' 文件存在于脚本 '{script_dir}' 目录下。")
        return

    # --- 数据清洗和准备 ---
    # print("\n--- 开始数据清洗与预处理 ---") # 已禁止输出
    
    # 识别用于总和计算的化学列
    excluded_cols = ['文物采样点', '类型', '表面风化']
    chemical_cols = [col for col in df.columns if col not in excluded_cols]
    
    # 用0填充缺失的化学数据以便计算
    df[chemical_cols] = df[chemical_cols].fillna(0)
    
    # 计算成分总和并筛选"有效数据"(85%到105%)
    df['sum'] = df[chemical_cols].sum(axis=1)
    df_filtered = df[(df['sum'] >= 85) & (df['sum'] <= 105)].copy()
    initial_count = len(df)
    final_count = len(df_filtered)
    # print(f"成分总和筛选：从 {initial_count} 条记录中筛选出 {final_count} 条有效记录 (成分总和在 85% 到 105% 之间)。") # 已禁止输出
    
    # 如有必要,将数值型'表面风化'映射为描述性字符串
    if df_filtered['表面风化'].dtype in ['int64', 'float64']:
        # print("将'表面风化'列从数值映射为文本...") # 已禁止输出
        df_filtered = df_filtered.copy()
        df_filtered.loc[df_filtered['表面风化'] == 1, '表面风化'] = '无风化'
        df_filtered.loc[df_filtered['表面风化'] == 2, '表面风化'] = '风化'
    
    # 删除任何剩余的缺少关键数据的行
    df_cleaned = df_filtered.copy()
    # 删除'表面风化'或'类型'缺失的行
    mask = pd.notna(df_cleaned['表面风化']) & pd.notna(df_cleaned['类型'])
    df_cleaned = df_cleaned[mask]
    
    if len(df_cleaned) < final_count:
        pass # print(f"丢弃 {final_count - len(df_cleaned)} 条缺少'类型'或'表面风化'信息的记录。") # 已禁止输出
    
    # print(f"数据准备完成。最终用于分析的独立采样点共 {len(df_cleaned)} 个。") # 已禁止输出
    # print("---------------------------------") # 已禁止输出
    
    # --- 应用CLR变换 ---
    # print("\n--- 对化学成分数据应用CLR变换 ---") # 已禁止输出
    chemical_cols_for_transform = [col for col in chemical_cols if col != 'sum']
    df_clr = apply_clr_transformation(df_cleaned, chemical_cols_for_transform)
    # print("CLR变换完成。") # 已禁止输出
    
    # --- 执行分析 ---
    report_data = {} # 用于保存最终报告所有数据的字典
    glass_types_to_analyze = pd.unique(df_clr['类型'])
    
    for g_type in glass_types_to_analyze:
        analyze_chemical_composition(df_clr, g_type, report_data, output_dir)
    
    # --- 生成并保存Markdown报告 ---
    # print(f"\n{'='*60}") # 已禁止输出
    # print(" 分析完成，开始生成Markdown格式的最终报告") # 已禁止输出
    # print(f"{'='*60}\n") # 已禁止输出
    markdown_report = generate_markdown_report(report_data)
    
    report_filename = "chemical_analysis_report.md"
    report_filepath = os.path.join(output_dir, report_filename)
    try:
        with open(report_filepath, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        # print(f"Markdown报告已成功生成并保存为 '{report_filename}'") # 已禁止输出
    except Exception as e:
        print(f"保存Markdown报告失败: {e}")

    # 同时在最终输出中显示报告内容
    # print(markdown_report)

if __name__ == '__main__':
    main()
