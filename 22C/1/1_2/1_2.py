import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind
import warnings
import numpy as np
import matplotlib.font_manager as fm # Import font_manager
import os # Import the os module

# --- Global Settings ---
# Ignore warnings for better output readability
warnings.filterwarnings('ignore')
# Set plot style and color palette
sns.set_theme(style="whitegrid", palette="muted")

# 'PingFang SC' is a good choice. 'Heiti TC' is another option.
# 更健壮的字体配置：直接设置字体列表，让Matplotlib自动查找。
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'sans-serif']
# print("字体尝试设置为: PingFang SC, Heiti TC, Arial Unicode MS, SimHei, Microsoft YaHei (及备用字体)") # Suppressed
plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs


# --- CLR Transformation ---

def apply_clr_transformation(df, chemical_cols):
    """
    Applies Centered Log-Ratio (CLR) transformation to chemical composition data.
    This version processes only non-zero values, ignoring zeros in the calculation
    of the geometric mean and leaving them as 0 in the transformed data.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_transformed = df.copy()
    
    # Isolate the chemical data for transformation
    composition_data = df_transformed[chemical_cols].copy()

    # Replace 0s with NaN to ignore them in calculations
    composition_calc = composition_data.replace(0, np.nan)

    # Calculate geometric mean for each row, ignoring NaNs.
    # gmean = exp(mean(log(x))). We use np.nanmean on the log-transformed data.
    # This correctly computes the geometric mean of only the non-zero values.
    log_data = np.log(composition_calc)
    mean_log_data = np.nanmean(log_data, axis=1)
    geometric_means = np.exp(mean_log_data)

    # Apply CLR transformation: log(x_i / g(x))
    # This will calculate the CLR for non-NaN values. The result for NaNs will be NaN.
    clr_data = np.log(composition_calc.div(geometric_means, axis=0))

    # Replace NaN results (which came from original zeros) back to 0.
    clr_data.fillna(0, inplace=True)
    
    # Update the dataframe's chemical columns with the new CLR data
    df_transformed[chemical_cols] = clr_data
    
    return df_transformed


# --- Core Analysis Function ---

def analyze_chemical_composition(df, glass_type, report_data, output_dir):
    """
    Analyzes the statistical relationship between weathering and chemical composition
    for a specific type of glass, treating each row as an independent sample.

    Args:
        df (pd.DataFrame): The pre-filtered and cleaned dataframe.
        glass_type (str): The type of glass to analyze ('高钾' or '铅钡').
        report_data (dict): A dictionary to store results for the final report.
        output_dir (str): The output directory for saving files.
    """
    # Suppressed detailed console output
    # print(f"\n{'='*60}")
    # print(f" 开始分析: {glass_type}玻璃的风化与化学成分统计规律")
    # print(f"{'='*60}\n")
    
    # Initialize dictionary for this glass type's report data
    report_data[glass_type] = {}
    
    # Filter data for the specific glass type
    df_type = df[df['类型'] == glass_type].copy()
    
    # Separate into weathered and unweathered groups
    weathered_group = df_type[df_type['表面风化'] == '风化']
    unweathered_group = df_type[df_type['表面风化'] == '无风化']

    # Check if there are enough samples in both groups to perform analysis
    if weathered_group.empty or unweathered_group.empty:
        print(f"警告：{glass_type}玻璃类型缺少风化或未风化样本，无法进行比较分析。")
        report_data[glass_type]['error'] = "缺少足够样本（风化或未风化组为空）进行比较分析。"
        return

    # Identify chemical composition columns automatically
    # Excludes identifier, categorical, and calculated sum columns
    excluded_cols = ['文物采样点', '类型', '表面风化', 'sum']
    chemical_cols = [col for col in df.columns if col not in excluded_cols and df[col].dtype in ['float64', 'int64']]
    # print(f"识别出的化学成分列: {chemical_cols}\n") # Suppressed
    
    # 1. Component Content Difference, Variation, and T-Test
    # print("--- 1. 成分含量差异、变异及显著性检验 (T-Test) ---") # Suppressed
    results = []
    for col in chemical_cols:
        stat_result = {'component': col}
        
        data_w = weathered_group[col].dropna()
        data_u = unweathered_group[col].dropna()

        # A. Descriptive statistics
        stat_result['mean_weathered'] = data_w.mean()
        stat_result['std_weathered'] = data_w.std()
        stat_result['mean_unweathered'] = data_u.mean()
        stat_result['std_unweathered'] = data_u.std()
        
        # B. T-Test prerequisites and execution
        # Ensure there are enough data points for statistical tests
        if len(data_w) > 2 and len(data_u) > 2:
            # Normality Test (Shapiro-Wilk)
            _, p_norm_w = shapiro(data_w)
            _, p_norm_u = shapiro(data_u)
            stat_result['normality_p_weathered'] = p_norm_w
            stat_result['normality_p_unweathered'] = p_norm_u
            
            # Variance Homogeneity Test (Levene)
            _, p_levene = levene(data_w, data_u)
            stat_result['levene_p'] = p_levene
            # Decide whether to assume equal variances based on Levene test result
            equal_var = p_levene > 0.05
            
            # Independent T-Test
            t_stat, p_ttest = ttest_ind(data_w, data_u, equal_var=equal_var, nan_policy='omit')
            stat_result['t_stat'] = t_stat
            stat_result['p_value'] = p_ttest
        else:
            # If not enough data, fill stats with NaN
            stat_result['t_stat'] = np.nan
            stat_result['p_value'] = np.nan

        results.append(stat_result)
        
    results_df = pd.DataFrame(results).fillna(0) # Fill NaNs for cleaner tables
    
    # Add a column to indicate statistical significance
    results_df['是否显著'] = np.where((results_df['p_value'] < 0.05) & (results_df['p_value'] != 0), '是', '否')
    
    report_data[glass_type]['stats_table'] = results_df

    # Save statistical results to a CSV file in the specified output directory
    csv_filename = f"{glass_type}分析结果.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        results_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        # print(f"统计分析结果已保存到 '{csv_filepath}'") # Suppressed
    except Exception as e:
        print(f"保存CSV文件 '{csv_filepath}' 失败: {e}")

    # Suppressed detailed console output
    # print("T检验结果摘要 (p<0.05表示差异显著):")
    # print(results_df[['component', 'mean_unweathered', 'mean_weathered', 'p_value']].round(4))
    # print("-" * 60)
    
    # 2. Visualization - Combined Boxplot for significant components
    # print("\n--- 2. 可视化分析 - 成分含量箱形图 ---") # Suppressed
    significant_components = results_df[results_df['p_value'] < 0.05]['component'].tolist()
    if not significant_components:
        # If no significant components, use the top 4 by average content as a fallback
        significant_components = df_type[chemical_cols].mean().sort_values(ascending=False).head(4).index.tolist()

    # Only proceed if there are components to plot
    if significant_components:
        # Melt the dataframe to long format for easier plotting of multiple components
        df_plot = df_type[significant_components + ['表面风化']].melt(id_vars=['表面风化'], 
                                                                 var_name='化学成分', 
                                                                 value_name='CLR变换值')
        
        plt.figure(figsize=(14, 8)) # Adjust figure size for multiple components
        sns.boxplot(data=df_plot, x='化学成分', y='CLR变换值', hue='表面风化', 
                    order=significant_components, palette='muted')
        
        plt.title(f'{glass_type}玻璃: 主要化学成分CLR变换值在风化前后对比', fontsize=16, fontweight='bold')
        plt.xlabel('化学成分', fontsize=12)
        plt.ylabel('CLR 变换值', fontsize=12)
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
        plt.legend(title='表面风化状态')
            plt.tight_layout()
        
        plot_filename_combined_boxplot = f'boxplot_combined_{glass_type}.png'
        plot_filepath_combined_boxplot = os.path.join(output_dir, plot_filename_combined_boxplot)
        plt.savefig(plot_filepath_combined_boxplot)
        plt.close() # Close the figure to free up memory
        report_data[glass_type]['combined_boxplot_path'] = plot_filename_combined_boxplot
    else:
        print("没有可供可视化的成分。")
    # print("-" * 60) # Suppressed

    # 3. Correlation Analysis
    # print("\n--- 3. 成分相关性规律分析 - 热力图 ---") # Suppressed
    corr_unweathered = unweathered_group[chemical_cols].corr()
    corr_weathered = weathered_group[chemical_cols].corr()
    report_data[glass_type]['corr_unweathered'] = corr_unweathered
    report_data[glass_type]['corr_weathered'] = corr_weathered

    # --- Heatmap for Unweathered Samples ---
    plt.figure(figsize=(12, 10))
    ax1 = sns.heatmap(corr_unweathered, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8})
    ax1.set_title(f'{glass_type} - 未风化样本成分相关性', fontsize=16, pad=20)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plot_filename_unweathered = f'heatmap_{glass_type}_unweathered.png'
    plot_filepath_unweathered = os.path.join(output_dir, plot_filename_unweathered)
    plt.savefig(plot_filepath_unweathered)
    plt.close() # Close the figure to free up memory
    report_data[glass_type]['heatmap_path_unweathered'] = plot_filename_unweathered

    # --- Heatmap for Weathered Samples ---
    plt.figure(figsize=(12, 10))
    ax2 = sns.heatmap(corr_weathered, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8})
    ax2.set_title(f'{glass_type} - 风化样本成分相关性', fontsize=16, pad=20)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plot_filename_weathered = f'heatmap_{glass_type}_weathered.png'
    plot_filepath_weathered = os.path.join(output_dir, plot_filename_weathered)
    plt.savefig(plot_filepath_weathered)
    plt.close() # Close the figure to free up memory
    report_data[glass_type]['heatmap_path_weathered'] = plot_filename_weathered
    
    # print(f"相关性热力图已分别保存为 '{plot_filename_unweathered}' 和 '{plot_filename_weathered}'") # Suppressed


# --- Markdown Report Generation ---

def generate_markdown_report(report_data):
    """Generates a comprehensive analysis report in Markdown format."""
    
    markdown_content = "# 古代玻璃化学成分与风化关系分析报告 (基于CLR变换)\n\n"
    markdown_content += f"**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += "本报告对化学成分数据进行了中心对数比（CLR）变换，以消除成分数据固有的约束。所有后续的统计分析（t检验、相关性分析等）都在变换后的数据上进行。这使得分析结果在统计上更为可靠。\n\n---\n\n"
    
    for glass_type, data in report_data.items():
        markdown_content += f"## {glass_type}玻璃分析\n\n"
        
        if 'error' in data:
            markdown_content += f"**分析中止**: {data['error']}\n\n"
            continue
            
        # 1. Component Content Analysis
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
        
        # 2. Visualization
        markdown_content += "\n### 2. 主要成分CLR变换值分布可视化\n\n"
        markdown_content += "以下图通过箱形图展示了CLR值差异显著（或含量最高）的化学成分在风化与未风化样本中的分布情况，直观地体现了其CLR值的中位数、范围和离散程度的变化。\n\n"
        
        # boxplot_paths = data.get('boxplot_paths', []) # Old line, no longer used
        combined_boxplot_path = data.get('combined_boxplot_path', '')
        if combined_boxplot_path:
            # for path in boxplot_paths: # Old loop, no longer used
            markdown_content += f"**{glass_type}玻璃主要成分CLR变换值分布**\n"
            markdown_content += f"![{glass_type}玻璃 - 主要成分CLR值箱形图](./{combined_boxplot_path})\n\n"
        else:
            markdown_content += "未发现CLR值有显著差异的化学成分，或无数据可供可视化。\n\n"

        # 3. Correlation Analysis
        markdown_content += "### 3. CLR变换后成分相关性变化分析\n\n"
        markdown_content += "为探究风化过程中各化学元素间的协同变化关系，我们分别计算了风化前后样本在CLR空间中的成分相关性矩阵，并通过热力图进行可视化。CLR变换消除了成分数据的伪相关性，因此这里的相关性更能反映真实的关联关系。\n\n"
        
        markdown_content += f"**未风化样本CLR变换后成分相关性**\n"
        markdown_content += f"![{glass_type}玻璃未风化成分相关性热力图](./{data.get('heatmap_path_unweathered', '')})\n\n"
        
        markdown_content += f"**风化样本CLR变换后成分相关性**\n"
        markdown_content += f"![{glass_type}玻璃风化成分相关性热力图](./{data.get('heatmap_path_weathered', '')})\n\n"

        markdown_content += "**解读**：对比以上两张热力图，可以观察到风化前后，各化学成分在CLR空间中的相关性（由相关系数的数值和颜色表示）发生了怎样的变化，例如哪些成分间的正/负相关性增强、减弱或逆转。\n\n---\n\n"

    return markdown_content


# --- Main Execution Block ---

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    # --- Define Output Directory ---
    output_dir = '/Users/Mac/Downloads/22C/1/1_2'
    # Ensure the output directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Data Loading ---
    # Construct the absolute path to the CSV file
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '附件2_处理后.csv')

    try:
        # Each row is a unique sampling point and treated as an independent sample
        df = pd.read_csv(file_path, encoding='utf-8')
        # print(f"成功从 '{file_path}' 加载数据。共 {len(df)} 个采样点。") # Suppressed
    except Exception as e:
        print(f"数据加载失败: {e}")
        print(f"请确保 '{os.path.basename(file_path)}' 文件存在于脚本 '{script_dir}' 目录下。")
        return

    # --- Data Cleaning and Preparation ---
    # print("\n--- 开始数据清洗与预处理 ---") # Suppressed
    
    # Identify chemical columns for sum calculation
    excluded_cols = ['文物采样点', '类型', '表面风化']
    chemical_cols = [col for col in df.columns if col not in excluded_cols]
    
    # Fill missing chemical data with 0 for calculation purposes
    df[chemical_cols] = df[chemical_cols].fillna(0)
    
    # Calculate sum of components and filter for "effective data" (85% to 105%)
    df['sum'] = df[chemical_cols].sum(axis=1)
    df_filtered = df[(df['sum'] >= 85) & (df['sum'] <= 105)].copy()
    initial_count = len(df)
    final_count = len(df_filtered)
    # print(f"成分总和筛选：从 {initial_count} 条记录中筛选出 {final_count} 条有效记录 (成分总和在 85% 到 105% 之间)。") # Suppressed
    
    # Map numerical '表面风化' to descriptive strings if necessary
    if df_filtered['表面风化'].dtype in ['int64', 'float64']:
        # print("将'表面风化'列从数值映射为文本...") # Suppressed
        df_filtered = df_filtered.copy()
        df_filtered.loc[df_filtered['表面风化'] == 1, '表面风化'] = '无风化'
        df_filtered.loc[df_filtered['表面风化'] == 2, '表面风化'] = '风化'
    
    # Drop rows with any remaining missing critical data
    df_cleaned = df_filtered.copy()
    # Remove rows where '表面风化' or '类型' are missing
    mask = pd.notna(df_cleaned['表面风化']) & pd.notna(df_cleaned['类型'])
    df_cleaned = df_cleaned[mask]
    
    if len(df_cleaned) < final_count:
        pass # print(f"丢弃 {final_count - len(df_cleaned)} 条缺少'类型'或'表面风化'信息的记录。") # Suppressed
    
    # print(f"数据准备完成。最终用于分析的独立采样点共 {len(df_cleaned)} 个。") # Suppressed
    # print("---------------------------------") # Suppressed
    
    # --- Apply CLR Transformation ---
    # print("\n--- 对化学成分数据应用CLR变换 ---") # Suppressed
    chemical_cols_for_transform = [col for col in chemical_cols if col != 'sum']
    df_clr = apply_clr_transformation(df_cleaned, chemical_cols_for_transform)
    # print("CLR变换完成。") # Suppressed
    
    # --- Execute Analysis ---
    report_data = {} # Dictionary to hold all data for the final report
    glass_types_to_analyze = pd.unique(df_clr['类型'])
    
    for g_type in glass_types_to_analyze:
        analyze_chemical_composition(df_clr, g_type, report_data, output_dir)
    
    # --- Generate and Save Markdown Report ---
    # print(f"\n{'='*60}") # Suppressed
    # print(" 分析完成，开始生成Markdown格式的最终报告") # Suppressed
    # print(f"{'='*60}\n") # Suppressed
    markdown_report = generate_markdown_report(report_data)
    
    report_filename = "chemical_analysis_report.md"
    report_filepath = os.path.join(output_dir, report_filename)
    try:
        with open(report_filepath, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        # print(f"Markdown报告已成功生成并保存为 '{report_filename}'") # Suppressed
    except Exception as e:
        print(f"保存Markdown报告失败: {e}")

    # Display the report content in the final output as well
    # print(markdown_report)

if __name__ == '__main__':
    main()
