import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- 0. 环境设置 ---
output_base_dir = '2.1/'
os.makedirs(output_base_dir, exist_ok=True)
report_content = []

def print_to_console_and_report(message, is_heading=False, is_code_block=False):
    """辅助函数，同时打印到控制台并收集内容到报告列表"""
    print(message)
    if is_heading:
        report_content.append(f"\n## {message.strip(' =')}\n")
    elif is_code_block:
        report_content.append(f"\n```\n{message}\n```\n")
    else:
        report_content.append(f"{message}\n")

def run_clustering_analysis(df, glass_type, features, max_k=6, chosen_k=3):
    """一个完整的聚类分析流程函数"""
    print_to_console_and_report(f"----- 开始处理: {glass_type}玻璃 -----", is_heading=True)

    # 1. 数据准备
    df_type = df[df['类型'] == glass_type].copy()
    if df_type.empty:
        print_to_console_and_report(f"数据集中没有找到类型为 '{glass_type}' 的样本。")
        return None, None
    
    print_to_console_and_report(f"选定的聚类特征: {', '.join(features)}")
    X = df_type[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 肘部法则确定k值
    print_to_console_and_report("\n--- 2.1. 肘部法则确定亚类数量(k) ---")
    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(k_range, sse, marker='o', linestyle='--')
    plt.xlabel('亚类数量 (k)')
    plt.ylabel('误差平方和 (SSE)')
    plt.title(f'{glass_type}玻璃的k-SSE肘部图')
    plt.xticks(k_range)
    plt.grid(True)
    elbow_plot_path = os.path.join(output_base_dir, f'elbow_plot_{glass_type}.png')
    plt.savefig(elbow_plot_path)
    plt.close()
    print_to_console_and_report(f"k-SSE肘部图已保存到: {os.path.basename(elbow_plot_path)}")

    # 3. K-Means聚类
    print_to_console_and_report(f"\n--- 2.2. K-Means聚类 (选择 k={chosen_k}) ---")
    kmeans_final = KMeans(n_clusters=chosen_k, init='k-means++', random_state=42, n_init='auto')
    df_type[f'亚类'] = kmeans_final.fit_predict(X_scaled)

    # 4. 结果分析与可视化
    print_to_console_and_report("\n各亚类化学成分中心 (标准化后):")
    centers_scaled = kmeans_final.cluster_centers_
    centers_df = pd.DataFrame(centers_scaled, columns=features)
    centers_df.index.name = '亚类'
    print_to_console_and_report(centers_df.to_markdown(), is_code_block=True)
    
    print_to_console_and_report("\n各亚类化学成分中心 (原始值):")
    df_type['亚类'] = df_type['亚类'].astype('category')
    centers_original = df_type.groupby('亚类')[features].mean()
    print_to_console_and_report(centers_original.to_markdown(), is_code_block=True)

    # 可视化聚类结果
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    pairplot = sns.pairplot(df_type, vars=features, hue='亚类', palette='viridis')
    pairplot.fig.suptitle(f'{glass_type}玻璃亚类划分的可视化', y=1.02)
    pairplot_path = os.path.join(output_base_dir, f'pairplot_{glass_type}.png')
    pairplot.savefig(pairplot_path)
    plt.close()
    print_to_console_and_report(f"亚类可视化Pair Plot图已保存到: {os.path.basename(pairplot_path)}")

    # 保存模型和缩放器
    joblib.dump(kmeans_final, os.path.join(output_base_dir, f'kmeans_model_{glass_type}.joblib'))
    joblib.dump(scaler, os.path.join(output_base_dir, f'scaler_{glass_type}.joblib'))
    print_to_console_and_report(f"K-Means模型和缩放器已保存。")

    return df_type, centers_original

# --- Main Execution ---
# 1. 加载数据
print_to_console_and_report("="*20 + " 1. 数据加载 " + "="*20, is_heading=True)
try:
    df_main = pd.read_csv('/Users/Mac/Downloads/22C/2.1/附件2_处理后.csv')
    print_to_console_and_report("数据 '2.1/附件2_处理后.csv' 加载完成。")
except FileNotFoundError:
    print_to_console_and_report("错误：未找到 '2/附件2_处理后.csv'。请先运行初始的数据处理脚本。")
    exit()

# 2. 高钾玻璃聚类分析
features_gaojia = ['氧化钾(K2O)', '二氧化硅(SiO2)', '氧化钙(CaO)', '氧化镁(MgO)']
# 根据肘部图，假设选择k=2作为高钾玻璃的最佳亚类数
df_gaojia_clustered, centers_gaojia = run_clustering_analysis(
    df_main, '高钾', features_gaojia, max_k=6, chosen_k=2
)

# 3. 铅钡玻璃聚类分析
features_qianbei = ['氧化铅(PbO)', '氧化钡(BaO)', '二氧化硅(SiO2)', '氧化铝(Al2O3)']
# 根据肘部图，假设选择k=3作为铅钡玻璃的最佳亚类数
df_qianbei_clustered, centers_qianbei = run_clustering_analysis(
    df_main, '铅钡', features_qianbei, max_k=6, chosen_k=3
)

# 4. 合并结果并保存
print_to_console_and_report("="*20 + " 4. 结果合并与保存 " + "="*20, is_heading=True)

# 提取非高钾和非铅钡的样本
other_types_df = df_main[~df_main['类型'].isin(['高钾', '铅钡'])].copy()

# 初始化一个空的DataFrame列表，用于存放处理过的部分
processed_dfs = []

if df_gaojia_clustered is not None and centers_gaojia is not None:
    # 比如，根据K2O含量命名高钾亚类
    gaojia_map = {i: f"高钾-亚类{i+1}" for i in range(len(centers_gaojia))}
    gaojia_series = pd.Series(df_gaojia_clustered['亚类'])
    df_gaojia_clustered['亚类'] = gaojia_series.map(gaojia_map)
    processed_dfs.append(df_gaojia_clustered)

if df_qianbei_clustered is not None and centers_qianbei is not None:
    # 比如，根据PbO含量命名铅钡亚类
    qianbei_map = {i: f"铅钡-亚类{i+1}" for i in range(len(centers_qianbei))}
    qianbei_series = pd.Series(df_qianbei_clustered['亚类'])
    df_qianbei_clustered['亚类'] = qianbei_series.map(qianbei_map)
    processed_dfs.append(df_qianbei_clustered)

# 将其他类型的样本也加入列表
if not other_types_df.empty:
    processed_dfs.append(other_types_df)

# 使用pd.concat进行最终的、健壮的合并
if processed_dfs:
    final_df = pd.concat(processed_dfs).sort_index()
else:
    final_df = df_main.copy() # 如果没有任何聚类发生，则使用原始df

output_csv_path = os.path.join(output_base_dir, '附件2_亚类划分后.csv')
final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print_to_console_and_report(f"包含亚类划分结果的最终数据已保存到: {os.path.basename(output_csv_path)}")

# 5. 生成最终报告
markdown_report_path = os.path.join(output_base_dir, 'subclass_analysis_report.md')
with open(markdown_report_path, 'w', encoding='utf-8') as f:
    f.write("# 文物玻璃亚类划分聚类分析报告\n")
    for line in report_content:
        f.write(line)
    f.write(f"\n## 附图：高钾玻璃\n\n![Elbow Plot for 高钾]({os.path.basename(f'elbow_plot_高钾.png')})\n\n![Pair Plot for 高钾]({os.path.basename(f'pairplot_高钾.png')})\n\n")
    f.write(f"\n## 附图：铅钡玻璃\n\n![Elbow Plot for 铅钡]({os.path.basename(f'elbow_plot_铅钡.png')})\n\n![Pair Plot for 铅钡]({os.path.basename(f'pairplot_铅钡.png')})\n\n")

print("\n" + "="*20 + " 分析报告生成完成 " + "="*20)
print(f"亚类划分分析报告已保存到: {markdown_report_path}") 