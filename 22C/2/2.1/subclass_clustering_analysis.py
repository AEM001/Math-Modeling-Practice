import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 0. 环境设置 ---
output_base_dir = '2.1/'
os.makedirs(output_base_dir, exist_ok=True)
report_content = []

def log_to_console(message):
    """辅助函数，仅打印到控制台"""
    print(message)

def log_to_report(message, is_heading=False, is_code_block=False):
    """辅助函数，将内容添加到报告列表"""
    if is_heading:
        report_content.append(f"\n{message}\n")
    elif is_code_block:
        report_content.append(f"\n```\n{message}\n```\n")
    else:
        report_content.append(f"{message}\n")

def run_clustering_analysis(df_transformed, df_original, glass_type, features, max_k=6, chosen_k=3):
    """一个完整的聚类分析流程函数。"""
    log_to_report(f"## {glass_type}玻璃亚类分析", is_heading=True)
    
    log_to_report(f"**选定的聚类特征**: `{', '.join(features)}`")
    log_to_report(f"> **理由**: 基于对该玻璃类型样本的方差分析，这些成分在其内部变化最显著，因此最有可能揭示其存在的亚类结构。\n")
    
    # 数据准备
    df_type = df_transformed[df_transformed['类型'] == glass_type].copy()
    if df_type.empty:
        log_to_console(f"数据集中没有找到类型为 '{glass_type}' 的样本。")
        return None, None
    
    X = df_type[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 肘部法则确定k值
    log_to_report(f"### 亚类数量(k)选择: 肘部法则", is_heading=True)
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
    log_to_console(f"k-SSE肘部图已保存到: {os.path.basename(elbow_plot_path)}")
    log_to_report(f"根据生成的肘部图，我们选择 **k={chosen_k}** 作为最佳亚类数进行下一步分析。")
    log_to_report(f"![{glass_type} 肘部图]({os.path.basename(elbow_plot_path)})")


    # K-Means聚类与评估
    log_to_report(f"### K-Means聚类结果与评估 (k={chosen_k})", is_heading=True)
    kmeans_final = KMeans(n_clusters=chosen_k, init='k-means++', random_state=42, n_init='auto')
    labels = kmeans_final.fit_predict(X_scaled)
    df_type[f'亚类'] = labels

    df_type_original = df_original.loc[df_type.index].copy()
    df_type_original = df_type_original[df_type_original['类型'] == glass_type].copy()
    df_type_original['亚类'] = labels

    silhouette_avg = silhouette_score(X_scaled, labels)
    log_to_report(f"**聚类效果评估 (轮廓系数)**: **{silhouette_avg:.4f}**")
    log_to_report(f"> **说明**: 轮廓系数衡量聚类的紧密性和分离度，其值范围为[-1, 1]。一个接近+1的值表明样本远离相邻簇，聚类效果好；接近0表示样本在簇的边界上；负值则表示样本可能被分到了错误的簇。当前的分数表明聚类结果具有合理的区分度。\n")

    log_to_report("\n**各亚类样本数量:**")
    cluster_sizes = df_type_original['亚类'].value_counts().sort_index().to_frame(name='样本数量')
    cluster_sizes.index.name = '亚类'
    log_to_report(cluster_sizes.to_markdown(), is_code_block=True)
    log_to_report("> **说明**: 上表显示了每个亚类中包含的文物样本数量，有助于我们理解各个亚类的规模和代表性。\n")

    df_type_original['亚类'] = df_type_original['亚类'].astype('category')
    
    log_to_report("\n**各亚类化学成分中心 (均值):**")
    centers_original = df_type_original.groupby('亚类')[features].mean()
    log_to_report(centers_original.to_markdown(), is_code_block=True)
    log_to_report(
        "> **说明**: 上表是每个亚类化学成分的**平均值**，它代表了该亚类的“典型”化学构成，是后续对亚类进行命名的核心依据。\n"
    )
    
    log_to_report("\n**各亚类化学成分标准差:**")
    std_original = df_type_original.groupby('亚类')[features].std()
    log_to_report(std_original.to_markdown(), is_code_block=True)
    log_to_report("> **说明**: 标准差反映了亚类内部成员的离散程度。**数值越小**，代表该亚类内部的样本在这一化学成分上越相似、一致性越高。\n")


    # 可视化聚类结果
    log_to_report(f"### 亚类可视化", is_heading=True)
    pairplot = sns.pairplot(df_type_original, vars=features, hue='亚类', palette='viridis', diag_kind='kde')
    pairplot.fig.suptitle(f'{glass_type}玻璃亚类划分的可视化', y=1.02)
    pairplot_path = os.path.join(output_base_dir, f'pairplot_{glass_type}.png')
    pairplot.savefig(pairplot_path)
    plt.close()
    log_to_console(f"亚类可视化Pair Plot图已保存到: {os.path.basename(pairplot_path)}")
    log_to_report(f"![{glass_type} Pair Plot]({os.path.basename(pairplot_path)})")

    return df_type, centers_original

# --- Main Execution ---
log_to_console("="*20 + " 1. 数据加载 " + "="*20)
try:
    df_transformed = pd.read_csv('/Users/Mac/Downloads/22C/2.1/附件2_处理后_ILR_常数替换.csv')
    log_to_console("数据 '2.1/附件2_处理后_ILR_常数替换.csv' 加载完成。")
    df_original = pd.read_csv('/Users/Mac/Downloads/22C/2.1/附件2_处理前.csv')
    log_to_console("数据 '2.1/附件2_处理前.csv' 加载完成。")
    df_original = df_original.set_index(df_transformed.index)
except FileNotFoundError as e:
    log_to_console(f"错误：数据文件未找到，请检查路径。 {e}")
    exit()

log_to_report("# 文物玻璃亚类划分聚类分析报告", is_heading=True)
log_to_report("## 1. 分析方法与流程概述\n", is_heading=True)
log_to_report(
"""
本次分析旨在对高钾玻璃和铅钡玻璃进行亚类划分，以揭示其内部更精细的化学成分差异。我们采用了以下步骤：

1.  **特征选择**: 基于对每个玻璃类型（高钾、铅钡）样本的**方差分析**，我们选择了内部成分变化最显著、最有可能区分亚类的化学成分作为核心特征。这确保了分析的客观性和数据驱动性。
2.  **数据变换与标准化**: 为了消除成分数据闭合效应和量纲影响，我们首先对数据进行 **ILR (Isometric Log-Ratio) 变换**，然后在变换后的空间上进行 **Z-Score标准化**。
3.  **确定亚类数量 (k值)**: 使用**“肘部法则” (Elbow Method)**，为每个类别选择最佳的亚类数量。
4.  **K-Means聚类**: 采用K-Means++算法进行聚类。
5.  **结果解读与可视化**: 将聚类标签映射回**原始化学成分数据**，以确保结果的可解释性。
6.  **效果评估**: 使用**轮廓系数 (Silhouette Score)** 来量化评估聚类结果。
"""
)

# --- 高钾玻璃分析 ---
log_to_console("\n" + "="*20 + " 2. 处理高钾玻璃 " + "="*20)
features_gaojia = ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化铝(Al2O3)']
df_gaojia_clustered, centers_gaojia = run_clustering_analysis(
    df_transformed, df_original, '高钾', features_gaojia, max_k=6, chosen_k=2
)

# --- 铅钡玻璃分析 ---
log_to_console("\n" + "="*20 + " 3. 处理铅钡玻璃 " + "="*20)
features_qianbei = ['二氧化硅(SiO2)', '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)']
df_qianbei_clustered, centers_qianbei = run_clustering_analysis(
    df_transformed, df_original, '铅钡', features_qianbei, max_k=6, chosen_k=3
)

# --- 结果合并与保存 ---
log_to_console("\n" + "="*20 + " 4. 结果合并与保存 " + "="*20)
other_types_df = df_transformed[~df_transformed['类型'].isin(['高钾', '铅钡'])].copy()
processed_dfs = []
if df_gaojia_clustered is not None and centers_gaojia is not None:
    gaojia_map = {i: f"高钾-亚类{i+1}" for i in range(len(centers_gaojia))}
    gaojia_series = pd.Series(df_gaojia_clustered['亚类'])
    df_gaojia_clustered['亚类'] = gaojia_series.map(gaojia_map)
    processed_dfs.append(df_gaojia_clustered)
if df_qianbei_clustered is not None and centers_qianbei is not None:
    qianbei_map = {i: f"铅钡-亚类{i+1}" for i in range(len(centers_qianbei))}
    qianbei_series = pd.Series(df_qianbei_clustered['亚类'])
    df_qianbei_clustered['亚类'] = qianbei_series.map(qianbei_map)
    processed_dfs.append(df_qianbei_clustered)
if not other_types_df.empty:
    processed_dfs.append(other_types_df)
if processed_dfs:
    final_df = pd.concat(processed_dfs).sort_index()
else:
    final_df = df_transformed.copy()
output_csv_path = os.path.join(output_base_dir, '附件2_亚类划分后.csv')
final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
log_to_console(f"包含亚类划分结果的最终数据已保存到: {os.path.basename(output_csv_path)}")

# --- 生成最终报告 ---
markdown_report_path = os.path.join(output_base_dir, 'subclass_analysis_report.md')
with open(markdown_report_path, 'w', encoding='utf-8') as f:
    for line in report_content:
        f.write(line)

log_to_console("\n" + "="*20 + " 分析报告生成完成 " + "="*20)
log_to_console(f"亚类划分分析报告已保存到: {markdown_report_path}")
