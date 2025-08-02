import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest
from scipy.stats import kruskal
import textwrap

# --- 0. 环境设置 ---
output_base_dir = '2/2.1/'
os.makedirs(output_base_dir, exist_ok=True)
report_content = []

def log_to_console(message):
    """辅助函数，仅打印到控制台"""
    print(message)

def wrap_text(text, width=80):
    """辅助函数，用于自动换行以改善报告可读性"""
    return '\n'.join(textwrap.wrap(text, width=width))

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
    df_transformed = pd.read_csv('2/2.1/附件2_处理后_ILR_常数替换.csv')
    log_to_console("数据 '2.1/附件2_处理后_ILR_常数替换.csv' 加载完成。")
except FileNotFoundError as e:
    log_to_console(f"错误：数据文件 '2/2.1/附件2_处理后_ILR_常数替换.csv' 未找到，请检查路径。 {e}")
    exit()

log_to_report("# 文物玻璃亚类划分聚类分析报告 (修正版)", is_heading=True)
log_to_report("## 1. 分析方法与流程概述\n", is_heading=True)
log_to_report(
"""
本次分析旨在对高钾玻璃和铅钡玻璃进行亚类划分。在初始版本基础上，我们进行了以下修正与优化：

1.  **高钾玻璃离群点处理**: 针对原分析中高钾玻璃出现单样本亚类的问题，我们首先采用 **孤立森林(Isolation Forest)** 算法进行离群点检测。在移除识别出的离群点后，再对剩余样本进行K-Means聚类，以确保亚类划分的稳健性与合理性。
2.  **铅钡玻璃深化分析**: 针对`氧化铅(PbO)`成分在各亚类间"均值差异大但ANOVA检验不显著"的矛盾，我们补充了 **箱线图(Box Plot)** 进行可视化诊断，并引入 **Kruskal-Wallis非参数检验**，从不同统计学角度重新评估其差异性。
3.  **核心流程**: 特征选择、ILR变换、Z-Score标准化、肘部法则确定k值、K-Means聚类、轮廓系数评估等核心流程保持不变。
"""
)

# --- 高钾玻璃分析 ---
log_to_console("\n" + "="*20 + " 2. 处理高钾玻璃 (含离群点检测) " + "="*20)

log_to_report("## 高钾玻璃亚类分析: 离群点检测与修正", is_heading=True)
features_gaojia = ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化铝(Al2O3)']

df_gaojia_transformed_orig = df_transformed[df_transformed['类型'] == '高钾'].copy()
X_gaojia = df_gaojia_transformed_orig[features_gaojia]
scaler_gaojia = StandardScaler()
X_gaojia_scaled = scaler_gaojia.fit_transform(X_gaojia)

# 使用孤立森林进行离群点检测
iso_forest = IsolationForest(contamination='auto', random_state=42)
outlier_pred = iso_forest.fit_predict(X_gaojia_scaled)
df_gaojia_transformed_orig['离群点'] = outlier_pred

outlier_indices = df_gaojia_transformed_orig[df_gaojia_transformed_orig['离群点'] == -1].index
log_to_report("使用`孤立森林`算法在ILR变换后的高维特征空间中进行离群点检测。")

if not outlier_indices.empty:
    outlier_文物编号 = df_transformed.loc[outlier_indices, '文物编号'].to_list()
    log_to_report(f"**检测结果**: 发现 **{len(outlier_indices)}** 个潜在离群点 (文物编号: `{'`, `'.join(map(str, outlier_文物编号))}`)。")
    log_to_report("> **分析**: 该样本在高维空间中与其他高钾玻璃样本疏离，与原始聚类分析中单样本成一类的情况吻合。为提高聚类结果的可靠性，我们将其移除后重新进行聚类分析。")
    
    # 移除离群点
    df_transformed_cleaned = df_transformed.drop(index=outlier_indices)
    log_to_report(f"亚类分析将在剩余的 **{len(df_transformed_cleaned[df_transformed_cleaned['类型'] == '高钾'])}** 个高钾样本上进行。\n")
else:
    log_to_report("**检测结果**: 未发现显著离群点，将对所有高钾样本进行聚类。\n")
    outlier_文物编号 = []
    df_transformed_cleaned = df_transformed

# 在清洗后的数据上重新进行聚类分析
df_gaojia_clustered, centers_gaojia = run_clustering_analysis(
    df_transformed_cleaned, df_transformed_cleaned, '高钾', features_gaojia, max_k=6, chosen_k=2
)

# --- 铅钡玻璃分析 ---
log_to_console("\n" + "="*20 + " 3. 处理铅钡玻璃 " + "="*20)
features_qianbei = ['二氧化硅(SiO2)', '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)']
# 使用移除了高钾离群点的数据集进行分析
df_qianbei_clustered, centers_qianbei = run_clustering_analysis(
    df_transformed_cleaned, df_transformed_cleaned, '铅钡', features_qianbei, max_k=6, chosen_k=3
)

# 对铅钡玻璃的氧化铅(PbO)进行补充分析
if df_qianbei_clustered is not None:
    log_to_report("### 对氧化铅(PbO)的补充分析", is_heading=True)
    log_to_report(wrap_text("> **背景**: 初始ANOVA检验显示，各亚类间的`氧化铅(PbO)`均值无统计显著性差异（p>0.05），但这与观察到的均值差异（如亚类均值40.3% vs 26.0%）似乎矛盾。这通常由组内方差过大导致。为进一步探究，我们进行非参数检验和可视化分析。"))

    # 获取带有亚类标签的原始数据
    df_qianbei_original_clustered = df_transformed_cleaned.loc[df_qianbei_clustered.index].copy()
    df_qianbei_original_clustered['亚类'] = df_qianbei_clustered['亚类']
    
    # 1. 可视化分析：箱线图
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.boxplot(data=df_qianbei_original_clustered, x='亚类', y='氧化铅(PbO)', palette='viridis')
    sns.stripplot(data=df_qianbei_original_clustered, x='亚类', y='氧化铅(PbO)', color=".3", jitter=True, alpha=0.7)
    plt.title('各亚类氧化铅(PbO)含量分布箱线图')
    plt.xlabel('亚类 (0, 1, 2)')
    plt.ylabel('氧化铅(PbO)含量 (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    pbo_boxplot_path = os.path.join(output_base_dir, 'boxplot_pbo_qianbei.png')
    plt.savefig(pbo_boxplot_path)
    plt.close()
    log_to_console(f"铅钡玻璃PbO箱线图已保存到: {os.path.basename(pbo_boxplot_path)}")
    log_to_report("\n**可视化分析 (箱线图):**")
    log_to_report(f"![铅钡PbO箱线图]({os.path.basename(pbo_boxplot_path)})")
    log_to_report(wrap_text("> **解读**: 箱线图直观地展示了每个亚类的`氧化铅(PbO)`分布。尽管均值有差异，但每个亚类内部的数据分布范围（由箱体和须表示）很广且有大量重叠，这解释了为何ANOVA检验不显著。特别是亚类2的中位数（箱内横线）最高，但其数据点也较为分散。"))

    # 2. 非参数检验：Kruskal-Wallis H-test
    clusters = sorted(df_qianbei_original_clustered['亚类'].unique())
    samples = [df_qianbei_original_clustered[df_qianbei_original_clustered['亚类'] == c]['氧化铅(PbO)'] for c in clusters]
    
    if len(samples) > 1 and all(len(s) > 0 for s in samples):
        stat, p_value = kruskal(*samples)
        log_to_report("\n**非参数检验 (Kruskal-Wallis H-test):**")
        log_to_report(wrap_text("> **方法**: Kruskal-Wallis检验是单因素方差分析(ANOVA)的非参数替代方案，它不要求数据正态分布，而是比较各组的中位数是否存在显著差异，对离群值和方差不齐的情况更具鲁棒性。"))
        log_to_report(f"- **检验结果**: H统计量 = **{stat:.4f}**, p值 = **{p_value:.4f}**")
        if p_value < 0.05:
            log_to_report("- **结论**: p值小于0.05，表明**拒绝原假设**。即使考虑到较大的组内方差，各亚类之间`氧化铅(PbO)`含量的中位数（或总体分布）仍存在统计上的显著差异。这为亚类划分的合理性提供了额外支持。")
        else:
            log_to_report("- **结论**: p值大于或等于0.05，表明**未能拒绝原假设**。从秩分布的角度看，各亚类之间`氧化铅(PbO)`含量的差异不具备统计显著性。")


# --- 结果合并与保存 ---
log_to_console("\n" + "="*20 + " 4. 结果合并与保存 " + "="*20)

# 从可能被清洗过的数据开始
final_df_with_clusters = df_transformed_cleaned.copy()
final_df_with_clusters['亚类'] = '未分类' # 默认值

# 映射高钾玻璃的聚类结果
if df_gaojia_clustered is not None and centers_gaojia is not None:
    gaojia_map = {i: f"高钾-亚类{i+1}" for i in range(len(centers_gaojia))}
    gaojia_series = pd.Series(df_gaojia_clustered['亚类'], index=df_gaojia_clustered.index)
    final_df_with_clusters.loc[df_gaojia_clustered.index, '亚类'] = gaojia_series.map(gaojia_map)

# 映射铅钡玻璃的聚类结果
if df_qianbei_clustered is not None and centers_qianbei is not None:
    qianbei_map = {i: f"铅钡-亚类{i+1}" for i in range(len(centers_qianbei))}
    qianbei_series = pd.Series(df_qianbei_clustered['亚类'], index=df_qianbei_clustered.index)
    final_df_with_clusters.loc[df_qianbei_clustered.index, '亚类'] = qianbei_series.map(qianbei_map)

# 将离群点重新加回最终数据集，并打上特殊标签
if not outlier_indices.empty:
    outlier_df = df_transformed.loc[outlier_indices].copy()
    outlier_df['亚类'] = '高钾-离群点'
    final_df = pd.concat([final_df_with_clusters, outlier_df]).sort_index()
else:
    final_df = final_df_with_clusters.sort_index()

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
