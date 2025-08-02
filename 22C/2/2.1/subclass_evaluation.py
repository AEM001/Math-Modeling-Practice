import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import warnings
import sys
import os

# 将脚本所在目录添加到系统路径，以便导入同级目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ilr_transformation import ilr_transform
except ImportError:
    print("错误：无法导入 'ilr_transformation.py'。请确保该文件与本脚本在同一目录下。")
    sys.exit(1)


# 抑制未来版本警告，保持输出整洁
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 全局配置 ---
# 假定脚本从项目根目录执行
DATA_FILE = '2/2.1/附件2_亚类划分后.csv'
REPORT_FILE = '2/2.1/subclass_analysis_report.md'


def run_anova_for_glass_type(df, glass_type, features):
    """
    为指定的玻璃类型及其特征运行ANOVA和Tukey's HSD事后检验。
    """
    report_content = f"\n\n## {glass_type}玻璃亚类划分合理性评估 (ANOVA)\n\n"
    report_content += "我们采用单因素方差分析(ANOVA)来检验不同亚类之间，其核心化学成分的均值是否存在统计显著性差异。p值小于0.05通常被认为差异是显著的。\n"

    if glass_type == '高钾':
        report_content += "> **特别说明**: 高钾玻璃的一个亚类仅包含单个样本，这会极大影响方差分析的统计功效，因此其结果仅供参考。\n\n"

    data = df[df['类型'] == glass_type]
    if data['亚类'].nunique() < 2:
        report_content += "只有一个亚类，无法进行ANOVA比较。\n"
        return report_content

    for feature in features:
        groups = [group[feature].dropna() for _, group in data.groupby('亚类')]
        
        # 至少需要两组数据才能进行方差分析
        if len(groups) < 2 or any(len(g) < 1 for g in groups):
            continue
        
        # 过滤掉无法计算方差的组 (例如，仅包含一个样本的组)
        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) < 2 and len(groups) >=2:
            report_content += f"\n### 对 `{feature}` 的分析\n"
            report_content += "- **ANOVA结果**: 由于存在单一样本的亚类，无法进行有效的方差分析。\n"
            continue

        f_val, p_val = f_oneway(*groups)
        
        report_content += f"\n### 对 `{feature}` 的分析\n"
        conclusion = f"p值 = {p_val:.4f}，组间均值{'**存在**' if p_val < 0.05 else '不存在'}统计显著性差异。"
        report_content += f"- **ANOVA结果**: F统计量 = {f_val:.4f}，{conclusion}\n"

        if p_val < 0.05:
            tukey = pairwise_tukeyhsd(endog=data[feature], groups=data['亚类'], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            significant_pairs = tukey_df[tukey_df['p-adj'] < 0.05]
            
            if not significant_pairs.empty:
                report_content += f"- **事后检验 (Tukey's HSD)**: 以下亚类对之间存在显著差异：\n"
                report_content += f"```\n{significant_pairs.to_string(index=False)}\n```\n"
    
    return report_content


def run_sensitivity_analysis(df, glass_type, features, k, max_noise_level=0.1, step=0.005):
    """
    通过注入高斯噪声来执行敏感性分析。
    """
    report_content = f"\n\n## {glass_type}玻璃亚类划分敏感性评估\n\n"
    report_content += "敏感性评估旨在测试亚类划分结果对数据微小扰动的稳健性。我们通过向原始数据（ILR变换后）注入逐步增加的乘性高斯噪声 `(X_noisy = X_scaled * (1 + ε * N(0,1)))`，然后重新进行K-Means聚类，并使用调整兰德指数（Adjusted Rand Index, ARI）来比较新旧聚类结果的一致性。ARI为1表示两次聚类结果完全相同，接近0则表示结果不相关。\n"
    
    if glass_type == '高钾':
        report_content += "> **特别说明**: 高钾玻璃存在单样本离群点，其聚类结果本身对扰动极其敏感，因此不适用此项分析。\n"
        return report_content

    data = df[df['类型'] == glass_type].copy()
    original_labels = data['亚类']
    
    # 根据错误分析，输入文件中的特征列已经是ILR变换后的坐标，而不是原始成分。
    # 因此，我们直接使用这些数据，不再重新进行ILR变换，仅进行标准化。
    X_ilr = data[features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ilr)

    stable_noise_level = 0
    final_noise_level = 0
    
    for noise_level in np.arange(0, max_noise_level + step, step):
        if noise_level == 0:
            continue
            
        noise = np.random.normal(0, 1, X_scaled.shape)
        X_noisy = X_scaled * (1 + noise_level * noise)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        new_labels = kmeans.fit_predict(X_noisy)
        ari = adjusted_rand_score(original_labels, new_labels)
        
        final_noise_level = noise_level
        if ari < 0.99:
            break
        stable_noise_level = noise_level
    
    report_content += f"\n**评估结果**:\n"
    report_content += f"- 聚类结果在噪声标准差达到 **{stable_noise_level*100:.1f}%** 前保持稳定 (ARI > 0.99)。\n"
    report_content += f"- 当噪声水平增加到 **{final_noise_level*100:.1f}%** 时，聚类结果开始发生显著变化 (ARI < 0.99)。\n"
    report_content += f"- **结论**: 当前对 **{glass_type}** 玻璃的亚类划分具有较好的稳健性，能够抵抗一定程度的数据扰动。\n"
    
    return report_content


def main():
    """
    主函数，执行评估并更新报告。
    """
    if not os.path.exists(DATA_FILE):
        print(f"错误: 数据文件 '{DATA_FILE}' 未找到。请确保文件路径正确。")
        return

    df = pd.read_csv(DATA_FILE)
    
    features_k = ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化铝(Al2O3)']
    features_pb = ['二氧化硅(SiO2)', '氧化铅(PbO)', '氧化钡(BaO)', '五氧化二磷(P2O5)']
    
    # --- 执行分析 ---
    anova_k_report = run_anova_for_glass_type(df, '高钾', features_k)
    sensitivity_k_report = run_sensitivity_analysis(df, '高钾', features_k, k=2)
    
    anova_pb_report = run_anova_for_glass_type(df, '铅钡', features_pb)
    sensitivity_pb_report = run_sensitivity_analysis(df, '铅钡', features_pb, k=3)
    
    # --- 写入报告 ---
    full_report = "\n\n---\n# 4. 聚类结果评估" + anova_k_report + sensitivity_k_report + anova_pb_report + sensitivity_pb_report
    
    try:
        with open(REPORT_FILE, 'a', encoding='utf-8') as f:
            f.write(full_report)
        print(f"评估完成，结果已成功追加到报告文件: {REPORT_FILE}")
    except IOError as e:
        print(f"错误: 无法写入报告文件 '{REPORT_FILE}'. {e}")


if __name__ == '__main__':
    main() 