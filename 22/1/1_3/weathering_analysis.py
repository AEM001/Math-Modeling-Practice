import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

def generate_detailed_report(path_附件2: str, output_md_path: str):
    """
    加载数据，按"类型"分组，为每个组进行独立分析，
    并将每一步的详细结果输出到一个Markdown文件中。
    """
    print("开始执行详细报告生成流程...")
    # --- 1. 数据加载与准备 ---
    df = pd.read_csv(path_附件2)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    chem_cols = [col for col in df.columns if '(' in col and ')' in col]
    identifier_cols = [col for col in df.columns if col in chem_cols or col in ['文物采样点', '类型', '表面风化']]
    
    glass_types = df['类型'].unique()
    print(f"在数据中发现的类型: {glass_types}")
    
    # 准备报告内容
    report_content = "# 按类型分类的风化预测分析详细报告\n\n"
    report_content += "本文档详细记录了针对不同类型玻璃文物的风化预测分析过程，旨在提供完全的透明度以便于检查。\n\n"

    # --- 2. 分类型进行独立分析与报告 ---
    for glass_type in glass_types:
        df_group = df[df['类型'] == glass_type].copy()
        
        report_content += f"---\n\n## 分析类型: {glass_type}\n\n"
        report_content += f"此部分的所有分析仅基于 **{glass_type}** 类型的 **{len(df_group)}** 个样本。\n\n"

        # --- 步骤 2a: 聚类分析 ---
        report_content += "### 步骤 1: 聚类分配\n"
        report_content += "使用K-Means算法，仅根据化学成分将此类型内的样本分为4个簇。\n\n"
        
        n_clusters = 4
        if len(df_group) < n_clusters:
            report_content += f"**警告**: 样本数量 ({len(df_group)}) 少于要求的簇数 ({n_clusters})，无法进行聚类分析。已跳过此类型。\n\n"
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df_group['簇'] = kmeans.fit_predict(df_group[chem_cols])
        
        # 将聚类结果添加到报告
        cluster_assignment = df_group[['文物采样点', '簇']].sort_values('簇')
        report_content += cluster_assignment.to_markdown(index=False) + "\n\n"

        # --- 步骤 2b: 时序关系建立 ---
        report_content += "### 步骤 2: 时序关系建立\n"
        report_content += "根据每个簇的风化比例（从低到高）来确定风化过程的时序 `t` (1-4)。\n\n"
        
        df_group['风化数值'] = df_group['表面风化'].apply(lambda x: 1 if x == '风化' else 0)
        cluster_info = df_group.groupby('簇')['风化数值'].agg(['sum', 'count'])
        cluster_info.rename(columns={'sum': '风化样本数', 'count': '样本总数'}, inplace=True)
        cluster_info['风化比例'] = cluster_info['风化样本数'] / cluster_info['样本总数']
        sorted_clusters = cluster_info.sort_values('风化比例')
        t_mapping = {cluster_index: t_value for t_value, cluster_index in enumerate(sorted_clusters.index, 1)}
        sorted_clusters['时序t'] = sorted_clusters.index.map(t_mapping)
        
        report_content += sorted_clusters.to_markdown() + "\n\n"
        df_group['时序t'] = df_group['簇'].map(t_mapping)

        # --- 步骤 2c: 拟合回归模型 ---
        regression_coeffs = pd.DataFrame(index=chem_cols, columns=['a', 'b', 'c'])
        for component in chem_cols:
            coeffs = np.polyfit(df_group['时序t'], df_group[component], 2)
            regression_coeffs.loc[component] = coeffs
            
        # --- 步骤 2d: 预测与逆变换 ---
        weathered_samples = df_group[df_group['表面风化'] == '风化'].copy()
        if weathered_samples.empty:
            report_content += "### 步骤 3 & 4: 预测结果\n\n此类型中没有风化样本，无需预测。\n\n"
            continue
            
        # CLR 空间预测
        report_content += "### 步骤 3: CLR空间中的预测结果\n"
        report_content += "对风化样本，预测其在未风化状态 (t=1) 时的化学成分，结果处于CLR变换空间。\n\n"
        
        predictions_clr = weathered_samples[['文物采样点']].copy()
        for component in chem_cols:
            a, b, _ = regression_coeffs.loc[component]
            t = weathered_samples['时序t']
            y = weathered_samples[component]
            c1 = y - (a * t**2 + b * t)
            y_pred_clr = a + b + c1
            predictions_clr[component] = y_pred_clr
            
        report_content += predictions_clr.to_markdown(index=False) + "\n\n"
        
        # 逆CLR变换
        report_content += "### 步骤 4: 最终预测结果 (原始百分比)\n"
        report_content += "将CLR空间中的预测值进行逆变换，得到真实的化学成分百分比含量。\n\n"
        
        predicted_chem_data = predictions_clr[chem_cols]
        exp_data = np.exp(predicted_chem_data)
        sum_exp_data = exp_data.sum(axis=1)
        inversed_data = exp_data.div(sum_exp_data, axis=0) * 100
        
        final_predictions = pd.concat([predictions_clr[['文物采样点']], inversed_data], axis=1)
        report_content += final_predictions.to_markdown(index=False) + "\n\n"

    # --- 3. 保存报告 ---
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"详细分析报告已成功生成于: {output_md_path}")


if __name__ == '__main__':
    base_dir = '/Users/Mac/Downloads/22C/22/1/1_3'
    path_附件2 = os.path.join(base_dir, '附件2_处理后_CLR.csv')
    report_output_file = os.path.join(base_dir, '分类分析详细报告.md')
    
    generate_detailed_report(path_附件2, report_output_file) 