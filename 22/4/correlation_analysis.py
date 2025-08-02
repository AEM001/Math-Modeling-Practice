import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os

def calculate_correlation_and_significance(df, chem_cols):
    """
    计算给定DataFrame中化学成分的相关系数矩阵和对应的p值矩阵。
    
    参数:
    - df: 包含数据的DataFrame。
    - chem_cols: 要分析的化学成分列名列表。
    
    返回:
    - corr_matrix: Pearson相关系数矩阵。
    - p_matrix: t检验的p值矩阵。
    """
    df_chem = df[chem_cols]
    n = len(df_chem)
    corr_matrix = df_chem.corr(method='spearman')
    
    p_matrix = pd.DataFrame(np.zeros(corr_matrix.shape), columns=chem_cols, index=chem_cols)
    for col1 in chem_cols:
        for col2 in chem_cols:
            if col1 == col2:
                p_matrix.loc[col1, col2] = 0.0
            else:
                r = corr_matrix.loc[col1, col2]
                if pd.isna(r):
                    p_matrix.loc[col1, col2] = np.nan
                    continue
                # 计算t统计量和p值
                t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                p_value = stats.t.sf(np.abs(t_stat), n - 2) * 2  # 双尾检验
                p_matrix.loc[col1, col2] = p_value
                
    return corr_matrix, p_matrix

def plot_heatmap(corr_matrix, p_matrix, glass_type, output_dir, significance_level=0.05):
    """
    绘制相关性热力图，并在显著相关的单元格上标注星号。
    
    参数:
    - corr_matrix: 相关系数矩阵。
    - p_matrix: p值矩阵。
    - glass_type: 玻璃类型 ('高钾' 或 '铅钡')。
    - output_dir: 图片保存目录。
    - significance_level: 显著性水平。
    """
    plt.figure(figsize=(16, 12))
    
    # 根据p值创建标注 (显著的标'*')
    annot_text = p_matrix.applymap(lambda p: '*' if p < significance_level else '')
    
    # 绘制热力图
    sns.heatmap(corr_matrix, cmap='vlag', center=0, annot=annot_text, fmt='s',
                linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(f'{glass_type}玻璃化学成分Spearman相关性热力图 (显著性水平 α=0.05)', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'heatmap_{glass_type}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"热力图已保存至: {output_path}")

def compare_correlations_fisher_z(corr1, n1, corr2, n2, glass_type1, glass_type2):
    """
    使用Fisher's Z变换比较两种类型玻璃中化学成分的相关性差异。
    
    参数:
    - corr1, corr2: 两个类型的相关系数矩阵。
    - n1, n2: 两个类型的样本量。
    - glass_type1, glass_type2: 两个类型的名称。
    
    返回:
    - comparison_results: 包含所有成分对比较结果的DataFrame。
    """
    # 找出在任一类型中强相关的成分对
    strong_corr_mask = (corr1.abs() > 0.5) | (corr2.abs() > 0.5)
    
    results = []
    
    for col1 in corr1.columns:
        for col2 in corr1.columns:
            if col1 >= col2: # 避免重复和自身比较
                continue
                
            if not strong_corr_mask.loc[col1, col2]:
                continue

            r1 = corr1.loc[col1, col2]
            r2 = corr2.loc[col1, col2]
            
            # Fisher Z变换
            z1 = np.arctanh(r1)
            z2 = np.arctanh(r2)
            
            # 计算检验统计量Z
            se_diff = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
            z_stat = (z1 - z2) / se_diff
            p_value = stats.norm.sf(np.abs(z_stat)) * 2 # 双尾检验
            
            results.append({
                '成分对': f'{col1} - {col2}',
                f'相关系数_{glass_type1} (r1)': r1,
                f'相关系数_{glass_type2} (r2)': r2,
                'Z分数': z_stat,
                'p值': p_value,
                '差异是否显著 (α=0.05)': '是' if p_value < 0.05 else '否'
            })
            
    return pd.DataFrame(results)

def main():
    """主执行函数"""
    # --- 0. 设置 ---
    INPUT_PATH = '4/附件2_处理后_CLR.csv'
    OUTPUT_DIR = '4/analysis_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. 加载和准备数据 ---
    print("正在加载和准备数据...")
    df = pd.read_csv(INPUT_PATH)
    
    # 识别化学成分列
    info_cols = ['文物采样点', '类型', '表面风化']
    chem_cols = [col for col in df.columns if col not in info_cols]
    
    # 按类型分离数据
    df_gaojia = df[df['类型'] == '高钾'].copy()
    df_qianbei = df[df['类型'] == '铅钡'].copy()
    n_gaojia = len(df_gaojia)
    n_qianbei = len(df_qianbei)
    print(f"数据分离完成: 高钾玻璃({n_gaojia}条), 铅钡玻璃({n_qianbei}条)。")
    
    # --- 2. 同类内部关联分析 ---
    print("\n" + "="*20 + " 1. 同类内部关联分析 " + "="*20)
    for glass_type, df_type in [('高钾', df_gaojia), ('铅钡', df_qianbei)]:
        print(f"\n--- 正在分析 {glass_type} 玻璃 ---")
        
        # 计算相关性和显著性
        corr, p_values = calculate_correlation_and_significance(df_type, chem_cols)
        
        # 保存结果到CSV
        corr_path = os.path.join(OUTPUT_DIR, f'correlation_{glass_type}.csv')
        p_path = os.path.join(OUTPUT_DIR, f'pvalues_{glass_type}.csv')
        corr.to_csv(corr_path, encoding='utf-8-sig')
        p_values.to_csv(p_path, encoding='utf-8-sig')
        print(f"相关系数矩阵已保存至: {corr_path}")
        print(f"p值矩阵已保存至: {p_path}")
        
        # 可视化
        plot_heatmap(corr, p_values, glass_type, OUTPUT_DIR)
        
    # --- 3. 类间关联差异比较 ---
    print("\n" + "="*20 + " 2. 类间关联差异比较 " + "="*20)
    corr_gaojia, _ = calculate_correlation_and_significance(df_gaojia, chem_cols)
    corr_qianbei, _ = calculate_correlation_and_significance(df_qianbei, chem_cols)
    
    comparison_results = compare_correlations_fisher_z(corr_gaojia, n_gaojia, corr_qianbei, n_qianbei, '高钾', '铅钡')
    
    # 保存比较结果
    comparison_path = os.path.join(OUTPUT_DIR, 'fisher_z_test_results.csv')
    comparison_results.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\nFisher Z检验结果已保存至: {comparison_path}")
    
    # 打印显著差异的结果
    significant_diffs = comparison_results[comparison_results['差异是否显著 (α=0.05)'] == '是']
    print("\n--- 具有显著差异的化学成分关联对 ---")
    if significant_diffs.empty:
        print("在α=0.05的显著性水平下，未发现两类玻璃中存在显著差异的化学成分关联。")
    else:
        print(significant_diffs.to_string())

if __name__ == '__main__':
    # 设置中文字体以避免乱码
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    main() 