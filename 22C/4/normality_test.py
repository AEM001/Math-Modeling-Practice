import pandas as pd
from scipy import stats
import os

def normality_test_report(df, chem_cols):
    """
    对给定的DataFrame中的每一类玻璃的化学成分进行Shapiro-Wilk正态性检验。
    
    参数:
    - df: 包含所有数据的DataFrame。
    - chem_cols: 要分析的化学成分列名列表。
    
    返回:
    - test_results_df: 包含所有检验结果的DataFrame。
    """
    # 按类型分离数据
    df_gaojia = df[df['类型'] == '高钾'].copy()
    df_qianbei = df[df['类型'] == '铅钡'].copy()
    
    results = []
    
    # 对两类数据分别进行检验
    for glass_type, df_type in [('高钾', df_gaojia), ('铅钡', df_qianbei)]:
        for col in chem_cols:
            # 执行Shapiro-Wilk检验
            # H0: 数据服从正态分布
            stat, p_value = stats.shapiro(df_type[col])
            
            results.append({
                '玻璃类型': glass_type,
                '化学成分': col,
                'W统计量': stat,
                'p值': p_value,
                '是否服从正态分布 (α=0.05)': '是' if p_value > 0.05 else '否'
            })
            
    return pd.DataFrame(results)

def main():
    """主执行函数"""
    # --- 设置 ---
    INPUT_PATH = '4/附件2_处理后_CLR.csv'
    OUTPUT_DIR = '4/analysis_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 加载数据 ---
    print("正在加载数据...")
    df = pd.read_csv(INPUT_PATH)
    
    # 识别化学成分列
    info_cols = ['文物采样点', '类型', '表面风化']
    chem_cols = [col for col in df.columns if col not in info_cols]
    
    # --- 执行正态性检验 ---
    print("正在对所有化学成分进行Shapiro-Wilk正态性检验...")
    normality_results = normality_test_report(df, chem_cols)
    
    # --- 保存并展示结果 ---
    output_path = os.path.join(OUTPUT_DIR, 'normality_test_results.csv')
    normality_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n正态性检验结果已保存至: {output_path}")
    
    print("\n" + "="*50)
    print("正态性检验结果摘要")
    print("="*50)
    print(normality_results.to_string())
    
    # --- 结论 ---
    print("\n" + "="*50)
    print("结论")
    print("="*50)
    
    # 统计不服从正态分布的变量数量
    non_normal_gaojia = normality_results[(normality_results['玻璃类型'] == '高钾') & (normality_results['p值'] <= 0.05)].shape[0]
    non_normal_qianbei = normality_results[(normality_results['玻璃类型'] == '铅钡') & (normality_results['p值'] <= 0.05)].shape[0]
    total_vars = len(chem_cols)
    
    print(f"在高钾玻璃中, {total_vars}个化学成分里有 {non_normal_gaojia} 个不服从正态分布。")
    print(f"在铅钡玻璃中, {total_vars}个化学成分里有 {non_normal_qianbei} 个不服从正态分布。")
    
    if non_normal_gaojia > total_vars / 3 or non_normal_qianbei > total_vars / 3:
        print("\n决策：由于大量变量不满足正态性假设，使用Pearson相关系数可能导致结果不准确。")
        print("建议将相关性分析方法切换为Spearman等级相关系数，它更适用于非正态数据。")
    else:
        print("\n决策：大部分变量满足正态性假设，可以继续使用Pearson相关系数。")

if __name__ == '__main__':
    main() 