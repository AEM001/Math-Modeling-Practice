import pandas as pd
import numpy as np
import os

def apply_clr_transformation(df, chemical_cols):
    """
    对化学成分数据应用中心对数比（CLR）变换。
    此版本严格遵循用户定义的逻辑：仅处理非零值，在计算几何平均数时忽略零值，
    并在变换后的数据中将它们保留为0。

    Args:
        df (pd.DataFrame): 包含原始数据的DataFrame。
        chemical_cols (list): 需要进行变换的化学成分列名列表。

    Returns:
        pd.DataFrame: 包含CLR变换后数据的新DataFrame。
    """
    print("--- 开始应用中心对数比(CLR)变换 ---")
    
    # 创建一个副本以避免修改原始DataFrame
    df_transformed = df.copy()
    
    # 仅提取需要变换的化学成分数据
    composition_data = df_transformed[chemical_cols].copy()

    # 步骤 1: 将0替换为NaN，以便在计算中忽略它们
    composition_calc = composition_data.replace(0, np.nan)
    print(f"  - 步骤 1: 已将 {composition_data.isin([0]).sum().sum()} 个零值临时替换为NaN。")

    # 步骤 2: 忽略NaN计算每行的几何平均值
    # 几何平均值 = exp(mean(log(x)))。np.nanmean会自动忽略NaN。
    log_data = np.log(composition_calc)
    mean_log_data = np.nanmean(log_data, axis=1)
    geometric_means = np.exp(mean_log_data)
    print("  - 步骤 2: 已为每行计算非零成分的几何平均值。")

    # 步骤 3: 执行CLR变换 log(x_i / g(x))
    # 原始为0（现在是NaN）的值，其结果也会是NaN。
    clr_data = np.log(composition_calc.div(geometric_means, axis=0))
    print("  - 步骤 3: 已执行CLR变换。")

    # 步骤 4: 将结果中的NaN替换回0
    clr_data.fillna(0, inplace=True)
    print("  - 步骤 4: 已将结果中的NaN恢复为0。")
    
    # 将变换后的数据更新回主DataFrame
    df_transformed[chemical_cols] = clr_data
    
    print("--- CLR变换完成 ---\n")
    return df_transformed

def main():
    """
    主函数，执行数据加载、变换和保存的完整流程。
    """
    # 定义数据目录和文件路径
    data_dir = '/Users/Mac/Downloads/22C/4'
    input_file_path = os.path.join(data_dir, '附件2_处理后.csv')
    output_file_path = os.path.join(data_dir, '附件2_处理后_CLR.csv')
    
    # 确保输出目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. 加载数据
    try:
        df = pd.read_csv(input_file_path)
        print(f"成功从 '{input_file_path}' 加载数据。共 {len(df)} 行。\n")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: '{input_file_path}'")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    # 2. 自动识别化学成分列
    # 假设除了标识列外，所有数值列都是化学成分
    excluded_cols = ['文物采样点', '类型', '表面风化', 'sum']
    chemical_cols = [col for col in df.columns if col not in excluded_cols and df[col].dtype in ['float64', 'int64']]
    print(f"识别出的化学成分列共 {len(chemical_cols)} 个: {', '.join(chemical_cols)}\n")

    # 3. 应用CLR变换
    df_clr = apply_clr_transformation(df, chemical_cols)

    # 4. 保存变换后的数据
    try:
        df_clr.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"变换后的数据已成功保存至 '{output_file_path}'")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == '__main__':
    main() 