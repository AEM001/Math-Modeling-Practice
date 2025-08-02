import pandas as pd
import warnings
import numpy as np
import os

# --- Global Settings ---
warnings.filterwarnings('ignore')

# --- Transformation Functions ---

def apply_clr_transformation(df, chemical_cols):
    """
    对化学成分数据应用中心对数比（CLR）变换。
    此版本仅处理非零值，在计算几何平均数时忽略零值，
    并在变换后的数据中将它们保留为0。
    """
    df_transformed = df.copy()
    composition_data = df_transformed[chemical_cols].copy()
    composition_calc = composition_data.replace(0, np.nan)
    
    log_data = np.log(composition_calc)
    mean_log_data = np.nanmean(log_data, axis=1)
    geometric_means = np.exp(mean_log_data)
    
    clr_data = np.log(composition_calc.div(geometric_means, axis=0))
    clr_data.fillna(0, inplace=True)
    
    df_transformed[chemical_cols] = clr_data
    return df_transformed

def inverse_clr_transformation(df, chemical_cols):
    """
    对CLR变换后的数据应用逆变换，将其还原为百分比成分。
    在逆变换前，将原始的零值（在CLR空间中为0）映射为-inf，
    以确保它们在变换后正确地返回为0。
    """
    df_inv = df.copy()
    clr_data = df_inv[chemical_cols].copy()

    # 将在CLR空间中为0的值（对应原始的0）替换为-inf
    clr_data.replace(0, -np.inf, inplace=True)
    
    exp_data = np.exp(clr_data)
    sum_exp_data = exp_data.sum(axis=1)
    
    composition_data = exp_data.div(sum_exp_data, axis=0).multiply(100)
    composition_data.fillna(0, inplace=True)
    
    df_inv[chemical_cols] = composition_data
    return df_inv

# --- Main Analysis Functions ---

def load_deltas_from_csv(analysis_files):
    """
    从分析结果CSV文件中加载CLR空间中的统计规律（均值差异Delta）。
    """
    print("--- 步骤 1: 从CSV文件加载风化规律（CLR空间差异） ---")
    
    deltas = {}
    for glass_type, file_path in analysis_files.items():
        try:
            df_analysis = pd.read_csv(file_path, encoding='utf-8')
            deltas[glass_type] = {}
            print(f"\n成功加载规律文件: '{file_path}'")
            print(f"开始为玻璃类型 '{glass_type}' 分析规律...")

            significant_changes = df_analysis[df_analysis['是否显著'] == '是'].copy()
            
            if significant_changes.empty:
                print(f"  - 在 '{file_path}' 中未找到任何显著变化的成分。")
            else:
                print(f"  - 找到 {len(significant_changes)} 个显著变化的成分。开始计算差值 Delta...")

            for _, row in significant_changes.iterrows():
                component = row['component']
                # 这两个均值是在CLR空间中的值
                mean_u = row['mean_unweathered']
                mean_w = row['mean_weathered']
                
                delta = mean_w - mean_u
                deltas[glass_type][component] = delta
                print(f"  - 成分 '{component}' 变化显著。计算差值 Delta = {delta:.4f}")

        except FileNotFoundError:
            print(f"  - 错误: 分析文件 '{file_path}' 未找到，跳过该类型。")
            continue
            
    print("\n--- 从CSV加载规律完成 ---\n")
    return deltas

def predict_pre_weathering_composition(df_to_predict, deltas):
    """
    根据风化样本的实测数据和计算出的CLR空间差值Delta，预测其风化前的成分含量。
    """
    print("--- 步骤 2: 对风化样本进行风化前成分预测 (CLR方法) ---")
    
    weathered_samples = df_to_predict[df_to_predict['表面风化'] == '风化'].copy()
    if weathered_samples.empty:
        print("数据中没有需要预测的风化样本。")
        return None

    print(f"找到 {len(weathered_samples)} 个风化采样点需要进行预测。")
    
    # 自动识别化学成分列
    excluded_cols = ['文物采样点', '类型', '表面风化', 'sum']
    chemical_cols = [col for col in df_to_predict.columns if col not in excluded_cols and df_to_predict[col].dtype in ['float64', 'int64']]

    # 1. 对风化样本应用CLR变换
    weathered_clr = apply_clr_transformation(weathered_samples, chemical_cols)
    print("  - 已对风化样本应用CLR变换。")

    predicted_clr = weathered_clr.copy()

    # 2. 在CLR空间中进行预测
    for index, row in weathered_clr.iterrows():
        glass_type = row['类型']
        
        if glass_type not in deltas:
            continue
            
        type_deltas = deltas[glass_type]

        for col in chemical_cols:
            # 仅对显著变化的成分应用差值
            if col in type_deltas:
                delta = type_deltas[col]
                predicted_clr.loc[index, col] = row[col] - delta
    
    print("  - 已在CLR空间中完成预测。")
    
    # 3. 对预测结果应用逆CLR变换，还原为百分比
    predictions_df = inverse_clr_transformation(predicted_clr, chemical_cols)
    print("  - 已对预测结果应用逆CLR变换。")
    
    # 重新计算百分比总和
    predictions_df['sum'] = predictions_df[chemical_cols].sum(axis=1)

    print("\n--- 预测完成 ---\n")
    return predictions_df

def save_predictions_to_csv(original_weathered, predictions_df, filename):
    """
    将风化后实测值与风化前预测值对比，并保存到CSV文件。
    """
    print("--- 步骤 3: 保存预测结果到CSV文件 ---")
    if predictions_df is None or predictions_df.empty:
        print("没有预测数据可供保存。")
        return

    excluded_cols = ['文物采样点', '类型', '表面风化', 'sum']
    chemical_cols = [col for col in predictions_df.columns if col not in excluded_cols and predictions_df[col].dtype in ['float64', 'int64']]

    original_renamed = original_weathered.rename(columns={col: f'{col}_风化后(%)' for col in chemical_cols + ['sum']})
    predicted_renamed = predictions_df.rename(columns={col: f'{col}_预测风化前(%)' for col in chemical_cols + ['sum']})

    comparison_df = pd.concat([
        original_renamed[['文物采样点', '类型']].reset_index(drop=True),
        original_renamed[[f'{col}_风化后(%)' for col in chemical_cols + ['sum']]].reset_index(drop=True),
        predicted_renamed[[f'{col}_预测风化前(%)' for col in chemical_cols + ['sum']]].reset_index(drop=True)
    ], axis=1)
    
    final_cols = ['文物采样点', '类型']
    for col in chemical_cols:
        final_cols.append(f'{col}_风化后(%)')
        final_cols.append(f'{col}_预测风化前(%)')
    final_cols.append('sum_风化后(%)')
    final_cols.append('sum_预测风化前(%)')
    
    comparison_df = comparison_df[final_cols]

    try:
        comparison_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"详细预测对比数据已成功保存到 '{os.path.basename(filename)}'")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

def generate_prediction_report(predictions_df, csv_filename):
    """
    生成一个Markdown格式的简要报告。
    """
    print("--- 步骤 4: 生成Markdown格式的预测报告 ---")

    if predictions_df is None or predictions_df.empty:
        return "# 风化前成分预测报告\n\n数据集中没有找到需要预测的风化样本。\n"

    report = "# 风化前化学成分预测报告 (基于CLR变换)\n\n"
    report += "本报告根据风化样本的实测化学成分含量，并依据 `高钾分析结果.csv` 和 `钡铅分析结果.csv` 文件中基于**CLR变换**的统计规律，对其风化前的成分含量进行预测。\n\n"
    report += "**预测方法**：\n"
    report += "1.  对风化样本的化学成分进行中心对数比（CLR）变换。\n"
    report += "2.  在CLR变换空间中，根据统计规律（风化前后均值之差`Delta`），应用公式 `预测CLR值 = 实测CLR值 - Delta` 进行预测。\n"
    report += "3.  对无显著变化的成分，其CLR值保持不变。\n"
    report += "4.  最后，通过逆CLR变换将预测的CLR值还原为常规的百分比含量。\n\n---\n\n"
    report += f"**分析概要**：\n\n"
    report += f"本次分析对 **{len(predictions_df)}** 个风化样本进行了风化前成分预测。\n\n"
    report += f"详细的逐项预测数据（包含风化前后对比）已导出至CSV文件： **`{os.path.basename(csv_filename)}`**。\n\n"
    report += "可使用该CSV文件进行后续的数据可视化与定量分析。\n"
    
    print("--- 报告生成完成 ---\n")
    return report

# --- Main Execution Block ---

def main():
    """
    主函数，执行整个预测流程。
    """
    # 0. 定义输入和输出目录
    output_dir = '/Users/Mac/Downloads/22C/1/1_3'
    input_analysis_dir = '/Users/Mac/Downloads/22C/1/1_2'
    input_data_dir = '/Users/Mac/Downloads/22C/1/1_2'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 定义规律文件和主数据文件
    analysis_files = {
        '高钾': os.path.join(input_analysis_dir, '高钾分析结果.csv'),
        '铅钡': os.path.join(input_analysis_dir, '钡铅分析结果.csv')
    }
    main_data_file = os.path.join(input_data_dir, '附件2_处理后.csv')

    # --- Data Loading for Prediction Target ---
    try:
        df_main = pd.read_csv(main_data_file, encoding='utf-8')
        print(f"成功从 '{os.path.basename(main_data_file)}' 加载数据。共 {len(df_main)} 个采样点。\n")
    except Exception as e:
        print(f"主数据文件加载失败: {e}")
        return

    # --- Data Cleaning and Preparation ---
    chemical_cols = [col for col in df_main.columns if '氧化' in col or '二氧化' in col or '五氧化' in col]
    df_main[chemical_cols] = df_main[chemical_cols].fillna(0)
    df_main['sum'] = df_main[chemical_cols].sum(axis=1)
    df_filtered = df_main[(df_main['sum'] >= 85) & (df_main['sum'] <= 105)].copy()

    # --- Execute Analysis and Prediction ---
    # 步骤1: 从CSV加载规律并计算Delta值
    change_deltas = load_deltas_from_csv(analysis_files)
    
    # 步骤2: 进行预测
    predictions = predict_pre_weathering_composition(df_filtered, change_deltas)
    
    # 步骤3: 保存预测结果到CSV
    original_weathered_samples = df_filtered[df_filtered['表面风化'] == '风化'].copy()
    
    csv_report_filename = os.path.join(output_dir, "prediction_results.csv")
    if predictions is not None:
        save_predictions_to_csv(original_weathered_samples, predictions, csv_report_filename)
    
    # 步骤4: 生成Markdown简报
    markdown_report = generate_prediction_report(predictions, csv_report_filename)

    report_filename = os.path.join(output_dir, "prediction_report_final.md")
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        print(f"\n最终分析报告已成功生成并保存为 '{os.path.basename(report_filename)}'")
    except Exception as e:
        print(f"保存Markdown报告失败: {e}")

if __name__ == '__main__':
    main()
