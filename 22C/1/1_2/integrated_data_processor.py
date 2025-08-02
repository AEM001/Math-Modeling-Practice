#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
integrated_data_processor.py

该脚本整合了数据处理的各个阶段，包括：
1. 从 '附件2.csv' 加载原始数据。
2. 根据 '附件1.csv' 和 '附件.csv' 中的信息，为数据添加"类型"和"表面风化"列。
3. 填充所有缺失值（NaN）为0。
4. 将处理后的数据保存为 '附件2_处理后.csv'。

本脚本旨在一次性完成从原始数据到可用于后续分析的清理和增强数据的整个过程。
"""

import pandas as pd
import re
import os

# --- 用户配置 ---
# 输入文件
FILE_ATTACHMENT1_PATH = '附件1.csv'
FILE_ATTACHMENT2_ORIGINAL_PATH = '附件2.csv' # Assuming this is the initial target file
FILE_ATTACHMENT_REF_PATH = '附件.csv'

# 输出文件
OUTPUT_PROCESSED_FILE_PATH = '附件2_处理后.csv'

# --- 列名配置 ---
# 附件1.csv 中的列名
ID_COL_F1 = '文物编号'
TYPE_COL_F1 = '类型'

# 附件.csv 中的列名
ID_COL_REF = '文物编号'
WEATHERING_COL_REF = '表面风化'

# 目标文件（附件2.csv及其后续处理版本）中的列名
NAME_COL_TARGET = '文物采样点'      # 包含 '06部位1' 这种格式的列
TARGET_TYPE_COL = '类型'             # 需要被填充的目标列
TARGET_WEATHERING_COL = '表面风化'   # 需要被填充的目标列

def integrated_data_processing():
    """
    主处理函数：整合读取、匹配数据、填充缺失值并生成更新后的文件。
    """
    print("--- 启动数据整合处理流程 ---")

    # 1. 检查所有必需的输入文件是否存在
    required_files = [FILE_ATTACHMENT1_PATH, FILE_ATTACHMENT2_ORIGINAL_PATH, FILE_ATTACHMENT_REF_PATH]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"错误：找不到文件 '{f_path}'。请确保所有必需文件与脚本在同一目录下。")
            return

    try:
        # --- 步骤 1: 读取原始目标数据 ---
        print(f"正在读取原始目标文件: {FILE_ATTACHMENT2_ORIGINAL_PATH}...")
        df_final = pd.read_csv(FILE_ATTACHMENT2_ORIGINAL_PATH)
        print("成功加载原始目标数据。")

        # --- 步骤 2: 从附件1.csv 添加"类型"信息 ---
        print(f"正在读取参考文件: {FILE_ATTACHMENT1_PATH} 并添加\"类型\"信息...")
        df_ref_type = pd.read_csv(FILE_ATTACHMENT1_PATH, dtype={ID_COL_F1: str})
        df_ref_type[ID_COL_F1] = df_ref_type[ID_COL_F1].str.strip().str.zfill(2)
        type_mapping = pd.Series(df_ref_type[TYPE_COL_F1].values, index=df_ref_type[ID_COL_F1]).to_dict()

        def get_type_from_name(artifact_name):
            match = re.match(r'(\d{2})', str(artifact_name))
            if match:
                artifact_id = match.group(1)
                return type_mapping.get(artifact_id, '未在附件1中找到对应类型')
            else:
                return '名称中无有效编号'
        
        df_final[TARGET_TYPE_COL] = df_final[NAME_COL_TARGET].apply(get_type_from_name)
        print("\"类型\"信息已成功添加。")

        # --- 步骤 3: 从附件.csv 添加"表面风化"信息 ---
        print(f"正在读取参考文件: {FILE_ATTACHMENT_REF_PATH} 并添加\"表面风化\"信息...")
        df_ref_weathering = pd.read_csv(FILE_ATTACHMENT_REF_PATH, dtype={ID_COL_REF: str})
        df_ref_weathering[ID_COL_REF] = df_ref_weathering[ID_COL_REF].str.strip().str.zfill(2)
        weathering_mapping = pd.Series(df_ref_weathering[WEATHERING_COL_REF].values, index=df_ref_weathering[ID_COL_REF]).to_dict()

        def get_weathering_from_name(artifact_name):
            match = re.match(r'(\d{2})', str(artifact_name))
            if match:
                artifact_id = match.group(1)
                return weathering_mapping.get(artifact_id, '未在附件中找到对应风化信息')
            else:
                return '名称中无有效编号'

        df_final[TARGET_WEATHERING_COL] = df_final[NAME_COL_TARGET].apply(get_weathering_from_name)
        print("\"表面风化\"信息已成功添加。")

        # --- 步骤 4: 填充所有缺失值（NaN）为0 ---
        print("正在填充所有缺失值（NaN）为0...")
        df_final.fillna(0, inplace=True)
        print("缺失值填充完成。")

        # --- 步骤 5: 保存最终处理结果 ---
        print(f"正在保存最终处理后的数据到: {OUTPUT_PROCESSED_FILE_PATH}...")
        df_final.to_csv(OUTPUT_PROCESSED_FILE_PATH, index=False, encoding='utf-8-sig')
        print("-" * 30)
        print(f"🎉 数据整合处理完成！结果已成功保存到 '{OUTPUT_PROCESSED_FILE_PATH}'。")

    except Exception as e:
        print(f"处理过程中发生了一个错误: {e}")
        print("请检查CSV文件的格式和列名是否正确。")

# --- 脚本开始执行 ---
if __name__ == "__main__":
    integrated_data_processing() 