#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4烯烃收率优化主程序 (按装料方式分别建模)

本程序实现了完整的C4烯烃收率优化流程:
1. 数据加载与预处理
2. 按A,B两种装料方式分离数据
3. 为每种装料方式独立构建响应面模型
4. 为每种装料方式独立进行无约束和有约束优化
5. 为每个最优解进行稳健性分析
6. 对比并总结最终结果
7. 所有输出均保存到 analysis_log.txt
"""

import os
import sys
import pandas as pd
import numpy as np

# 导入自定义模块
from data_processor import load_and_prepare_data, get_continuous_bounds
from model_builder import build_and_train_rsm
from optimizer import find_optimal_conditions, analyze_optimization_results
from robustness_analyzer import analyze_robustness_and_report

class Logger:
    """将stdout重定向到文件和控制台"""
    def __init__(self, filename="analysis_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_analysis_for_method(method_name, method_data):
    """
    为单一装料方式执行完整的分析流程

    Args:
        method_name (str): 装料方式名称 ('A系列' 或 'B系列')
        method_data (pd.DataFrame): 对应装料方式的数据

    Returns:
        dict: 包含该方法下所有分析结果的字典
    """
    print("\n" + "="*80)
    print(f"开始分析装料方式: {method_name}")
    print("="*80)

    # 1. 获取该方法数据的变量边界
    continuous_bounds = get_continuous_bounds(method_data)
    print("\n第1步: 获取变量边界")
    print("-" * 50)
    print(f"✓ 连续变量边界:")
    for var, bounds in continuous_bounds.items():
        print(f"  {var}: {bounds[0]:.2f} - {bounds[1]:.2f}")

    # 2. 模型构建
    print("\n第2步: 特征工程与模型构建")
    print("-" * 50)
    try:
        # 移除不再需要的列
        model_data = method_data.drop(columns=['catalyst_id', 'M'])
        model, poly_features, scaler = build_and_train_rsm(model_data, method_name)
        print(f"✓ {method_name}的响应面模型构建完成")
    except Exception as e:
        print(f"错误: {method_name}模型构建失败 - {e}")
        return None

    # 3. 无约束优化
    print("\n第3步: 无约束优化求解")
    print("-" * 50)
    try:
        print("正在执行无约束优化...")
        unconstrained_solution = find_optimal_conditions(
            model, poly_features, scaler, continuous_bounds
        )
        analyze_optimization_results(unconstrained_solution)
        print(f"✓ {method_name}无约束优化完成")
    except Exception as e:
        print(f"错误: {method_name}无约束优化失败 - {e}")
        unconstrained_solution = None

    # 4. 约束优化 (T < 350°C)
    print("\n第4步: 约束优化求解 (T < 350°C)")
    print("-" * 50)
    try:
        constrained_continuous_bounds = continuous_bounds.copy()
        constrained_continuous_bounds['T'] = (continuous_bounds['T'][0], 350)
        print(f"正在执行约束优化(温度范围: {constrained_continuous_bounds['T'][0]:.1f}-{constrained_continuous_bounds['T'][1]:.1f}°C)...")
        constrained_solution = find_optimal_conditions(
            model, poly_features, scaler, constrained_continuous_bounds
        )
        analyze_optimization_results(constrained_solution)
        print(f"✓ {method_name}约束优化完成")
    except Exception as e:
        print(f"错误: {method_name}约束优化失败 - {e}")
        constrained_solution = None

    # 5. 稳健性分析
    print("\n第5步: 稳健性分析")
    print("-" * 50)
    if unconstrained_solution and unconstrained_solution['best_params']:
        print(f"--- 正在分析 {method_name} 无约束最优解的稳健性 ---")
        analyze_robustness_and_report(unconstrained_solution, model, poly_features, scaler)
    if constrained_solution and constrained_solution['best_params']:
        print(f"--- 正在分析 {method_name} 约束最优解的稳健性 ---")
        analyze_robustness_and_report(constrained_solution, model, poly_features, scaler)

    return {
        'unconstrained': unconstrained_solution,
        'constrained': constrained_solution
    }

def main():
    """
    主函数: 执行完整的C4烯烃收率优化分析
    """
    # 重定向所有print输出到日志文件和控制台
    sys.stdout = Logger("analysis_log.txt")

    print("="*80)
    print("C4烯烃收率优化分析系统 (按装料方式分别建模)")
    print("="*80)

    # 定义文件路径
    PATH_ATTACHMENT1 = '附件1.csv'
    PATH_INDICATORS = '每组指标.csv'
    
    if not (os.path.exists(PATH_ATTACHMENT1) and os.path.exists(PATH_INDICATORS)):
        print(f"错误: 找不到数据文件 {PATH_ATTACHMENT1} 或 {PATH_INDICATORS}")
        return

    # 加载并预处理所有数据
    print("正在加载和预处理所有数据...")
    try:
        prepared_data = load_and_prepare_data(PATH_ATTACHMENT1, PATH_INDICATORS)
        print("✓ 数据加载完成")
    except Exception as e:
        print(f"错误: 数据预处理失败 - {e}")
        return

    # 分离不同装料方式的数据
    data_a = prepared_data[prepared_data['M'] == 0].copy()
    data_b = prepared_data[prepared_data['M'] == 1].copy()
    print(f"A系列数据: {len(data_a)} 行")
    print(f"B系列数据: {len(data_b)} 行")

    # 分别对两种方式进行分析
    results_a = run_analysis_for_method('A系列', data_a)
    results_b = run_analysis_for_method('B系列', data_b)

    # 最终结果对比
    print("\n" + "="*80)
    print("最终结果对比分析")
    print("="*80)

    def print_summary(name, result):
        if result and result['best_params']:
            params = result['best_params']
            yield_val = result['max_yield']
            print(f"  - {name}:")
            print(f"    - 预测收率: {yield_val:.2f}%")
            print(f"    - 工艺条件: T={params['T']:.1f}°C, 总质量={params['total_mass']:.1f}mg, "
                  f"装料比={params['loading_ratio']:.2f}, Co负载量={params['C']:.2f}wt%, "
                  f"乙醇浓度={params['C_e']:.2f}ml/min")
        else:
            print(f"  - {name}: 未能找到有效解。")

    print("\n--- 无约束优化对比 (追求最高收率) ---")
    print_summary("A系列", results_a['unconstrained'] if results_a else None)
    print_summary("B系列", results_b['unconstrained'] if results_b else None)

    print("\n--- 约束优化对比 (T < 350°C) ---")
    print_summary("A系列 (T<350°C)", results_a['constrained'] if results_a else None)
    print_summary("B系列 (T<350°C)", results_b['constrained'] if results_b else None)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
