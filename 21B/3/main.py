#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4烯烃收率优化主程序

本程序实现了完整的C4烯烃收率优化流程：
1. 数据加载与预处理
2. 响应面模型构建（使用RidgeCV正则化回归）
3. 无约束优化求解
4. 约束优化求解
5. 稳健性分析

作者：数学建模团队
日期：2024年
"""

import os
import sys
import pandas as pd
import numpy as np

# 导入自定义模块
from data_processor import load_and_prepare_data, get_discrete_options, get_continuous_bounds
from model_builder import build_and_train_rsm
from optimizer import find_optimal_conditions, analyze_optimization_results
from robustness_analyzer import analyze_robustness_and_report

def main():
    """
    主函数：执行完整的C4烯烃收率优化分析
    """
    
    print("="*80)
    print("C4烯烃收率优化分析系统")
    print("="*80)
    print("基于响应面模型（RSM）的工艺条件优化")
    print("使用RidgeCV正则化回归提升模型泛化能力")
    print("特征工程：总质量和装料比")
    print("="*80)
    
    # 1. 定义文件路径和参数
    print("\n第1步：项目设置与环境准备")
    print("-" * 50)
    
    # 文件路径
    PATH_ATTACHMENT1 = '附件1.csv'
    PATH_INDICATORS = '每组指标.csv'
    
    # 检查文件是否存在
    if not os.path.exists(PATH_ATTACHMENT1):
        print(f"错误：找不到文件 {PATH_ATTACHMENT1}")
        return
    
    if not os.path.exists(PATH_INDICATORS):
        print(f"错误：找不到文件 {PATH_INDICATORS}")
        return
    
    print(f"✓ 附件1数据文件: {PATH_ATTACHMENT1}")
    print(f"✓ 每组指标文件: {PATH_INDICATORS}")
    
    # 2. 数据加载与预处理
    print("\n第2步：数据加载与预处理")
    print("-" * 50)
    
    try:
        prepared_data = load_and_prepare_data(PATH_ATTACHMENT1, PATH_INDICATORS)
        
        # 获取离散变量选项和连续变量边界
        discrete_options = get_discrete_options(prepared_data)
        continuous_bounds = get_continuous_bounds(prepared_data)
        
        print(f"✓ 数据预处理完成")
        print(f"✓ 连续变量边界:")
        for var, bounds in continuous_bounds.items():
            print(f"  {var}: {bounds[0]:.1f} - {bounds[1]:.1f}")
        print(f"✓ 离散变量选项:")
        for var, options in discrete_options.items():
            print(f"  {var}: {options}")
            
    except Exception as e:
        print(f"错误：数据预处理失败 - {e}")
        return
    
    # 3. 模型构建
    print("\n第3步：特征工程与模型构建")
    print("-" * 50)
    
    try:
        model, poly_features, scaler = build_and_train_rsm(prepared_data)
        print(f"✓ 响应面模型构建完成")
        
    except Exception as e:
        print(f"错误：模型构建失败 - {e}")
        return
    
    # 4. 无约束优化
    print("\n第4步：无约束优化求解")
    print("-" * 50)
    
    try:
        print("正在执行无约束优化...")
        unconstrained_solution = find_optimal_conditions(
            model, poly_features, scaler,
            continuous_bounds, discrete_options
        )
        
        # 分析无约束优化结果
        analyze_optimization_results(unconstrained_solution, discrete_options)
        
        print(f"✓ 无约束优化完成")
        
    except Exception as e:
        print(f"错误：无约束优化失败 - {e}")
        return
    
    # 5. 约束优化
    print("\n第5步：约束优化求解")
    print("-" * 50)
    
    try:
        # 设置温度约束：T < 350°C
        constrained_continuous_bounds = continuous_bounds.copy()
        constrained_continuous_bounds['T'] = (continuous_bounds['T'][0], 350)
        print(f"正在执行约束优化（温度范围：{constrained_continuous_bounds['T'][0]}-{constrained_continuous_bounds['T'][1]}°C）...")
        
        constrained_solution = find_optimal_conditions(
            model, poly_features, scaler,
            constrained_continuous_bounds, discrete_options
        )
        
        # 分析约束优化结果
        analyze_optimization_results(constrained_solution, discrete_options)
        
        print(f"✓ 约束优化完成")
        
    except Exception as e:
        print(f"错误：约束优化失败 - {e}")
        return
    
    # 6. 稳健性分析
    print("\n第6步：稳健性分析")
    print("-" * 50)
    
    try:
        print("正在分析无约束最优解的稳健性...")
        analyze_robustness_and_report(
            unconstrained_solution, model, poly_features, scaler, 
            discrete_options
        )
        
        print("\n正在分析约束最优解的稳健性...")
        analyze_robustness_and_report(
            constrained_solution, model, poly_features, scaler, 
            discrete_options
        )
        
        print(f"✓ 稳健性分析完成")
        
    except Exception as e:
        print(f"错误：稳健性分析失败 - {e}")
        return
    
    # 7. 结果总结
    print("\n" + "="*80)
    print("分析结果总结")
    print("="*80)
    
    # 比较两种优化结果
    unconstrained_yield = unconstrained_solution['max_yield']
    constrained_yield = constrained_solution['max_yield']
    yield_decrease = unconstrained_yield - constrained_yield
    yield_decrease_percent = (yield_decrease / unconstrained_yield) * 100
    
    print(f"\n优化结果对比:")
    print(f"无约束最优收率: {unconstrained_yield:.4f}")
    print(f"约束最优收率: {constrained_yield:.4f}")
    print(f"收率下降: {yield_decrease:.4f} ({yield_decrease_percent:.2f}%)")
    
    # 给出建议
    print(f"\n工艺优化建议:")
    if yield_decrease_percent < 5:
        print("✅ 温度约束对收率影响较小，建议采用约束优化方案")
        print("   理由：在保证安全的前提下，仍能获得较高的收率")
    elif yield_decrease_percent < 15:
        print("⚠️  温度约束对收率有一定影响，需要权衡安全性和收率")
        print("   建议：在确保安全的前提下，可考虑适当放宽温度约束")
    else:
        print("❌ 温度约束对收率影响较大，建议重新评估温度约束的必要性")
        print("   建议：考虑采用其他安全措施，而不是严格限制温度")
    
    print(f"\n最终推荐方案:")
    if yield_decrease_percent < 10:
        print("推荐采用约束优化方案（T < 350°C）")
        best_solution = constrained_solution
    else:
        print("推荐采用无约束优化方案")
        best_solution = unconstrained_solution
    
    best_params = best_solution['best_params']
    print(f"\n推荐工艺条件:")
    print(f"温度: {best_params['T']:.2f}°C")
    print(f"总质量: {best_params['total_mass']:.2f} mg")
    print(f"装料比: {best_params['loading_ratio']:.2f}")
    print(f"Co负载量: {best_params['C']} wt%") 
    print(f"乙醇浓度: {best_params['C_e']} ml/min")
    print(f"装料方式: {'B系列' if best_params['M'] == 1 else 'A系列'}")
    print(f"预测C4烯烃收率: {best_solution['max_yield']:.4f}")
    
    print(f"\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()