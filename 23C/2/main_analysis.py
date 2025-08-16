#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蔬菜类商品自动定价与补货决策 - 主分析流程
整合的完整分析工作流，替代原有的多个独立脚本
"""

import os
import sys
from vegetable_optimizer import VegetableOptimizer
from report_generator import ReportGenerator


def check_required_files():
    """检查必要的数据文件"""
    print("="*80)
    print("蔬菜类商品自动定价与补货决策 - 完整分析流程")
    print("="*80)
    
    print("\n【前置检查】检查必要的数据文件...")
    
    required_files = [
        '单品级每日汇总表.csv',
        '品类级每日汇总表.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 缺少必要数据文件：{', '.join(missing_files)}")
        print("请确保数据文件位于当前目录中")
        return False
    
    return True


def run_analysis():
    """运行完整分析流程"""
    print("\n【阶段一】核心分析引擎执行...")
    print("-" * 50)
    
    # 初始化优化器
    optimizer = VegetableOptimizer()
    
    # 运行完整分析
    if not optimizer.run_full_analysis():
        print("❌ 核心分析失败，终止执行")
        return False
    
    print("\n✅ 核心分析完成！")
    return True


def generate_reports():
    """生成所有报告"""
    print("\n【阶段二】报告生成...")
    print("-" * 50)
    
    # 初始化报告生成器
    generator = ReportGenerator()
    
    # 生成所有报告
    if not generator.generate_all_reports():
        print("❌ 报告生成失败")
        return False
    
    print("\n✅ 报告生成完成！")
    return True


def check_output_files():
    """检查输出文件"""
    print("\n【输出文件检查】")
    print("-" * 50)
    
    output_files = [
        ('demand_model_results.csv', '需求模型结果'),
        ('demand_models.json', '需求模型参数'),
        ('validation_results.csv', '模型验证结果'),
        ('wholesale_forecasts.json', '批发价预测'),
        ('daily_optimization_results.csv', '单品日优化结果'),
        ('weekly_category_strategy.csv', '品类周策略')
    ]
    
    print("\n生成的输出文件：")
    all_exist = True
    for filename, description in output_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename:<35} - {description}")
        else:
            print(f"  ❌ {filename:<35} - {description}")
            all_exist = False
    
    return all_exist


def print_summary():
    """打印分析结果摘要"""
    print("\n" + "="*80)
    print("🎉 分析流程完成！")
    print("\n📊 主要成果：")
    print("1. 成功建立了单品级需求预测模型")
    print("2. 为2023年7月1-7日制定了最优补货与定价策略")
    print("3. 生成了完整的分析报告和决策表格")
    
    # 尝试读取结果数据显示概要
    try:
        import pandas as pd
        category_results = pd.read_csv('weekly_category_strategy.csv')
        total_profit = category_results['品类总利润(元)'].sum()
        total_replenishment = category_results['品类补货总量(千克)'].sum()
        
        print(f"4. 预期一周总收益：¥{total_profit:.2f}")
        print(f"5. 一周总补货量：{total_replenishment:.1f} kg")
        print(f"6. 平均日收益：¥{total_profit/7:.2f}")
    except:
        print("4. 详细收益数据请查看生成的报告文件")
    
    print("\n📋 查看结果的方式：")
    print("- 运行此脚本已显示完整报告")
    print("- 单独生成报告：python report_generator.py")
    print("- 查看详细数据：")
    print("  * daily_optimization_results.csv - 单品明细数据")
    print("  * weekly_category_strategy.csv - 品类汇总数据")
    print("  * demand_model_results.csv - 模型训练结果")
    print("="*80)


def main():
    """主函数"""
    try:
        # 1. 检查必要文件
        if not check_required_files():
            return
        
        # 2. 运行核心分析
        if not run_analysis():
            return
        
        # 3. 生成报告
        if not generate_reports():
            return
        
        # 4. 检查输出文件
        if not check_output_files():
            print("\n⚠️ 部分输出文件未生成，请检查分析过程")
        
        # 5. 打印摘要
        print_summary()
        
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()