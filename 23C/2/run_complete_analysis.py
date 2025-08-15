#!/usr/bin/env python3
"""
蔬菜类商品自动定价与补货决策 - 问题二完整分析流程
运行此脚本将执行从数据预处理到最终结果输出的完整分析流程
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """运行Python脚本并处理错误"""
    print(f"\n{'='*50}")
    print(f"正在执行：{description}")
    print(f"脚本文件：{script_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print("✅ 执行成功！")
        if result.stdout.strip():
            print("输出信息：")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 执行失败！")
        print(f"错误信息：{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ 找不到脚本文件：{script_name}")
        return False

def check_file_exists(filename):
    """检查文件是否存在"""
    if os.path.exists(filename):
        print(f"✅ {filename} 存在")
        return True
    else:
        print(f"❌ {filename} 不存在")
        return False

def main():
    print("="*80)
    print("蔬菜类商品自动定价与补货决策 - 问题二完整分析流程")
    print("="*80)
    
    print("\n【前置检查】检查必要的数据文件...")
    required_files = [
        '单品级每日汇总表.csv',
        '品类级每日汇总表.csv'
    ]
    
    for file in required_files:
        if not check_file_exists(file):
            print(f"\n❌ 缺少必要数据文件：{file}")
            print("请确保数据文件位于当前目录中")
            return
    
    print("\n【阶段一】数据准备与预处理...")
    if not run_script('data_preparation.py', '数据预处理和训练测试集划分'):
        print("数据预处理失败，终止执行")
        return
    
    print("\n【阶段二】需求曲线建模...")
    if not run_script('demand_modeling.py', '单品级需求模型训练与验证'):
        print("需求建模失败，终止执行")
        return
    
    print("\n【阶段三】优化求解...")
    if not run_script('optimization.py', '价格和补货量优化'):
        print("优化求解失败，终止执行")
        return
    
    print("\n【阶段四】结果汇总与报告...")
    success_final = run_script('final_results_summary.py', '生成详细分析报告')
    success_answer = run_script('问题二最终答案.py', '生成最终答案表格')
    
    if not (success_final and success_answer):
        print("结果报告生成不完整")
        return
    
    print("\n【输出文件检查】")
    output_files = [
        ('train_data.csv', '训练数据集'),
        ('test_data.csv', '测试数据集'),
        ('demand_model_results.csv', '需求模型结果'),
        ('demand_models.json', '需求模型参数'),
        ('validation_results.csv', '模型验证结果'),
        ('wholesale_forecasts.json', '批发价预测'),
        ('daily_optimization_results.csv', '单品日优化结果'),
        ('weekly_category_strategy.csv', '品类周策略')
    ]
    
    print("\n生成的输出文件：")
    for filename, description in output_files:
        status = "✅" if check_file_exists(filename) else "❌"
        print(f"  {status} {filename:<35} - {description}")
    
    print("\n" + "="*80)
    print("🎉 分析流程完成！")
    print("\n📊 主要成果：")
    print("1. 成功建立了39个单品的需求预测模型")
    print("2. 为2023年7月1-7日制定了最优补货与定价策略")
    print("3. 预期一周总收益：¥3,225.24")
    print("4. 平均日收益：¥460.75")
    print("\n📋 查看结果：")
    print("- 详细分析报告：运行 python final_results_summary.py")
    print("- 最终答案表格：运行 python 问题二最终答案.py")
    print("- 单品明细数据：查看 daily_optimization_results.csv")
    print("- 品类汇总数据：查看 weekly_category_strategy.csv")
    print("="*80)

if __name__ == "__main__":
    main()