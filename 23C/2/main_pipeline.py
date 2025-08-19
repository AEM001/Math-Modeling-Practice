#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蔬菜定价与补货策略分析管道 - 精简版
整合所有分析模块，提供统一的执行接口
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_auditor import DataAuditor
from feature_engineer import FeatureEngineer
from demand_modeler import DemandModeler
from optimizer import VegetableOptimizer
from visualizer import Visualizer

class VegetablePricingPipeline:
    """蔬菜定价与补货策略分析管道"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化管道"""
        self.config_path = config_path
        self.config = self.load_config()
        self.execution_log = []
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    def log_execution(self, module_name, status, message="", duration=0):
        """记录执行日志"""
        log_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "module": module_name,
            "status": status,
            "message": message,
            "duration_seconds": duration
        }
        self.execution_log.append(log_entry)
        
        status_icon = "✓" if status == "success" else "✗" if status == "error" else "⚠"
        print(f"{status_icon} {module_name}: {message}")
    
    def run_data_audit(self):
        """运行数据质量审计"""
        try:
            start_time = datetime.now()
            auditor = DataAuditor(self.config_path)
            success = auditor.run_full_audit()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.log_execution("数据质量审计", "success", "数据清洗完成", duration)
                return True
            else:
                self.log_execution("数据质量审计", "error", "数据审计失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("数据质量审计", "error", f"执行异常: {str(e)}")
            return False
    
    def run_feature_engineering(self):
        """运行特征工程"""
        try:
            start_time = datetime.now()
            engineer = FeatureEngineer(self.config_path)
            success = engineer.run_feature_engineering()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.log_execution("特征工程", "success", "特征构建完成", duration)
                return True
            else:
                self.log_execution("特征工程", "error", "特征工程失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("特征工程", "error", f"执行异常: {str(e)}")
            return False
    
    def run_demand_modeling(self):
        """运行需求建模"""
        try:
            start_time = datetime.now()
            modeler = DemandModeler(self.config_path)
            
            # run_modeling方法没有返回值，我们检查结果文件是否生成
            modeler.run_modeling()
            
            # 检查是否生成了结果文件
            results_path = os.path.join(modeler.output_paths['results_dir'], 'demand_model_results.csv')
            
            if os.path.exists(results_path):
                # 检查文件是否为空
                import pandas as pd
                try:
                    results_df = pd.read_csv(results_path)
                    
                    # 添加is_best列（所有模型都标记为最佳，因为每个品类只有一个模型）
                    results_df['is_best'] = True
                    results_df.to_csv(results_path, index=False, encoding='utf-8')
                    
                    model_count = len(results_df)
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    self.log_execution("需求建模", "success", f"模型训练完成，生成了{model_count}个模型", duration)
                    return True
                except Exception as e:
                    self.log_execution("需求建模", "error", f"结果文件错误: {str(e)}")
                    return False
            else:
                duration = (datetime.now() - start_time).total_seconds()
                self.log_execution("需求建模", "error", "没有生成模型结果文件", duration)
                return False
                
        except Exception as e:
            self.log_execution("需求建模", "error", f"执行异常: {str(e)}")
            return False
    
    def run_optimization(self):
        """运行优化算法"""
        try:
            start_time = datetime.now()
            optimizer = VegetableOptimizer(self.config_path)
            
            # 加载需求模型
            optimizer.load_demand_models()
            
            if not optimizer.demand_models:
                self.log_execution("优化算法", "error", "没有加载到需求模型")
                return False
            
            # 运行优化
            results_df = optimizer.run_daily_optimization()
            
            if results_df.empty:
                self.log_execution("优化算法", "error", "优化结果为空")
                return False
            
            # 保存结果
            optimizer.save_optimization_results(results_df)
            
            # 生成周策略汇总
            weekly_summary = results_df.groupby('category').agg({
                'wholesale_cost': 'mean',
                'optimal_price': 'mean', 
                'optimal_quantity': 'mean',
                'expected_profit': 'sum'
            }).reset_index()
            
            weekly_summary.columns = ['category', 'avg_wholesale_cost', 'avg_optimal_price', 'avg_optimal_quantity', 'total_expected_profit']
            weekly_summary['profit_margin'] = (weekly_summary['avg_optimal_price'] - weekly_summary['avg_wholesale_cost']) / weekly_summary['avg_wholesale_cost']
            
            # 保存周策略
            import os
            # 保存到项目根目录下的标准输出路径（而不是相对config目录）
            project_root = os.path.abspath(os.path.dirname(__file__))
            results_dir = os.path.join(project_root, self.config['output_paths']['results_dir'])
            os.makedirs(results_dir, exist_ok=True)
            weekly_path = os.path.join(results_dir, 'weekly_strategy.csv')
            weekly_summary.to_csv(weekly_path, index=False, encoding='utf-8-sig')
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_execution("优化算法", "success", f"策略优化完成，生成了{len(results_df)}条优化结果", duration)
            return True
                
        except Exception as e:
            self.log_execution("优化算法", "error", f"执行异常: {str(e)}")
            return False
    
    def run_visualization(self):
        """运行可视化生成"""
        try:
            start_time = datetime.now()
            visualizer = Visualizer(self.config_path)
            results = visualizer.generate_all_visualizations()
            duration = (datetime.now() - start_time).total_seconds()
            
            # 检查是否有图表生成成功
            success_count = sum(1 for result in results.values() if result is not None)
            
            if success_count > 0:
                self.log_execution("可视化生成", "success", f"生成了{success_count}个图表", duration)
                return True
            else:
                self.log_execution("可视化生成", "error", "没有图表生成成功", duration)
                return False
                
        except Exception as e:
            self.log_execution("可视化生成", "error", f"执行异常: {str(e)}")
            return False
    
    def run_full_pipeline(self, modules=None):
        """运行完整管道"""
        print("=" * 60)
        print("蔬菜定价与补货策略分析管道启动")
        print("=" * 60)
        
        pipeline_start = datetime.now()
        
        # 定义执行模块
        all_modules = [
            ("数据质量审计", self.run_data_audit),
            ("特征工程", self.run_feature_engineering),
            ("需求建模", self.run_demand_modeling),
            ("优化算法", self.run_optimization),
            ("可视化生成", self.run_visualization)
        ]
        
        # 如果指定了特定模块，只运行这些模块
        if modules:
            module_map = {
                'audit': 0, 'features': 1, 'modeling': 2, 'optimization': 3, 'visualization': 4
            }
            selected_modules = []
            for module in modules:
                if module in module_map:
                    selected_modules.append(all_modules[module_map[module]])
            
            if selected_modules:
                all_modules = selected_modules
        
        success_count = 0
        for module_name, module_func in all_modules:
            print(f"\n开始执行: {module_name}")
            if module_func():
                success_count += 1
            else:
                print(f"⚠ 警告: {module_name}执行失败")
                # 对于关键模块，如果失败则停止
                if module_name in ["数据质量审计", "特征工程"]:
                    print(f"关键模块{module_name}失败，停止执行")
                    break
        
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        
        # 生成执行摘要
        print("\n" + "=" * 60)
        print("管道执行摘要")
        print("=" * 60)
        print(f"总执行时间: {pipeline_duration:.2f} 秒")
        print(f"成功模块数: {success_count}/{len(all_modules)}")
        print(f"成功率: {success_count/len(all_modules):.1%}")
        
        # 保存执行日志
        self.save_execution_log()
        
        # 显示生成的文件
        self.show_output_files()
        
        return success_count == len(all_modules)
    
    def save_execution_log(self):
        """保存执行日志"""
        log_file = f"pipeline_execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        execution_summary = {
            "pipeline_config": self.config,
            "execution_log": self.execution_log,
            "summary": {
                "total_modules": len(self.execution_log),
                "successful_modules": len([log for log in self.execution_log if log["status"] == "success"]),
                "failed_modules": len([log for log in self.execution_log if log["status"] == "error"]),
                "total_duration": sum([log["duration_seconds"] for log in self.execution_log])
            }
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(execution_summary, f, ensure_ascii=False, indent=2)
        
        print(f"执行日志已保存: {log_file}")
    
    def show_output_files(self):
        """显示输出文件"""
        output_files = [
            ("数据审计报告", "reports/data_audit_report.md"),
            ("特征工程报告", "reports/feature_engineering_report.md"),
            ("需求建模报告", "reports/demand_modeling_report.md"),
            ("优化策略报告", "reports/optimization_report.md"),
            ("建模结果", "outputs/results/demand_model_results.csv"),
            ("优化结果", "outputs/results/optimization_results.csv"),
            ("周策略汇总", "outputs/results/weekly_strategy.csv"),
            ("需求模型性能图", "outputs/figures/demand_model_performance.png"),
            ("优化结果图", "outputs/figures/optimization_results.png"),
            ("汇总仪表板", "outputs/figures/summary_dashboard.png")
        ]
        
        print("\n生成的输出文件:")
        for file_desc, file_path in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✓ {file_desc}: {file_path} ({file_size} bytes)")
            else:
                print(f"✗ {file_desc}: {file_path} (未生成)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='蔬菜定价与补货策略分析管道')
    parser.add_argument('--config', '-c', default='config/config.json', help='配置文件路径')
    parser.add_argument('--modules', '-m', nargs='+', 
                       choices=['audit', 'features', 'modeling', 'optimization', 'visualization'],
                       help='指定要运行的模块')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件 {args.config} 不存在")
        sys.exit(1)
    
    # 创建管道实例
    pipeline = VegetablePricingPipeline(args.config)
    
    # 运行管道
    success = pipeline.run_full_pipeline(args.modules)
    
    if success:
        print("\n🎉 管道执行成功完成！")
        sys.exit(0)
    else:
        print("\n⚠ 管道执行完成，但存在部分失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
