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
            success = modeler.run_modeling()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.log_execution("需求建模", "success", "模型训练完成", duration)
                return True
            else:
                self.log_execution("需求建模", "error", "需求建模失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("需求建模", "error", f"执行异常: {str(e)}")
            return False
    
    def run_optimization(self):
        """运行优化算法"""
        try:
            start_time = datetime.now()
            optimizer = VegetableOptimizer(self.config_path)
            success = optimizer.run_optimization()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.log_execution("优化算法", "success", "策略优化完成", duration)
                return True
            else:
                self.log_execution("优化算法", "error", "优化算法失败", duration)
                return False
                
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
