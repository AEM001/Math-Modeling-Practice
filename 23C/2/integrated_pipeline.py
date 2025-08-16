"""
蔬菜定价与补货策略集成管道
整合所有分析模块，提供统一的执行接口
"""

import os
import sys
import argparse
from datetime import datetime
import json

# 导入各个模块
from data_quality_audit import DataQualityAuditor
from exploratory_analysis import ExploratoryAnalyzer
from feature_engineering import FeatureEngineer
from enhanced_demand_modeling import EnhancedDemandModeler
from enhanced_optimizer import EnhancedOptimizer
from backtesting_validation import BacktestingValidator
from comprehensive_report_generator import ComprehensiveReportGenerator

class VegetablePricingPipeline:
    """蔬菜定价与补货策略集成管道"""
    
    def __init__(self, config_file=None):
        self.config = self.load_config(config_file)
        self.results = {}
        self.execution_log = []
        
    def load_config(self, config_file):
        """加载配置文件"""
        default_config = {
            "data_files": {
                "train_data": "train_data.csv",
                "test_data": "test_data.csv",
                "category_data": "品类级每日汇总表.csv",
                "item_data": "单品级每日汇总表.csv"
            },
            "modules": {
                "data_audit": True,
                "exploratory_analysis": True,
                "feature_engineering": True,
                "demand_modeling": True,
                "optimization": True,
                "backtesting": True,
                "reporting": True
            },
            "audit_config": {
                "outlier_threshold": 3.0,
                "price_multiplier_threshold": 3.0,
                "min_sales_threshold": 0.1
            },
            "modeling_config": {
                "test_size": 0.2,
                "cv_folds": 5,
                "random_state": 42
            },
            "optimization_config": {
                "optimization_horizon": 7,
                "service_level": 0.8,
                "min_markup_ratio": 1.1,
                "max_markup_ratio": 2.0,
                "max_daily_price_change": 0.1
            },
            "backtest_config": {
                "min_train_days": 30,
                "test_days": 7,
                "step_days": 7,
                "max_splits": 8
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 合并配置
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
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
        if not self.config["modules"]["data_audit"]:
            return True
            
        try:
            start_time = datetime.now()
            auditor = DataQualityAuditor(self.config["audit_config"])
            success = auditor.run_full_audit()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["data_audit"] = "completed"
                self.log_execution("数据质量审计", "success", "数据清洗完成", duration)
                return True
            else:
                self.log_execution("数据质量审计", "error", "数据审计失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("数据质量审计", "error", f"执行异常: {str(e)}")
            return False
    
    def run_exploratory_analysis(self):
        """运行探索性分析"""
        if not self.config["modules"]["exploratory_analysis"]:
            return True
            
        try:
            start_time = datetime.now()
            analyzer = ExploratoryAnalyzer()
            success = analyzer.run_full_analysis()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["exploratory_analysis"] = "completed"
                self.log_execution("探索性分析", "success", "EDA分析完成", duration)
                return True
            else:
                self.log_execution("探索性分析", "error", "EDA分析失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("探索性分析", "error", f"执行异常: {str(e)}")
            return False
    
    def run_feature_engineering(self):
        """运行特征工程"""
        if not self.config["modules"]["feature_engineering"]:
            return True
            
        try:
            start_time = datetime.now()
            engineer = FeatureEngineer()
            success = engineer.run_feature_engineering()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["feature_engineering"] = "completed"
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
        if not self.config["modules"]["demand_modeling"]:
            return True
            
        try:
            start_time = datetime.now()
            modeler = EnhancedDemandModeler(self.config["modeling_config"])
            success = modeler.run_enhanced_modeling()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["demand_modeling"] = "completed"
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
        if not self.config["modules"]["optimization"]:
            return True
            
        try:
            start_time = datetime.now()
            optimizer = EnhancedOptimizer(self.config["optimization_config"])
            success = optimizer.run_optimization()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["optimization"] = "completed"
                self.log_execution("优化算法", "success", "策略优化完成", duration)
                return True
            else:
                self.log_execution("优化算法", "error", "优化算法失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("优化算法", "error", f"执行异常: {str(e)}")
            return False
    
    def run_backtesting(self):
        """运行回测验证"""
        if not self.config["modules"]["backtesting"]:
            return True
            
        try:
            start_time = datetime.now()
            validator = BacktestingValidator(self.config["backtest_config"])
            success = validator.run_full_backtest()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["backtesting"] = "completed"
                self.log_execution("回测验证", "success", "模型验证完成", duration)
                return True
            else:
                self.log_execution("回测验证", "error", "回测验证失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("回测验证", "error", f"执行异常: {str(e)}")
            return False
    
    def run_reporting(self):
        """运行报告生成"""
        if not self.config["modules"]["reporting"]:
            return True
            
        try:
            start_time = datetime.now()
            generator = ComprehensiveReportGenerator()
            success = generator.run_comprehensive_reporting()
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self.results["reporting"] = "completed"
                self.log_execution("报告生成", "success", "综合报告完成", duration)
                return True
            else:
                self.log_execution("报告生成", "error", "报告生成失败", duration)
                return False
                
        except Exception as e:
            self.log_execution("报告生成", "error", f"执行异常: {str(e)}")
            return False
    
    def run_full_pipeline(self):
        """运行完整管道"""
        print("=" * 60)
        print("蔬菜定价与补货策略分析管道启动")
        print("=" * 60)
        
        pipeline_start = datetime.now()
        
        # 执行各个模块
        modules = [
            ("数据质量审计", self.run_data_audit),
            ("探索性分析", self.run_exploratory_analysis),
            ("特征工程", self.run_feature_engineering),
            ("需求建模", self.run_demand_modeling),
            ("优化算法", self.run_optimization),
            ("回测验证", self.run_backtesting),
            ("报告生成", self.run_reporting)
        ]
        
        success_count = 0
        for module_name, module_func in modules:
            if module_func():
                success_count += 1
            else:
                print(f"⚠ 警告: {module_name}执行失败，但管道将继续运行")
        
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        
        # 生成执行摘要
        print("\n" + "=" * 60)
        print("管道执行摘要")
        print("=" * 60)
        print(f"总执行时间: {pipeline_duration:.2f} 秒")
        print(f"成功模块数: {success_count}/{len(modules)}")
        print(f"成功率: {success_count/len(modules):.1%}")
        
        # 保存执行日志
        self.save_execution_log()
        
        # 显示生成的文件
        self.show_output_files()
        
        return success_count == len(modules)
    
    def save_execution_log(self):
        """保存执行日志"""
        log_file = f"pipeline_execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        execution_summary = {
            "pipeline_config": self.config,
            "execution_results": self.results,
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
            "data_quality_audit_report.md",
            "exploratory_analysis_report.md", 
            "feature_engineering_report.md",
            "enhanced_modeling_report.md",
            "enhanced_optimization_report.md",
            "backtest_validation_report.md",
            "comprehensive_analysis_report.md",
            "enhanced_weekly_category_strategy.csv",
            "comprehensive_performance_analysis.png",
            "category_strategy_visualization.png"
        ]
        
        print("\n生成的输出文件:")
        for file_name in output_files:
            if os.path.exists(file_name):
                file_size = os.path.getsize(file_name)
                print(f"✓ {file_name} ({file_size} bytes)")
            else:
                print(f"✗ {file_name} (未生成)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='蔬菜定价与补货策略分析管道')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--modules', '-m', nargs='+', 
                       choices=['audit', 'eda', 'features', 'modeling', 'optimization', 'backtest', 'report'],
                       help='指定要运行的模块')
    
    args = parser.parse_args()
    
    # 创建管道实例
    pipeline = VegetablePricingPipeline(args.config)
    
    # 如果指定了特定模块，只运行这些模块
    if args.modules:
        module_mapping = {
            'audit': 'data_audit',
            'eda': 'exploratory_analysis', 
            'features': 'feature_engineering',
            'modeling': 'demand_modeling',
            'optimization': 'optimization',
            'backtest': 'backtesting',
            'report': 'reporting'
        }
        
        # 禁用未指定的模块
        for key in pipeline.config["modules"]:
            pipeline.config["modules"][key] = False
        
        # 启用指定的模块
        for module in args.modules:
            if module in module_mapping:
                pipeline.config["modules"][module_mapping[module]] = True
    
    # 运行管道
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n🎉 管道执行成功完成！")
        sys.exit(0)
    else:
        print("\n⚠ 管道执行完成，但存在部分失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
