#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验设计主程序 - GPR实验设计系统
功能：整合所有模块，完成基于GPR+EI的实验设计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_processor import DataProcessor
from gpr_model import GPRModel
from candidate_generator import CandidateGenerator
from ei_optimizer import EIOptimizer

# 设置中文字体
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # Linux
plt.rcParams['axes.unicode_minus'] = False

class ExperimentDesigner:
    """实验设计器主类"""
    
    def __init__(self, random_state: int = 42):
        """初始化实验设计器"""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 初始化各模块
        self.data_processor = DataProcessor()
        self.gpr_model = GPRModel(random_state=random_state)
        self.candidate_generator = CandidateGenerator(random_state=random_state)
        self.ei_optimizer = EIOptimizer(xi=0.01)
        
        # 存储中间结果
        self.X_train = None
        self.y_train = None
        self.scaler = None
        self.variable_bounds = None
        self.candidates_scaled = None
        self.candidates_original = None
        self.gpr_fitted = None
        self.y_best = None
        
    def load_and_process_data(self, data_path: str = "附件1.csv", 
                            catalyst_path: str = "每组指标.csv") -> Tuple[np.ndarray, np.ndarray]:
        """
        加载和处理数据
        
        Args:
            data_path: 主数据文件路径
            catalyst_path: 催化剂数据文件路径
            
        Returns:
            X_train: 训练特征矩阵
            y_train: 训练目标向量
        """
        print("🔄 步骤1: 数据加载与预处理...")
        
        # 加载和预处理数据
        self.X_train, self.y_train, data_info = self.data_processor.load_and_prepare_data(data_path, catalyst_path)
        
        # 获取标准化器
        self.scaler = self.data_processor.scaler
        
        # 获取变量边界
        self.variable_bounds = self.data_processor.get_variable_bounds()
        
        # 记录当前最佳收率
        self.y_best = np.max(self.y_train)
        
        print(f"✅ 数据处理完成!")
        print(f"   📊 训练样本数: {len(self.X_train)}")
        print(f"   📋 特征维度: {self.X_train.shape[1]}")
        print(f"   🎯 当前最佳收率: {self.y_best:.4f}")
        
        return self.X_train, self.y_train
    
    def build_gpr_model(self) -> None:
        """构建和训练GPR模型"""
        print("\n🔄 步骤2: GPR模型构建与训练...")
        
        # 训练模型
        training_results = self.gpr_model.build_and_train(self.X_train, self.y_train)
        self.gpr_fitted = self.gpr_model
        
        # 获取验证结果
        validation_results = training_results['validation_results']
        
        print(f"✅ GPR模型训练完成!")
        print(f"   📈 交叉验证R²: {validation_results['cv_r2_mean']:.4f} ± {validation_results['cv_r2_std']:.4f}")
        print(f"   📉 交叉验证RMSE: {validation_results['cv_rmse_mean']:.4f} ± {validation_results['cv_rmse_std']:.4f}")
    
    def generate_candidate_points(self, n_candidates: int = 1000, 
                                use_grid: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成候选实验点
        
        Args:
            n_candidates: 候选点数量
            use_grid: 是否使用网格采样
            
        Returns:
            candidates_scaled: 标准化候选点
            candidates_original: 原始候选点
        """
        print(f"\n🔄 步骤3: 候选点生成...")
        
        if use_grid:
            # 网格采样
            self.candidates_scaled, self.candidates_original = \
                self.candidate_generator.generate_grid_candidates(
                    self.variable_bounds, self.scaler
                )
        else:
            # LHS采样
            self.candidates_scaled, self.candidates_original = \
                self.candidate_generator.generate_candidates(
                    self.variable_bounds, n_candidates, self.scaler
                )
        
        # 可行性过滤
        feasible_mask = self.candidate_generator.filter_feasible_candidates(
            self.candidates_original
        )
        
        self.candidates_scaled = self.candidates_scaled[feasible_mask]
        self.candidates_original = self.candidates_original[feasible_mask]
        
        print(f"✅ 候选点生成完成!")
        print(f"   📊 可行候选点数: {len(self.candidates_original)}")
        
        return self.candidates_scaled, self.candidates_original
    
    def optimize_experiments(self, n_experiments: int = 5, 
                           use_constraints: bool = True,
                           diversity_weight: float = 0.3) -> pd.DataFrame:
        """
        优化实验设计
        
        Args:
            n_experiments: 需要设计的实验数量
            use_constraints: 是否使用约束条件
            diversity_weight: 多样性权重
            
        Returns:
            results_df: 实验设计结果
        """
        print(f"\n🔄 步骤4: 实验设计优化...")
        
        # 计算EI值
        if use_constraints:
            constraints = {
                'temperature_stability': True,
                'co_loading_cost': True,
                'loading_ratio_stability': True,
                'combination_stability': True
            }
            ei_values = self.ei_optimizer.calculate_constrained_ei(
                self.candidates_scaled, self.gpr_fitted, self.y_best, constraints
            )
        else:
            ei_values = self.ei_optimizer.calculate_ei(
                self.candidates_scaled, self.gpr_fitted, self.y_best
            )
        
        # 选择最优实验点
        selected_indices, selected_points_scaled, selection_info = \
            self.ei_optimizer.select_optimal_experiments(
                self.candidates_scaled, ei_values, self.gpr_fitted,
                n_experiments, diversity_weight
            )
        
        # 转换回原始尺度
        selected_points_original = self.candidates_original[selected_indices]
        
        # 分析结果
        feature_names = ['T', 'total_mass', 'loading_ratio', 'Co_loading', 'ethanol_conc', 'loading_method']
        results_df = self.ei_optimizer.analyze_selection_results(
            selected_points_original, selection_info, feature_names
        )
        
        # 添加实验编号
        results_df.insert(0, 'experiment_id', [f'NEW_{i+1}' for i in range(len(results_df))])
        
        print(f"✅ 实验设计优化完成!")
        
        return results_df
    
    def generate_detailed_recommendations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """生成详细的实验建议"""
        print("\n🔄 步骤5: 生成详细实验建议...")
        
        # 创建详细建议DataFrame
        detailed_df = results_df.copy()
        
        # 转换装料方式
        detailed_df['loading_method_name'] = detailed_df['loading_method'].map({
            1: '装料方式I', 2: '装料方式II'
        })
        
        # 计算Co/SiO2和HAP的具体质量
        detailed_df['Co_SiO2_mass'] = detailed_df['total_mass'] * detailed_df['loading_ratio'] / (1 + detailed_df['loading_ratio'])
        detailed_df['HAP_mass'] = detailed_df['total_mass'] - detailed_df['Co_SiO2_mass']
        
        # 生成催化剂组合描述
        detailed_df['catalyst_description'] = detailed_df.apply(
            lambda row: f"{row['Co_SiO2_mass']:.1f}mg {row['Co_loading']}wt%Co/SiO2-{row['HAP_mass']:.1f}mg HAP-乙醇浓度 {row['ethanol_conc']:.2f}ml/min",
            axis=1
        )
        
        # 生成实验条件总结
        detailed_df['experiment_summary'] = detailed_df.apply(
            lambda row: f"温度{row['T']:.0f}°C, {row['catalyst_description']}, {row['loading_method_name']}",
            axis=1
        )
        
        # 预期收益分析
        detailed_df['expected_improvement'] = detailed_df['EI_value']
        detailed_df['improvement_potential'] = detailed_df.apply(
            lambda row: self._categorize_improvement_potential(row['EI_value'], row['predicted_yield']),
            axis=1
        )
        
        # 风险评估
        detailed_df['risk_level'] = detailed_df.apply(
            lambda row: self._assess_risk_level(row),
            axis=1
        )
        
        # 实验优先级
        detailed_df['priority'] = detailed_df.apply(
            lambda row: self._calculate_priority(row),
            axis=1
        )
        
        print("✅ 详细实验建议生成完成!")
        
        return detailed_df
    
    def _categorize_improvement_potential(self, ei_value: float, predicted_yield: float) -> str:
        """分类改进潜力"""
        if ei_value > 0.01 and predicted_yield > self.y_best:
            return "高潜力"
        elif ei_value > 0.005:
            return "中等潜力"
        else:
            return "探索性"
    
    def _assess_risk_level(self, row: pd.Series) -> str:
        """评估风险水平"""
        risk_score = 0
        
        # 温度风险
        if row['T'] > 400:
            risk_score += 2
        elif row['T'] < 275:
            risk_score += 1
        
        # Co负载量风险
        if row['Co_loading'] >= 5.0:
            risk_score += 2
        
        # 装料比风险
        if row['loading_ratio'] < 0.4 or row['loading_ratio'] > 1.8:
            risk_score += 1
        
        # 组合风险
        if row['T'] > 400 and row['Co_loading'] >= 5.0:
            risk_score += 2
        
        if risk_score >= 4:
            return "高风险"
        elif risk_score >= 2:
            return "中等风险"
        else:
            return "低风险"
    
    def _calculate_priority(self, row: pd.Series) -> str:
        """计算实验优先级"""
        if row['improvement_potential'] == "高潜力" and row['risk_level'] != "高风险":
            return "高优先级"
        elif row['improvement_potential'] == "中等潜力" or row['risk_level'] == "低风险":
            return "中等优先级"
        else:
            return "低优先级"
    
    def visualize_results(self, results_df: pd.DataFrame, save_plots: bool = True):
        """可视化结果"""
        print("\n🔄 步骤6: 结果可视化...")
        
        # 设置图形样式
        plt.style.use('default')
        # 确保中文字体正确显示
        plt.rcParams['font.size'] = 10
        fig = plt.figure(figsize=(20, 15))
        
        # 1. EI值分布
        ax1 = plt.subplot(2, 3, 1)
        plt.bar(range(len(results_df)), results_df['EI_value'], color='skyblue', alpha=0.7)
        plt.xlabel('实验编号')
        plt.ylabel('EI值')
        plt.title('期望改进值分布')
        plt.xticks(range(len(results_df)), results_df['experiment_id'], rotation=45)
        
        # 2. 预测收率vs不确定性
        ax2 = plt.subplot(2, 3, 2)
        scatter = plt.scatter(results_df['predicted_yield'], results_df['uncertainty'], 
                            c=results_df['EI_value'], cmap='viridis', s=100, alpha=0.7)
        plt.xlabel('预测收率')
        plt.ylabel('预测不确定性')
        plt.title('收率-不确定性关系')
        plt.colorbar(scatter, label='EI值')
        
        # 3. 温度vs Co负载量
        ax3 = plt.subplot(2, 3, 3)
        scatter = plt.scatter(results_df['T'], results_df['Co_loading'], 
                            c=results_df['predicted_yield'], cmap='RdYlBu_r', s=100, alpha=0.7)
        plt.xlabel('温度 (°C)')
        plt.ylabel('Co负载量 (wt%)')
        plt.title('温度-Co负载量分布')
        plt.colorbar(scatter, label='预测收率')
        
        # 4. 装料比分布
        ax4 = plt.subplot(2, 3, 4)
        plt.hist(results_df['loading_ratio'], bins=10, alpha=0.7, color='lightcoral')
        plt.xlabel('Co/SiO2和HAP装料比')
        plt.ylabel('频次')
        plt.title('装料比分布')
        
        # 5. 改进潜力饼图
        ax5 = plt.subplot(2, 3, 5)
        if 'improvement_potential' in results_df.columns:
            potential_counts = results_df['improvement_potential'].value_counts()
            plt.pie(potential_counts.values, labels=potential_counts.index, autopct='%1.1f%%')
            plt.title('改进潜力分布')
        
        # 6. 风险水平分布
        ax6 = plt.subplot(2, 3, 6)
        if 'risk_level' in results_df.columns:
            risk_counts = results_df['risk_level'].value_counts()
            colors = ['green', 'orange', 'red'][:len(risk_counts)]
            plt.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
            plt.xlabel('风险水平')
            plt.ylabel('实验数量')
            plt.title('风险水平分布')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('experiment_design_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 可视化图表已保存: experiment_design_analysis.png")
        
        plt.show()
        
        print("✅ 结果可视化完成!")
    
    def save_results(self, results_df: pd.DataFrame, detailed_df: pd.DataFrame = None):
        """保存结果"""
        print("\n🔄 步骤7: 保存结果...")
        
        # 保存基础结果
        results_df.to_csv('experiment_design_results.csv', index=False, encoding='utf-8-sig')
        print("💾 基础结果已保存: experiment_design_results.csv")
        
        # 保存详细建议
        if detailed_df is not None:
            detailed_df.to_csv('detailed_experiment_recommendations.csv', index=False, encoding='utf-8-sig')
            print("💾 详细建议已保存: detailed_experiment_recommendations.csv")
        
        # 生成实验报告
        self._generate_experiment_report(results_df, detailed_df)
        
        print("✅ 结果保存完成!")
    
    def _generate_experiment_report(self, results_df: pd.DataFrame, detailed_df: pd.DataFrame = None):
        """生成实验报告"""
        report_content = []
        report_content.append("# 乙醇偶合制备C4烯烃 - 实验设计报告\n")
        report_content.append("## 1. 实验设计概述\n")
        report_content.append(f"- **设计方法**: 基于GPR模型和EI准则的实验设计")
        report_content.append(f"- **实验数量**: {len(results_df)}个新实验")
        report_content.append(f"- **当前最佳收率**: {self.y_best:.4f}")
        report_content.append(f"- **预期最高收率**: {results_df['predicted_yield'].max():.4f}\n")
        
        report_content.append("## 2. 推荐实验条件\n")
        for i, row in results_df.iterrows():
            report_content.append(f"### 实验 {row['experiment_id']}")
            report_content.append(f"- **温度**: {row['T']:.0f}°C")
            report_content.append(f"- **Co负载量**: {row['Co_loading']:.1f}wt%")
            report_content.append(f"- **装料比**: {row['loading_ratio']:.2f}")
            report_content.append(f"- **乙醇浓度**: {row['ethanol_conc']:.2f}ml/min")
            report_content.append(f"- **装料方式**: {'I' if row['loading_method']==1 else 'II'}")
            report_content.append(f"- **预测收率**: {row['predicted_yield']:.4f} ± {row['uncertainty']:.4f}")
            report_content.append(f"- **EI值**: {row['EI_value']:.6f}")
            report_content.append(f"- **选择理由**: {row['selection_reason']}\n")
        
        if detailed_df is not None:
            report_content.append("## 3. 实验建议总结\n")
            report_content.append("| 实验ID | 优先级 | 改进潜力 | 风险水平 | 实验条件总结 |")
            report_content.append("|--------|--------|----------|----------|--------------|")
            for _, row in detailed_df.iterrows():
                report_content.append(f"| {row['experiment_id']} | {row['priority']} | {row['improvement_potential']} | {row['risk_level']} | {row['experiment_summary']} |")
        
        report_content.append("\n## 4. 实验执行建议\n")
        report_content.append("1. **优先执行高优先级实验**，这些实验具有最高的预期收益")
        report_content.append("2. **注意高风险实验的安全措施**，特别是高温和高Co负载量条件")
        report_content.append("3. **建议分批执行**，先执行2-3个实验验证模型预测准确性")
        report_content.append("4. **记录详细的实验数据**，用于后续模型优化")
        
        # 保存报告
        with open('experiment_design_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print("📋 实验报告已保存: experiment_design_report.md")
    
    def run_complete_design(self, n_experiments: int = 5, n_candidates: int = 1000,
                          use_constraints: bool = True, diversity_weight: float = 0.3,
                          save_plots: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        运行完整的实验设计流程
        
        Args:
            n_experiments: 实验数量
            n_candidates: 候选点数量
            use_constraints: 是否使用约束
            diversity_weight: 多样性权重
            save_plots: 是否保存图表
            
        Returns:
            results_df: 基础结果
            detailed_df: 详细建议
        """
        print("🚀 开始完整实验设计流程...\n")
        
        try:
            # 1. 数据处理
            self.load_and_process_data()
            
            # 2. 模型训练
            self.build_gpr_model()
            
            # 3. 候选点生成
            self.generate_candidate_points(n_candidates)
            
            # 4. 实验优化
            results_df = self.optimize_experiments(n_experiments, use_constraints, diversity_weight)
            
            # 5. 详细建议
            detailed_df = self.generate_detailed_recommendations(results_df)
            
            # 6. 可视化
            self.visualize_results(detailed_df, save_plots)
            
            # 7. 保存结果
            self.save_results(results_df, detailed_df)
            
            print("\n🎉 实验设计流程完成!")
            print("📋 请查看生成的文件:")
            print("   - experiment_design_results.csv: 基础结果")
            print("   - detailed_experiment_recommendations.csv: 详细建议")
            print("   - experiment_design_report.md: 实验报告")
            if save_plots:
                print("   - experiment_design_analysis.png: 分析图表")
            
            return results_df, detailed_df
            
        except Exception as e:
            print(f"❌ 实验设计流程失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """主函数"""
    print("🧪 乙醇偶合制备C4烯烃 - 实验设计系统")
    print("=" * 50)
    
    # 创建实验设计器
    designer = ExperimentDesigner(random_state=42)
    
    # 运行完整设计流程
    results_df, detailed_df = designer.run_complete_design(
        n_experiments=5,
        n_candidates=1000,
        use_constraints=True,
        diversity_weight=0.3,
        save_plots=True
    )
    
    if results_df is not None:
        print("\n📊 实验设计结果预览:")
        print(results_df[['experiment_id', 'T', 'Co_loading', 'predicted_yield', 'EI_value']].to_string(index=False))

if __name__ == "__main__":
    main()