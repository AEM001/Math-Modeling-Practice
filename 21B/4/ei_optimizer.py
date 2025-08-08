#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EI优化模块 - GPR实验设计系统
功能：计算期望改进(Expected Improvement)并选择最优实验点
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class EIOptimizer:
    """期望改进优化器"""
    
    def __init__(self, xi: float = 0.01):
        """
        初始化EI优化器
        
        Args:
            xi: 探索参数，控制探索与利用的平衡
        """
        self.xi = xi
        
    def calculate_ei(self, X: np.ndarray, gpr_model, y_best: float) -> np.ndarray:
        """
        计算期望改进值
        
        Args:
            X: 候选点矩阵 (n_samples, n_features)
            gpr_model: 训练好的GPR模型
            y_best: 当前最佳收率值
            
        Returns:
            ei_values: EI值数组
        """
        print(f" 计算{len(X)}个候选点的EI值...")
        
        # 获取GPR预测
        mu, sigma = gpr_model.predict(X, return_std=True)
        
        # 避免数值问题
        sigma = np.maximum(sigma, 1e-9)
        
        # 计算改进量
        improvement = mu - y_best - self.xi
        
        # 标准化改进量
        Z = improvement / sigma
        
        # 计算EI值
        ei_values = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # 处理数值异常
        ei_values = np.maximum(ei_values, 0)
        
        print(f" EI计算完成")
        print(f"    EI值范围: [{ei_values.min():.6f}, {ei_values.max():.6f}]")
        print(f"    平均EI值: {ei_values.mean():.6f}")
        
        return ei_values
    
    def calculate_constrained_ei(self, X: np.ndarray, gpr_model, y_best: float,
                                constraints: Dict = None) -> np.ndarray:
        """
        计算带约束的期望改进值
        
        Args:
            X: 候选点矩阵
            gpr_model: GPR模型
            y_best: 当前最佳值
            constraints: 约束条件字典
            
        Returns:
            constrained_ei: 约束后的EI值
        """
        # 基础EI计算
        ei_values = self.calculate_ei(X, gpr_model, y_best)
        
        if constraints is None:
            return ei_values
        
        print(" 应用约束条件...")
        
        # 应用约束惩罚
        penalty_factors = self._calculate_constraint_penalties(X, constraints)
        constrained_ei = ei_values * penalty_factors
        
        print(f"    约束后EI值范围: [{constrained_ei.min():.6f}, {constrained_ei.max():.6f}]")
        
        return constrained_ei
    
    def _calculate_constraint_penalties(self, X: np.ndarray, constraints: Dict) -> np.ndarray:
        """计算约束惩罚因子"""
        n_samples = len(X)
        penalty_factors = np.ones(n_samples)
        
        # 变量索引映射
        var_indices = {
            'T': 0, 'total_mass': 1, 'loading_ratio': 2,
            'Co_loading': 3, 'ethanol_conc': 4, 'loading_method': 5
        }
        
        for i in range(n_samples):
            penalty = 1.0
            candidate = X[i]
            
            # 温度约束
            T = candidate[var_indices['T']]
            if 'temperature_stability' in constraints:
                if T > 400:
                    penalty *= 0.8  # 高温稳定性惩罚
                if T < 275:
                    penalty *= 0.9  # 低温活性惩罚
            
            # Co负载量约束
            Co_loading = candidate[var_indices['Co_loading']]
            if 'co_loading_cost' in constraints:
                if Co_loading >= 5.0:
                    penalty *= 0.7  # 高负载量成本惩罚
            
            # 装料比约束
            loading_ratio = candidate[var_indices['loading_ratio']]
            if 'loading_ratio_stability' in constraints:
                if loading_ratio < 0.4 or loading_ratio > 1.8:
                    penalty *= 0.6  # 极端装料比惩罚
            
            # 组合约束
            if 'combination_stability' in constraints:
                if T > 400 and Co_loading >= 5.0:
                    penalty *= 0.5  # 高温高负载组合惩罚
                if T > 350 and (loading_ratio < 0.4 or loading_ratio > 1.8):
                    penalty *= 0.6  # 高温极端装料比惩罚
            
            penalty_factors[i] = penalty
        
        return penalty_factors
    
    def select_optimal_experiments(self, X: np.ndarray, ei_values: np.ndarray,
                                 gpr_model, n_experiments: int = 5,
                                 diversity_weight: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        选择最优实验点
        
        Args:
            X: 候选点矩阵
            ei_values: EI值数组
            gpr_model: GPR模型
            n_experiments: 需要选择的实验数量
            diversity_weight: 多样性权重
            
        Returns:
            selected_indices: 选中点的索引
            selected_points: 选中的实验点
            selection_info: 选择信息
        """
        print(f" 选择{n_experiments}个最优实验点...")
        
        selected_indices = []
        remaining_indices = np.arange(len(X))
        selection_info = {
            'ei_values': [],
            'predicted_yields': [],
            'uncertainties': [],
            'diversity_scores': []
        }
        
        for i in range(n_experiments):
            print(f"   选择第{i+1}个实验点...")
            
            if i == 0:
                # 第一个点：选择EI值最高的
                best_idx = np.argmax(ei_values[remaining_indices])
                selected_idx = remaining_indices[best_idx]
            else:
                # 后续点：考虑EI值和多样性
                scores = self._calculate_selection_scores(
                    X, ei_values, remaining_indices, selected_indices,
                    gpr_model, diversity_weight
                )
                best_idx = np.argmax(scores)
                selected_idx = remaining_indices[best_idx]
            
            # 记录选择信息
            mu, sigma = gpr_model.predict(X[selected_idx:selected_idx+1], return_std=True)
            
            selection_info['ei_values'].append(ei_values[selected_idx])
            selection_info['predicted_yields'].append(mu[0])
            selection_info['uncertainties'].append(sigma[0])
            
            if i > 0:
                diversity_score = self._calculate_diversity_score(
                    X[selected_idx], X[selected_indices]
                )
                selection_info['diversity_scores'].append(diversity_score)
            else:
                selection_info['diversity_scores'].append(0.0)
            
            # 更新选择列表
            selected_indices.append(selected_idx)
            remaining_indices = np.delete(remaining_indices, best_idx)
            
            print(f"     选中点{selected_idx}: EI={ei_values[selected_idx]:.6f}, "
                  f"预测收率={mu[0]:.4f}±{sigma[0]:.4f}")
        
        selected_points = X[selected_indices]
        
        print(f" 实验点选择完成！")
        
        return np.array(selected_indices), selected_points, selection_info
    
    def _calculate_selection_scores(self, X: np.ndarray, ei_values: np.ndarray,
                                  remaining_indices: np.ndarray, selected_indices: List[int],
                                  gpr_model, diversity_weight: float) -> np.ndarray:
        """计算选择分数（EI + 多样性）"""
        scores = np.zeros(len(remaining_indices))
        
        for i, idx in enumerate(remaining_indices):
            # EI分数（归一化）
            ei_score = ei_values[idx] / np.max(ei_values)
            
            # 多样性分数
            diversity_score = self._calculate_diversity_score(
                X[idx], X[selected_indices]
            )
            
            # 综合分数
            scores[i] = (1 - diversity_weight) * ei_score + diversity_weight * diversity_score
        
        return scores
    
    def _calculate_diversity_score(self, candidate: np.ndarray, selected_points: np.ndarray) -> float:
        """计算多样性分数"""
        if len(selected_points) == 0:
            return 1.0
        
        # 计算到已选点的最小距离
        distances = np.linalg.norm(selected_points - candidate, axis=1)
        min_distance = np.min(distances)
        
        # 归一化多样性分数
        max_possible_distance = np.sqrt(len(candidate))  # 假设特征已标准化
        diversity_score = min_distance / max_possible_distance
        
        return diversity_score
    
    def analyze_selection_results(self, selected_points: np.ndarray, selection_info: Dict,
                                feature_names: List[str]) -> pd.DataFrame:
        """分析选择结果"""
        print(" 分析实验点选择结果...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(selected_points, columns=feature_names)
        results_df['EI_value'] = selection_info['ei_values']
        results_df['predicted_yield'] = selection_info['predicted_yields']
        results_df['uncertainty'] = selection_info['uncertainties']
        results_df['diversity_score'] = selection_info['diversity_scores']
        
        # 计算选择理由
        results_df['selection_reason'] = self._generate_selection_reasons(
            results_df, feature_names
        )
        
        print(" 结果分析完成")
        
        return results_df
    
    def _generate_selection_reasons(self, results_df: pd.DataFrame, 
                                  feature_names: List[str]) -> List[str]:
        """生成选择理由"""
        reasons = []
        
        for i, row in results_df.iterrows():
            reason_parts = []
            
            # EI值分析
            if row['EI_value'] > results_df['EI_value'].mean():
                reason_parts.append("高EI值")
            
            # 预测收率分析
            if row['predicted_yield'] > results_df['predicted_yield'].mean():
                reason_parts.append("高预测收率")
            
            # 不确定性分析
            if row['uncertainty'] > results_df['uncertainty'].mean():
                reason_parts.append("高不确定性区域")
            
            # 多样性分析
            if i > 0 and row['diversity_score'] > 0.5:
                reason_parts.append("增强多样性")
            
            # 特殊条件分析
            if row['T'] > 400:
                reason_parts.append("高温条件探索")
            if row['Co_loading'] >= 5.0:
                reason_parts.append("高Co负载探索")
            
            if not reason_parts:
                reason_parts.append("综合优化选择")
            
            reasons.append(" + ".join(reason_parts))
        
        return reasons
    
    def save_results(self, results_df: pd.DataFrame, filepath: str):
        """保存选择结果"""
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f" 实验设计结果已保存: {filepath}")

def main():
    """测试EI优化模块"""
    print(" 测试EI优化模块...")
    
    try:
        # 创建测试数据
        np.random.seed(42)
        n_candidates = 100
        n_features = 6
        
        X_test = np.random.randn(n_candidates, n_features)
        y_best = 0.15  # 假设当前最佳收率为15%
        
        # 模拟GPR模型预测
        class MockGPRModel:
            def predict(self, X, return_std=False):
                mu = 0.1 + 0.05 * np.random.randn(len(X))
                if return_std:
                    sigma = 0.02 + 0.01 * np.random.rand(len(X))
                    return mu, sigma
                return mu
        
        mock_model = MockGPRModel()
        
        # 测试EI计算
        optimizer = EIOptimizer(xi=0.01)
        ei_values = optimizer.calculate_ei(X_test, mock_model, y_best)
        
        # 测试约束EI
        constraints = {
            'temperature_stability': True,
            'co_loading_cost': True,
            'combination_stability': True
        }
        constrained_ei = optimizer.calculate_constrained_ei(
            X_test, mock_model, y_best, constraints
        )
        
        # 测试实验点选择
        selected_indices, selected_points, selection_info = optimizer.select_optimal_experiments(
            X_test, constrained_ei, mock_model, n_experiments=5
        )
        
        # 分析结果
        feature_names = ['T', 'total_mass', 'loading_ratio', 'Co_loading', 'ethanol_conc', 'loading_method']
        results_df = optimizer.analyze_selection_results(
            selected_points, selection_info, feature_names
        )
        
        print("\n 选择结果:")
        print(results_df)
        
        # 保存结果
        optimizer.save_results(results_df, 'test_experiment_design.csv')
        
        print("\n EI优化模块测试通过！")
        
    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()