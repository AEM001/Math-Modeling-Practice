#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
候选点生成模块 - GPR实验设计系统
功能：使用拉丁超立方采样生成候选实验点
"""

import numpy as np
import pandas as pd
# from pyDOE import lhs  # 使用自定义LHS实现
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import itertools
import warnings
warnings.filterwarnings('ignore')

class CandidateGenerator:
    """候选实验点生成器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _lhs_sample(self, n_dims: int, n_samples: int) -> np.ndarray:
        """
        自定义拉丁超立方采样实现
        
        Args:
            n_dims: 维度数
            n_samples: 样本数
            
        Returns:
            samples: LHS样本矩阵 [0,1]^(n_samples x n_dims)
        """
        # 初始化样本矩阵
        samples = np.zeros((n_samples, n_dims))
        
        # 对每个维度进行LHS采样
        for dim in range(n_dims):
            # 生成等间距的区间
            intervals = np.linspace(0, 1, n_samples + 1)
            
            # 在每个区间内随机采样
            for i in range(n_samples):
                samples[i, dim] = np.random.uniform(intervals[i], intervals[i + 1])
            
            # 随机打乱顺序
            np.random.shuffle(samples[:, dim])
        
        return samples
        
    def generate_candidates(self, variable_bounds: Dict, n_candidates: int = 1000,
                          scaler: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成候选实验点
        
        Args:
            variable_bounds: 变量边界字典
            n_candidates: 候选点数量
            scaler: 数据标准化器
            
        Returns:
            candidates_scaled: 标准化后的候选点
            candidates_original: 原始尺度的候选点
        """
        print(f" 生成{n_candidates}个候选实验点...")
        
        # 1. 分离连续变量和离散变量
        continuous_vars, discrete_vars = self._separate_variables(variable_bounds)
        
        # 2. 生成连续变量的LHS采样
        print("   连续变量LHS采样...")
        continuous_samples = self._generate_continuous_samples(continuous_vars, n_candidates)
        
        # 3. 生成离散变量的随机采样
        print("   离散变量随机采样...")
        discrete_samples = self._generate_discrete_samples(discrete_vars, n_candidates)
        
        # 4. 合并候选点
        print("   合并候选点...")
        candidates_original = self._combine_samples(continuous_samples, discrete_samples, 
                                                   continuous_vars, discrete_vars)
        
        # 5. 标准化候选点
        if scaler is not None:
            print("   标准化候选点...")
            candidates_scaled = scaler.transform(candidates_original)
        else:
            candidates_scaled = candidates_original.copy()
        
        print(f" 候选点生成完成！")
        print(f"    候选点数量: {len(candidates_original)}")
        print(f"    特征维度: {candidates_original.shape[1]}")
        
        return candidates_scaled, candidates_original
    
    def _separate_variables(self, variable_bounds: Dict) -> Tuple[Dict, Dict]:
        """分离连续变量和离散变量"""
        continuous_vars = {}
        discrete_vars = {}
        
        for var_name, bounds in variable_bounds.items():
            if isinstance(bounds, tuple):
                # 连续变量
                continuous_vars[var_name] = bounds
            elif isinstance(bounds, list):
                # 离散变量
                discrete_vars[var_name] = bounds
            else:
                raise ValueError(f"变量{var_name}的边界格式不正确")
        
        return continuous_vars, discrete_vars
    
    def _generate_continuous_samples(self, continuous_vars: Dict, n_samples: int) -> np.ndarray:
        """生成连续变量的LHS采样"""
        if not continuous_vars:
            return np.empty((n_samples, 0))
        
        n_continuous = len(continuous_vars)
        
        # LHS采样 [0,1]^n
        lhs_samples = self._lhs_sample(n_continuous, n_samples)
        
        # 缩放到实际范围
        continuous_samples = np.zeros_like(lhs_samples)
        for i, (var_name, (min_val, max_val)) in enumerate(continuous_vars.items()):
            continuous_samples[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)
        
        return continuous_samples
    
    def _generate_discrete_samples(self, discrete_vars: Dict, n_samples: int) -> np.ndarray:
        """生成离散变量的随机采样"""
        if not discrete_vars:
            return np.empty((n_samples, 0))
        
        n_discrete = len(discrete_vars)
        discrete_samples = np.zeros((n_samples, n_discrete))
        
        for i, (var_name, values) in enumerate(discrete_vars.items()):
            # 随机选择离散值
            discrete_samples[:, i] = np.random.choice(values, size=n_samples)
        
        return discrete_samples
    
    def _combine_samples(self, continuous_samples: np.ndarray, discrete_samples: np.ndarray,
                        continuous_vars: Dict, discrete_vars: Dict) -> np.ndarray:
        """合并连续和离散样本"""
        # 确定变量顺序（按照标准顺序）
        var_order = ['T', 'total_mass', 'loading_ratio', 'Co_loading', 'ethanol_conc', 'loading_method']
        
        n_samples = max(continuous_samples.shape[0], discrete_samples.shape[0])
        combined_samples = np.zeros((n_samples, len(var_order)))
        
        continuous_idx = 0
        discrete_idx = 0
        
        for i, var_name in enumerate(var_order):
            if var_name in continuous_vars:
                combined_samples[:, i] = continuous_samples[:, continuous_idx]
                continuous_idx += 1
            elif var_name in discrete_vars:
                combined_samples[:, i] = discrete_samples[:, discrete_idx]
                discrete_idx += 1
            else:
                raise ValueError(f"未知变量: {var_name}")
        
        return combined_samples
    
    def generate_grid_candidates(self, variable_bounds: Dict, 
                               scaler: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成网格候选点（用于小规模精确搜索）
        
        Args:
            variable_bounds: 变量边界字典
            scaler: 数据标准化器
            
        Returns:
            candidates_scaled: 标准化后的候选点
            candidates_original: 原始尺度的候选点
        """
        print(" 生成网格候选点...")
        
        # 定义网格点
        grid_points = {}
        
        for var_name, bounds in variable_bounds.items():
            if isinstance(bounds, tuple):
                # 连续变量：生成等间距网格
                min_val, max_val = bounds
                if var_name == 'T':
                    # 温度：使用实际实验温度点
                    grid_points[var_name] = [250, 275, 300, 325, 350, 400, 450]
                elif var_name == 'total_mass':
                    # 总质量：10个等间距点
                    grid_points[var_name] = np.linspace(min_val, max_val, 10).tolist()
                elif var_name == 'loading_ratio':
                    # 装料比：8个等间距点
                    grid_points[var_name] = np.linspace(min_val, max_val, 8).tolist()
            elif isinstance(bounds, list):
                # 离散变量：使用所有可能值
                grid_points[var_name] = bounds
        
        # 生成所有组合
        var_order = ['T', 'total_mass', 'loading_ratio', 'Co_loading', 'ethanol_conc', 'loading_method']
        grid_values = [grid_points[var] for var in var_order]
        
        # 笛卡尔积
        all_combinations = list(itertools.product(*grid_values))
        candidates_original = np.array(all_combinations)
        
        # 标准化
        if scaler is not None:
            candidates_scaled = scaler.transform(candidates_original)
        else:
            candidates_scaled = candidates_original.copy()
        
        print(f" 网格候选点生成完成！")
        print(f"    候选点数量: {len(candidates_original)}")
        
        return candidates_scaled, candidates_original
    
    def filter_feasible_candidates(self, candidates_original: np.ndarray) -> np.ndarray:
        """
        过滤可行的候选点
        
        Args:
            candidates_original: 原始候选点
            
        Returns:
            feasible_mask: 可行点的布尔掩码
        """
        n_candidates = len(candidates_original)
        feasible_mask = np.ones(n_candidates, dtype=bool)
        
        # 变量索引
        var_indices = {
            'T': 0, 'total_mass': 1, 'loading_ratio': 2,
            'Co_loading': 3, 'ethanol_conc': 4, 'loading_method': 5
        }
        
        # 约束条件
        for i in range(n_candidates):
            candidate = candidates_original[i]
            
            # 1. 温度约束
            T = candidate[var_indices['T']]
            if T < 250 or T > 450:
                feasible_mask[i] = False
                continue
            
            # 2. 质量约束
            total_mass = candidate[var_indices['total_mass']]
            if total_mass < 20 or total_mass > 400:
                feasible_mask[i] = False
                continue
            
            # 3. 装料比约束
            loading_ratio = candidate[var_indices['loading_ratio']]
            if loading_ratio < 0.33 or loading_ratio > 2.03:
                feasible_mask[i] = False
                continue
            
            # 4. 工艺约束：高温+高Co负载量可能不稳定
            Co_loading = candidate[var_indices['Co_loading']]
            if T > 400 and Co_loading >= 5.0:
                feasible_mask[i] = False
                continue
            
            # 5. 装料约束：极端装料比+高温可能有问题
            if T > 350 and (loading_ratio < 0.4 or loading_ratio > 1.8):
                feasible_mask[i] = False
                continue
        
        print(f" 可行性过滤完成: {feasible_mask.sum()}/{len(feasible_mask)} 点可行")
        
        return feasible_mask
    
    def save_candidates(self, candidates_original: np.ndarray, filepath: str):
        """保存候选点"""
        feature_names = ['T', 'total_mass', 'loading_ratio', 'Co_loading', 'ethanol_conc', 'loading_method']
        
        df = pd.DataFrame(candidates_original, columns=feature_names)
        df.to_csv(filepath, index=False)
        print(f" 候选点已保存: {filepath}")

def main():
    """测试候选点生成模块"""
    from data_processor import DataProcessor
    
    print(" 测试候选点生成模块...")
    
    try:
        # 1. 获取变量边界
        processor = DataProcessor()
        variable_bounds = processor.get_variable_bounds()
        
        print(" 变量边界:")
        for var, bounds in variable_bounds.items():
            print(f"  {var}: {bounds}")
        
        # 2. 生成候选点
        generator = CandidateGenerator(random_state=42)
        
        # LHS采样
        candidates_scaled, candidates_original = generator.generate_candidates(
            variable_bounds, n_candidates=1000
        )
        
        # 网格采样
        grid_scaled, grid_original = generator.generate_grid_candidates(variable_bounds)
        
        # 3. 可行性过滤
        feasible_mask = generator.filter_feasible_candidates(candidates_original)
        feasible_candidates = candidates_original[feasible_mask]
        
        print(f"\n 候选点统计:")
        print(f"LHS候选点: {len(candidates_original)}")
        print(f"网格候选点: {len(grid_original)}")
        print(f"可行LHS候选点: {len(feasible_candidates)}")
        
        # 4. 保存候选点
        generator.save_candidates(feasible_candidates, 'feasible_candidates.csv')
        generator.save_candidates(grid_original, 'grid_candidates.csv')
        
        print("\n 候选点生成模块测试通过！")
        
    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()