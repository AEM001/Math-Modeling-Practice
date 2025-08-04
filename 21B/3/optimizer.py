import numpy as np
from scipy.optimize import minimize
from itertools import product
import pandas as pd

def create_objective_function(model, poly_features, scaler, fixed_discrete_params):
    """
    为多维连续优化创建目标函数
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        fixed_discrete_params: 一个包含固定的离散变量值的列表 [C, C_e, M]
        
    Returns:
        function: 目标函数，接收一个连续变量列表 [T, total_mass, loading_ratio]
    """
    
    def objective_function(continuous_params):
        """
        目标函数：返回负的C4烯烃收率（因为优化器求解最小值）
        
        Args:
            continuous_params: 连续变量 [T, total_mass, loading_ratio]
            
        Returns:
            float: 负的C4烯烃收率
        """
        # 组合连续变量和固定的离散变量
        X_input = np.array([
            continuous_params[0],  # T
            continuous_params[1],  # total_mass
            continuous_params[2],  # loading_ratio
            fixed_discrete_params[0],  # C
            fixed_discrete_params[1],  # C_e
            fixed_discrete_params[2]   # M
        ])
        
        # 标准化输入
        X_scaled = scaler.transform(X_input.reshape(1, -1))
        
        # 生成多项式特征
        X_poly = poly_features.transform(X_scaled)
        
        # 预测
        prediction = model.predict(X_poly)
        
        # 返回负值（因为优化器求解最小值）
        return -prediction[0]
    
    return objective_function

def find_optimal_conditions(model, poly_features, scaler, 
                          continuous_bounds, discrete_options):
    """
    寻找最优工艺条件
    
    优化策略：
    1. 遍历所有离散变量组合
    2. 对每个组合，优化三个连续变量（T, total_mass, loading_ratio）
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        continuous_bounds: 连续变量边界字典
        discrete_options: 离散变量选项字典
        
    Returns:
        dict: 包含最优参数组合和最大收率
    """
    
    print("开始寻找最优工艺条件...")
    print(f"温度范围: {continuous_bounds['T'][0]}°C - {continuous_bounds['T'][1]}°C")
    print(f"总质量范围: {continuous_bounds['total_mass'][0]} - {continuous_bounds['total_mass'][1]} mg")
    print(f"装料比范围: {continuous_bounds['loading_ratio'][0]} - {continuous_bounds['loading_ratio'][1]}")
    
    # 1. 获取所有离散变量组合
    discrete_vars = ['C', 'C_e', 'M']
    discrete_combinations = list(product(*[discrete_options[var] for var in discrete_vars]))
    
    print(f"需要评估 {len(discrete_combinations)} 种离散变量组合")
    
    # 2. 准备连续变量的边界
    bounds_list = [
        continuous_bounds['T'],
        continuous_bounds['total_mass'],
        continuous_bounds['loading_ratio']
    ]
    
    # 存储所有结果
    all_results = []
    best_result = None
    best_yield = -np.inf
    
    # 3. 遍历所有离散变量组合
    for i, discrete_combo in enumerate(discrete_combinations):
        if i % 10 == 0:
            print(f"正在评估第 {i+1}/{len(discrete_combinations)} 种离散组合...")
        
        # 4. 为当前离散组合创建目标函数
        objective_func = create_objective_function(model, poly_features, scaler, discrete_combo)
        
        # 5. 设置连续变量的初始猜测值 (使用边界的中间值)
        initial_guess = [np.mean(b) for b in bounds_list]
        
        # 6. 使用scipy.optimize.minimize进行多维优化
        try:
            result = minimize(
                objective_func,
                initial_guess,
                method='L-BFGS-B',  # 支持边界约束的高效算法
                bounds=bounds_list
            )
            
            if result.success:
                optimal_yield = -result.fun
                optimal_continuous_params = result.x
                
                # 记录结果
                result_dict = {
                    'T': optimal_continuous_params[0],
                    'total_mass': optimal_continuous_params[1],
                    'loading_ratio': optimal_continuous_params[2],
                    'C': discrete_combo[0],
                    'C_e': discrete_combo[1],
                    'M': discrete_combo[2],
                    'yield': optimal_yield
                }
                
                all_results.append(result_dict)
                
                # 更新最优结果
                if optimal_yield > best_yield:
                    best_yield = optimal_yield
                    best_result = result_dict.copy()
                    
        except Exception as e:
            print(f"组合 {discrete_combo} 优化失败: {e}")
            continue
    
    # 按收率排序
    all_results.sort(key=lambda x: x['yield'], reverse=True)
    
    print(f"\n优化完成！")
    print(f"评估了 {len(all_results)} 种有效组合")
    print(f"最优收率: {best_yield:.4f}")
    
    # 返回最优结果和所有结果
    return {
        'best_params': best_result,
        'max_yield': best_yield,
        'all_results': all_results[:10]  # 返回前10个最优结果
    }

def analyze_optimization_results(optimization_result, discrete_options):
    """
    分析优化结果
    
    Args:
        optimization_result: 优化结果字典
        discrete_options: 离散变量选项
        
    Returns:
        None (打印分析结果)
    """
    
    best_params = optimization_result['best_params']
    max_yield = optimization_result['max_yield']
    all_results = optimization_result['all_results']
    
    print("\n" + "="*50)
    print("优化结果分析")
    print("="*50)
    
    print(f"\n最优工艺条件:")
    print(f"温度 (T): {best_params['T']:.2f}°C")
    print(f"总质量 (total_mass): {best_params['total_mass']:.2f} mg")
    print(f"装料比 (loading_ratio): {best_params['loading_ratio']:.2f}")
    print(f"Co负载量 (C): {best_params['C']} wt%")
    print(f"乙醇浓度 (C_e): {best_params['C_e']} ml/min")
    print(f"装料方式 (M): {'B系列' if best_params['M'] == 1 else 'A系列'}")
    print(f"预测C4烯烃收率: {max_yield:.4f}")
    
    print(f"\n前5个最优组合:")
    for i, result in enumerate(all_results[:5]):
        print(f"\n第{i+1}名:")
        print(f"  温度: {result['T']:.2f}°C")
        print(f"  总质量: {result['total_mass']:.2f} mg")
        print(f"  装料比: {result['loading_ratio']:.2f}")
        print(f"  Co负载量: {result['C']} wt%")
        print(f"  乙醇浓度: {result['C_e']} ml/min")
        print(f"  装料方式: {'B系列' if result['M'] == 1 else 'A系列'}")
        print(f"  预测收率: {result['yield']:.4f}")
    
    # 分析各变量的分布
    print(f"\n最优解中各变量的分布:")
    top_results = pd.DataFrame(all_results[:10])
    
    for var in ['T', 'total_mass', 'loading_ratio', 'C', 'C_e', 'M']:
        if var in top_results.columns:
            unique_values = top_results[var].unique()
            print(f"{var}: {sorted(unique_values)}")
    
    return optimization_result 