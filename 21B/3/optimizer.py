import numpy as np
from scipy.optimize import minimize
from itertools import product
import pandas as pd

def create_objective_function(model, poly_features, scaler):
    """
    为多维连续优化创建目标函数
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        
    Returns:
        function: 目标函数，接收一个包含5个连续变量的列表
    """
    
    def objective_function(continuous_params):
        """
        目标函数：返回负的C4烯烃收率（因为优化器求解最小值）
        
        Args:
            continuous_params: [T, total_mass, loading_ratio, C, C_e]
            
        Returns:
            float: 负的C4烯烃收率
        """
        # 组合5个连续变量
        X_input = np.array(continuous_params)
        
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
                          continuous_bounds):
    """
    寻找最优工艺条件 (改进版：使用多起点优化策略)
    
    优化策略：
    1. 对每个 M 值，从多个随机起点进行优化，以寻找全局最优解
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        continuous_bounds: 连续变量边界字典
        
    Returns:
        dict: 包含最优参数组合和最大收率
    """
    
    print("开始寻找最优工艺条件 (采用多起点优化策略)...")
    print(f"温度范围: {continuous_bounds['T'][0]}°C - {continuous_bounds['T'][1]}°C")
    print(f"总质量范围: {continuous_bounds['total_mass'][0]} - {continuous_bounds['total_mass'][1]} mg")
    print(f"装料比范围: {continuous_bounds['loading_ratio'][0]} - {continuous_bounds['loading_ratio'][1]}")
    print(f"Co负载量范围: {continuous_bounds['C'][0]} - {continuous_bounds['C'][1]} wt%")
    print(f"乙醇浓度范围: {continuous_bounds['C_e'][0]} - {continuous_bounds['C_e'][1]} ml/min")
    
    # 1. 准备所有5个连续变量的边界
    bounds_list = [
        continuous_bounds['T'],
        continuous_bounds['total_mass'],
        continuous_bounds['loading_ratio'],
        continuous_bounds['C'],
        continuous_bounds['C_e']
    ]
    
    # 存储所有结果
    all_results = []
    best_result_overall = None
    best_yield_overall = -np.inf
    
    # 定义多起点优化的起点数量
    n_starts = 20
    print(f"将使用 {n_starts} 个随机起点进行优化，以增加找到全局最优解的概率。")
    
    # 2. 为当前 M 值创建目标函数
    objective_func = create_objective_function(model, poly_features, scaler)
    
    # 当前M值下的最优解
    best_yield_for_m = -np.inf
    best_params_for_m = None

    # 3. 多起点优化循环
    for i in range(n_starts):
        # 生成随机初始猜测值
        initial_guess = [np.random.uniform(low, high) for low, high in bounds_list]
        
        # 4. 使用scipy.optimize.minimize进行5维连续优化
        try:
            result = minimize(
                objective_func,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds_list
            )
            
            if result.success:
                current_yield = -result.fun
                # 如果找到了更好的解，则更新
                if current_yield > best_yield_for_m:
                    best_yield_for_m = current_yield
                    best_params_for_m = result.x
        except Exception as e:
            # 即使单个起点失败，也继续尝试其他起点
            continue
    
    # 记录当前M值的最佳结果
    if best_params_for_m is not None:
        result_dict = {
            'T': best_params_for_m[0],
            'total_mass': best_params_for_m[1],
            'loading_ratio': best_params_for_m[2],
            'C': best_params_for_m[3],
            'C_e': best_params_for_m[4],
            'yield': best_yield_for_m
        }
        all_results.append(result_dict)
        
        # 更新全局最优结果
        if best_yield_for_m > best_yield_overall:
            best_yield_overall = best_yield_for_m
            best_result_overall = result_dict.copy()

    # 按收率排序
    all_results.sort(key=lambda x: x['yield'], reverse=True)
    
    print(f"\n优化完成！")
    if best_result_overall:
        print(f"最优收率: {best_yield_overall:.4f}")
    else:
        print("警告：未能找到任何有效的优化解。")
    
    # 返回最优结果和所有结果
    return {
        'best_params': best_result_overall,
        'max_yield': best_yield_overall,
        'all_results': all_results
    }

def analyze_optimization_results(optimization_result):
    """
    分析优化结果
    
    Args:
        optimization_result: 优化结果字典
        
    Returns:
        None (打印分析结果)
    """
    
    best_params = optimization_result['best_params']
    max_yield = optimization_result['max_yield']
    all_results = optimization_result['all_results']
    
    print("\n" + "="*50)
    print("优化结果分析")
    print("="*50)
    
    if not best_params:
        print("\n未能找到有效的最优解。")
        return optimization_result

    print(f"\n最优工艺条件:")
    print(f"温度 (T): {best_params['T']:.2f}°C")
    print(f"总质量 (total_mass): {best_params['total_mass']:.2f} mg")
    print(f"装料比 (loading_ratio): {best_params['loading_ratio']:.2f}")
    print(f"Co负载量 (C): {best_params['C']:.2f} wt%")
    print(f"乙醇浓度 (C_e): {best_params['C_e']:.2f} ml/min")
    print(f"预测C4烯烃收率: {max_yield:.4f}")
    
    print(f"\n前5个最优组合:")
    if all_results:
        for i, result in enumerate(all_results[:5]):
            print(f"\n第{i+1}名:")
            print(f"  温度: {result['T']:.2f}°C")
            print(f"  总质量: {result['total_mass']:.2f} mg")
            print(f"  装料比: {result['loading_ratio']:.2f}")
            print(f"  Co负载量: {result['C']:.2f} wt%")
            print(f"  乙醇浓度: {result['C_e']:.2f} ml/min")
            print(f"  预测收率: {result['yield']:.4f}")
    
    # 分析各变量的分布
    print(f"\n最优解中各变量的分布:")
    if all_results:
        top_results = pd.DataFrame(all_results[:10])
        for var in ['T', 'total_mass', 'loading_ratio', 'C', 'C_e']:
            if var in top_results.columns:
                unique_values = top_results[var].unique()
                print(f"{var}: {sorted(unique_values)}")
    
    return optimization_result