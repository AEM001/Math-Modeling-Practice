import numpy as np
from scipy.optimize import minimize
from itertools import product
import pandas as pd

def create_objective_function(model, poly_features, scaler, fixed_m_value):
    """
    为多维连续优化创建目标函数
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        fixed_m_value: 固定的装料方式 M (0或1)
        
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
        # 组合5个连续变量和1个固定的离散变量
        X_input = np.array([
            continuous_params[0],  # T
            continuous_params[1],  # total_mass
            continuous_params[2],  # loading_ratio
            continuous_params[3],  # C (现在是连续的)
            continuous_params[4],  # C_e (现在是连续的)
            fixed_m_value          # M
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
    寻找最优工艺条件 (改进版：使用多起点优化策略)
    
    优化策略：
    1. 遍历唯一的离散变量 M (装料方式)
    2. 对每个 M 值，从多个随机起点进行优化，以寻找全局最优解
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        continuous_bounds: 连续变量边界字典
        discrete_options: 离散变量选项字典
        
    Returns:
        dict: 包含最优参数组合和最大收率
    """
    
    print("开始寻找最优工艺条件 (采用多起点优化策略)...")
    print(f"温度范围: {continuous_bounds['T'][0]}°C - {continuous_bounds['T'][1]}°C")
    print(f"总质量范围: {continuous_bounds['total_mass'][0]} - {continuous_bounds['total_mass'][1]} mg")
    print(f"装料比范围: {continuous_bounds['loading_ratio'][0]} - {continuous_bounds['loading_ratio'][1]}")
    print(f"Co负载量范围: {continuous_bounds['C'][0]} - {continuous_bounds['C'][1]} wt%")
    print(f"乙醇浓度范围: {continuous_bounds['C_e'][0]} - {continuous_bounds['C_e'][1]} ml/min")
    
    # 1. 获取唯一的离散变量组合 (现在只有M)
    m_options = discrete_options['M']
    print(f"需要为 {len(m_options)} 种装料方式分别进行优化")
    
    # 2. 准备所有5个连续变量的边界
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
    print(f"每个装料方式将使用 {n_starts} 个随机起点进行优化，以增加找到全局最优解的概率。")
    
    # 3. 遍历唯一的离散变量 M
    for m_value in m_options:
        print(f"\n正在为装料方式 M={m_value} ('{'B系列' if m_value == 1 else 'A系列'}') 进行优化...")
        
        # 4. 为当前 M 值创建目标函数
        objective_func = create_objective_function(model, poly_features, scaler, m_value)
        
        # 当前M值下的最优解
        best_yield_for_m = -np.inf
        best_params_for_m = None

        # 5. 多起点优化循环
        for i in range(n_starts):
            # 生成随机初始猜测值
            initial_guess = [np.random.uniform(low, high) for low, high in bounds_list]
            
            # 6. 使用scipy.optimize.minimize进行5维连续优化
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
                'M': m_value,
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
        'all_results': all_results[:10]
    }
    
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