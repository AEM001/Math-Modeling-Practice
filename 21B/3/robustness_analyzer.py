import numpy as np
import pandas as pd
from model_builder import predict_yield

def analyze_robustness_and_report(optimal_solution, model, poly_features, scaler, 
                                discrete_options):
    """
    稳健性分析
    
    对最优解进行参数扰动分析，评估工艺条件的稳定性
    
    Args:
        optimal_solution: 最优解字典
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        discrete_options: 离散变量选项
    """
    
    print("\n" + "="*60)
    print("稳健性分析报告")
    print("="*60)
    
    best_params = optimal_solution['best_params']
    max_yield = optimal_solution['max_yield']
    
    print(f"\n基准最优解:")
    print(f"温度: {best_params['T']:.2f}°C")
    print(f"总质量: {best_params['total_mass']:.2f} mg")
    print(f"装料比: {best_params['loading_ratio']:.2f}")
    print(f"Co负载量: {best_params['C']} wt%")
    print(f"乙醇浓度: {best_params['C_e']} ml/min")
    print(f"装料方式: {'B系列' if best_params['M'] == 1 else 'A系列'}")
    print(f"预测收率: {max_yield:.4f}")
    
    # 进行稳健性分析
    robustness_results = perform_robustness_analysis(
        best_params, model, poly_features, scaler, discrete_options
    )
    
    # 打印稳健性分析结果
    print_robustness_results(robustness_results)
    
    return robustness_results

def perform_robustness_analysis(best_params, model, poly_features, scaler, 
                              discrete_options):
    """
    执行稳健性分析
    
    Args:
        best_params: 最优参数
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        discrete_options: 离散变量选项
        
    Returns:
        dict: 稳健性分析结果
    """
    
    # 获取基准收率
    base_params = np.array([
        best_params['T'], best_params['total_mass'], best_params['loading_ratio'],
        best_params['C'], best_params['C_e'], best_params['M']
    ])
    
    # 标准化输入
    X_scaled = scaler.transform(base_params.reshape(1, -1))
    
    # 生成多项式特征
    X_poly = poly_features.transform(X_scaled)
    
    # 预测基准收率
    base_yield = model.predict(X_poly)[0]
    
    # 扰动分析结果
    perturbation_results = {}
    
    # 对每个参数进行扰动分析
    param_names = ['T', 'total_mass', 'loading_ratio', 'C', 'C_e', 'M']
    param_labels = ['温度', '总质量', '装料比', 'Co负载量', '乙醇浓度', '装料方式']
    
    for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        print(f"\n正在分析 {param_label} 的稳健性...")
        
        if param_name in ['T', 'total_mass', 'loading_ratio']:
            # 连续变量
            yield_changes = analyze_continuous_parameter(
                base_params, i, param_name, model, poly_features, scaler
            )
        else:
            # 离散变量
            yield_changes = analyze_discrete_parameter(
                base_params, i, param_name, discrete_options[param_name], 
                model, poly_features, scaler
            )
        
        perturbation_results[param_name] = {
            'label': param_label,
            'yield_changes': yield_changes,
            'sensitivity': calculate_sensitivity(yield_changes)
        }
    
    return {
        'base_yield': base_yield,
        'perturbation_results': perturbation_results
    }

def analyze_continuous_parameter(base_params, param_index, param_name, model, 
                              poly_features, scaler):
    """
    分析连续参数的稳健性
    
    Args:
        base_params: 基准参数
        param_index: 参数索引
        param_name: 参数名称
        model: 模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        
    Returns:
        dict: 收率变化结果
    """
    
    base_value = base_params[param_index]
    perturbation_range = 0.05  # ±5%扰动
    
    # 计算扰动值
    lower_value = base_value * (1 - perturbation_range)
    upper_value = base_value * (1 + perturbation_range)
    
    # 测试多个扰动点
    test_values = np.linspace(lower_value, upper_value, 11)
    yields = []
    
    for test_value in test_values:
        # 创建扰动后的参数
        perturbed_params = base_params.copy()
        perturbed_params[param_index] = test_value
        
        # 标准化输入
        X_scaled = scaler.transform(perturbed_params.reshape(1, -1))
        
        # 生成多项式特征
        X_poly = poly_features.transform(X_scaled)
        
        # 预测收率
        yield_pred = model.predict(X_poly)[0]
        yields.append(yield_pred)
    
    return {
        'test_values': test_values,
        'yields': yields,
        'min_yield': min(yields),
        'max_yield': max(yields),
        'yield_range': max(yields) - min(yields)
    }

def analyze_discrete_parameter(base_params, param_index, param_name, discrete_options, 
                            model, poly_features, scaler):
    """
    分析离散参数的稳健性
    
    Args:
        base_params: 基准参数
        param_index: 参数索引
        param_name: 参数名称
        discrete_options: 离散选项
        model: 模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        
    Returns:
        dict: 收率变化结果
    """
    
    base_value = base_params[param_index]
    yields = []
    test_values = []
    
    for option in discrete_options:
        # 创建扰动后的参数
        perturbed_params = base_params.copy()
        perturbed_params[param_index] = option
        
        # 标准化输入
        X_scaled = scaler.transform(perturbed_params.reshape(1, -1))
        
        # 生成多项式特征
        X_poly = poly_features.transform(X_scaled)
        
        # 预测收率
        yield_pred = model.predict(X_poly)[0]
        yields.append(yield_pred)
        test_values.append(option)
    
    return {
        'test_values': test_values,
        'yields': yields,
        'min_yield': min(yields),
        'max_yield': max(yields),
        'yield_range': max(yields) - min(yields)
    }

def calculate_sensitivity(yield_changes):
    """
    计算敏感性指标
    
    Args:
        yield_changes: 收率变化结果
        
    Returns:
        dict: 敏感性指标
    """
    
    max_yield = yield_changes['max_yield']
    min_yield = yield_changes['min_yield']
    yield_range = yield_changes['yield_range']
    
    # 计算相对变化率
    relative_change = yield_range / max_yield if max_yield > 0 else 0
    
    # 判断稳健性等级
    if relative_change < 0.05:
        robustness_level = "高"
    elif relative_change < 0.15:
        robustness_level = "中"
    else:
        robustness_level = "低"
    
    return {
        'relative_change': relative_change,
        'robustness_level': robustness_level
    }

def print_robustness_results(robustness_results):
    """
    打印稳健性分析结果
    
    Args:
        robustness_results: 稳健性分析结果
    """
    
    base_yield = robustness_results['base_yield']
    perturbation_results = robustness_results['perturbation_results']
    
    print(f"\n基准收率: {base_yield:.4f}")
    print("\n参数扰动对收率的影响:")
    print("-" * 80)
    print(f"{'参数':<12} {'最小值':<10} {'最大值':<10} {'变化范围':<10} {'相对变化':<10} {'稳健性':<8}")
    print("-" * 80)
    
    for param_name, result in perturbation_results.items():
        label = result['label']
        yield_changes = result['yield_changes']
        sensitivity = result['sensitivity']
        
        min_yield = yield_changes['min_yield']
        max_yield = yield_changes['max_yield']
        yield_range = yield_changes['yield_range']
        relative_change = sensitivity['relative_change']
        robustness = sensitivity['robustness_level']
        
        print(f"{label:<12} {min_yield:<10.4f} {max_yield:<10.4f} {yield_range:<10.4f} "
              f"{relative_change:<10.2%} {robustness:<8}")
    
    print("-" * 80)
    
    # 总结稳健性
    print("\n稳健性总结:")
    high_robustness = []
    medium_robustness = []
    low_robustness = []
    
    for param_name, result in perturbation_results.items():
        label = result['label']
        robustness = result['sensitivity']['robustness_level']
        
        if robustness == "高":
            high_robustness.append(label)
        elif robustness == "中":
            medium_robustness.append(label)
        else:
            low_robustness.append(label)
    
    if high_robustness:
        print(f"高稳健性参数: {', '.join(high_robustness)}")
    if medium_robustness:
        print(f"中等稳健性参数: {', '.join(medium_robustness)}")
    if low_robustness:
        print(f"低稳健性参数: {', '.join(low_robustness)}")
    
    # 给出建议
    print("\n工艺优化建议:")
    if low_robustness:
        print(f"⚠️  需要严格控制以下参数: {', '.join(low_robustness)}")
        print("   建议在生产过程中对这些参数进行实时监控和精确控制。")
    else:
        print("✅ 所有参数都具有良好的稳健性，工艺条件相对稳定。")
    
    if high_robustness:
        print(f"✅ 以下参数具有高稳健性: {', '.join(high_robustness)}")
        print("   这些参数可以在一定范围内灵活调整，不会显著影响收率。") 