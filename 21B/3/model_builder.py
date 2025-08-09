import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def build_and_train_rsm(dataframe):
    """
    构建并训练响应面模型（RSM）
    
    功能：
    1. 分离自变量和目标变量
    2. 数据标准化
    3. 生成多项式特征
    4. 使用RidgeCV进行正则化回归拟合
    5. 模型评估
    
    Args:
        dataframe: 预处理后的数据
        
    Returns:
        tuple: (训练好的模型, 多项式特征对象, 标准化对象)
    """
    
    print("开始构建响应面模型...")
    
    # 1. 分离自变量和目标变量
    feature_columns = ['T', 'total_mass', 'loading_ratio', 'C', 'C_e']
    X = dataframe[feature_columns].values
    y = dataframe['Y'].values
    
    print(f"自变量矩阵形状: {X.shape}")
    print(f"目标变量向量形状: {y.shape}")
    
    # 2. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 生成多项式特征（二次项和交叉项）
    print("正在生成多项式特征...")
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_scaled)
    
    print(f"多项式特征矩阵形状: {X_poly.shape}")
    
    # 4. 使用RidgeCV进行模型拟合与自动正则化
    print("正在使用RidgeCV进行模型拟合与正则化...")
    
    # 定义一系列alpha值（正则化强度）进行测试
    alphas = np.logspace(-6, 6, 13)
    
    # 使用RidgeCV，它会自动进行交叉验证来选择最佳alpha
    # cv=5 表示5折交叉验证
    ridge_cv_model = RidgeCV(alphas=alphas, cv=5)
    ridge_cv_model.fit(X_poly, y)
    
    # 5. 评估最终模型
    print(f"✓ RidgeCV模型构建完成")
    print(f"✓ 最佳正则化强度 (alpha): {ridge_cv_model.alpha_:.4f}")
    
    # 计算并打印R²
    r2_final = ridge_cv_model.score(X_poly, y)
    print(f"模型在训练集上的R²: {r2_final:.4f}")
    
    # 交叉验证评估
    print(f"交叉验证R²: {ridge_cv_model.best_score_:.4f}")
    
    # 6. 模型诊断
    print("\n模型诊断:")
    y_pred = ridge_cv_model.predict(X_poly)
    residuals = y - y_pred
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {np.mean(np.abs(residuals)):.4f}")
    
    # 7. 特征重要性分析
    print("\n特征重要性分析:")
    feature_names = poly_features.get_feature_names_out(feature_columns)
    coefficients = ridge_cv_model.coef_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("前10个最重要的特征:")
    print(feature_importance.head(10))
    
    # 8. 模型稳定性分析
    print("\n模型稳定性分析:")
    cv_scores = cross_val_score(ridge_cv_model, X_poly, y, cv=5, scoring='r2')
    print(f"独立交叉验证R² (5折): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 评估模型可靠性
    if ridge_cv_model.best_score_ > 0.6:
        print("✅ 模型具有良好的泛化能力")
    elif ridge_cv_model.best_score_ > 0.4:
        print("⚠️  模型泛化能力一般，建议谨慎使用")
    else:
        print("❌ 模型泛化能力不足，需要重新设计")
    
    # 9. 保存模型参数到文件
    save_model_parameters(ridge_cv_model, poly_features, feature_columns)
    
    print("\n响应面模型构建完成！")
    
    return ridge_cv_model, poly_features, scaler

def predict_yield(model, poly_features, scaler, X_input):
    """
    使用训练好的模型预测C4烯烃收率
    
    Args:
        model: 训练好的模型
        poly_features: 多项式特征对象
        scaler: 标准化对象
        X_input: 输入变量 [T, total_mass, loading_ratio, C, C_e]
        
    Returns:
        float: 预测的C4烯烃收率
    """
    # 标准化输入
    X_scaled = scaler.transform(X_input.reshape(1, -1))
    
    # 生成多项式特征
    X_poly = poly_features.transform(X_scaled)
    
    # 预测
    prediction = model.predict(X_poly)
    
    return prediction[0]

def save_model_parameters(model, poly_features, feature_columns, filename="model_equation.txt"):
    """
    将训练好的Ridge模型的完整方程保存到文本文件中。

    Args:
        model: 训练好的RidgeCV模型。
        poly_features: 训练时使用的PolynomialFeatures对象。
        feature_columns: 原始特征的名称列表。
        filename: 保存参数的文件名。
    """
    print(f"正在将模型方程保存到 {filename}...")
    
    # 获取多项式特征的名称
    poly_feature_names = poly_features.get_feature_names_out(feature_columns)
    
    # 获取模型的系数和截距
    coefficients = model.coef_
    intercept = model.intercept_
    
    # 构建方程字符串
    equation = f"Y = {intercept:.4f} \\\n"
    for coef, name in zip(coefficients, poly_feature_names):
        if coef >= 0:
            equation += f"    + {abs(coef):.4f} * ({name}) \\\n"
        else:
            equation += f"    - {abs(coef):.4f} * ({name}) \\\n"
            
    # 移除最后的 " \"
    equation = equation.rstrip(' \\\n')
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("响应面模型 (RidgeCV) 方程:\n")
            f.write("="*50 + "\n")
            f.write("Y 代表预测的C4烯烃收率。\n")
            f.write("所有特征（T, total_mass等）在代入方程前都经过了标准化处理。\n")
            f.write("="*50 + "\n\n")
            f.write(equation)
        print(f"✓ 模型方程已成功保存到 {filename}")
    except Exception as e:
        print(f"错误：保存模型参数失败 - {e}") 