import pandas as pd
import numpy as np
import joblib
import os

def classify_artifacts():
    """
    加载在问题二中训练的逻辑回归和决策树模型，
    对问题三中的未知类别玻璃文物进行分类，并输出结果。
    """
    # --- 1. 定义文件路径 ---
    data_path = '3/附件3_处理后_ALR.csv'
    lr_model_path = '2/logistic_regression/logistic_regression_model.joblib'
    lr_scaler_path = '2/logistic_regression/scaler.joblib'
    dt_model_path = '2/decision_tree/decision_tree_model.joblib'
    dt_scaler_path = '2/decision_tree/scaler.joblib'
    output_path = '3/prediction_results.csv'
    
    # --- 2. 加载数据 ---
    print(f"正在从 {data_path} 加载数据...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误：未找到数据文件 {data_path}。请确保文件存在。")
        return
    
    # 保存文物编号供最后输出
    artifact_ids = df['文物编号']
    
    # --- 3. 数据预处理与特征工程 ---
    print("正在进行数据预处理和特征工程...")
    
    # 识别ALR特征列
    alr_cols = [col for col in df.columns if col.startswith('ALR_')]
    
    # 处理-inf值，替换为0 (与训练过程中的NaN处理逻辑保持一致)
    df[alr_cols] = df[alr_cols].replace(-np.inf, 0)
    
    # 创建 'Weathering' (风化) 特征
    # '风化' -> 1, '无风化' -> 0
    df['Weathering'] = (df['表面风化'] == '风化').astype(int)
    
    # 创建交互项: 化学成分 * 风化
    interaction_features = pd.DataFrame()
    for col in alr_cols:
        interaction_features[f'{col}_x_Weathering'] = df[col] * df['Weathering']
        
    # 确定特征矩阵X的列顺序，必须与训练时完全一致
    # 从逻辑回归的训练脚本中可知，特征顺序是：alr_cols, Weathering, interaction_cols
    base_features_df = df[alr_cols + ['Weathering']]
    X = pd.concat([base_features_df, interaction_features], axis=1)

    print("特征工程完成。特征矩阵包含ALR特征、风化特征及交互项。")
    print(f"特征数量: {X.shape[1]}")

    # --- 4. 加载模型和缩放器 ---
    print("正在加载模型和数据缩放器...")
    try:
        lr_model = joblib.load(lr_model_path)
        lr_scaler = joblib.load(lr_scaler_path)
        dt_model = joblib.load(dt_model_path)
        dt_scaler = joblib.load(dt_scaler_path)
    except FileNotFoundError as e:
        print(f"错误：加载模型文件失败 - {e}。请确保路径正确且文件存在。")
        return
        
    # --- 5. 进行预测 ---
    print("正在使用加载的模型进行预测...")
    
    # 5.1. 逻辑回归预测
    X_scaled_lr = lr_scaler.transform(X)
    lr_predictions = lr_model.predict(X_scaled_lr)
    lr_probabilities = lr_model.predict_proba(X_scaled_lr)[:, 1] # 获取类别为1(高钾)的概率
    
    # 5.2. 决策树预测
    X_scaled_dt = dt_scaler.transform(X)
    dt_predictions = dt_model.predict(X_scaled_dt)

    # --- 6. 整理并保存结果 ---
    print("正在整理并保存预测结果...")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '文物编号': artifact_ids,
        '逻辑回归预测': lr_predictions,
        '逻辑回归预测为高钾的概率': lr_probabilities,
        '决策树预测': dt_predictions
    })
    
    # 将预测结果 (0/1) 映射回文本标签
    class_map = {0: '铅钡', 1: '高钾'}
    results_df['逻辑回归预测'] = results_df['逻辑回归预测'].map(class_map)
    results_df['决策树预测'] = results_df['决策树预测'].map(class_map)
    
    # 保存到CSV
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("预测完成！")
    print(f"结果已保存到: {output_path}")
    print("="*50)
    
    # 打印结果预览
    print("\n预测结果预览:")
    print(results_df)

if __name__ == '__main__':
    classify_artifacts() 