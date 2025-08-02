import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- 0. 辅助函数：特征工程 ---
def create_features(df, alr_cols_list):
    """根据输入DataFrame创建用于模型的特征矩阵X"""
    # 替换-inf为0
    df[alr_cols_list] = df[alr_cols_list].replace(-np.inf, 0)
    
    # 创建'Weathering'特征
    df['Weathering'] = (df['表面风化'] == '风化').astype(int)
    
    # 创建交互项
    interaction_features = pd.DataFrame()
    for col in alr_cols_list:
        interaction_features[f'{col}_x_Weathering'] = df[col] * df['Weathering']
        
    # 按训练时的顺序整合所有特征
    base_features_df = df[alr_cols_list + ['Weathering']]
    X = pd.concat([base_features_df, interaction_features], axis=1)
    
    # 确保没有NaN值残留
    X.fillna(0, inplace=True)
    
    return X

# --- 1. 自助法 (Bootstrap) 分析 ---
def run_bootstrap_analysis(df_train, df_test, original_results, alr_cols_list, n_iterations=500):
    """
    通过自助法评估模型对训练数据变化的敏感性。
    """
    print("\n" + "="*20 + " 1. 开始自助法敏感性分析 " + "="*20)
    
    # 准备测试数据 (只需做一次)
    X_test = create_features(df_test.copy(), alr_cols_list)
    
    # 准备训练数据
    df_train_filtered = df_train[df_train['类型'].isin(['高钾', '铅钡'])].copy()
    y_train_full = (df_train_filtered['类型'] == '高钾').astype(int)
    X_train_full = create_features(df_train_filtered.copy(), alr_cols_list)
    
    # 存储每次迭代对测试样本的预测概率
    all_probas = []

    print(f"将进行 {n_iterations} 次自助抽样和重新建模...")
    for i in tqdm(range(n_iterations)):
        # 1. 有放回抽样
        X_sample, y_sample = resample(X_train_full, y_train_full, random_state=i)
        
        # 2. 训练新模型
        scaler = StandardScaler()
        X_sample_scaled = scaler.fit_transform(X_sample)
        
        model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
        model.fit(X_sample_scaled, y_sample)
        
        # 3. 对测试集进行预测
        X_test_scaled = scaler.transform(X_test)
        probas = model.predict_proba(X_test_scaled)[:, 1] # 取判定为高钾的概率
        all_probas.append(probas)
        
    # 4. 分析结果
    probas_array = np.array(all_probas) # Shape: (n_iterations, n_test_samples)
    
    # 计算概率的标准差
    std_devs = np.std(probas_array, axis=0)
    
    # 计算预测一致性
    original_predictions_numeric = (original_results['逻辑回归预测'] == '高钾').astype(int)
    bootstrap_predictions = (probas_array > 0.5).astype(int)
    
    consistency_rate = np.mean(bootstrap_predictions == original_predictions_numeric.values, axis=0)
    
    # 整理成DataFrame
    bootstrap_results = pd.DataFrame({
        '文物编号': original_results['文物编号'],
        '原始预测': original_results['逻辑回归预测'],
        '高钾概率的标准差': std_devs,
        '与原始预测一致率': consistency_rate
    })
    
    print("自助法分析完成。")
    return bootstrap_results

# --- 2. 数据扰动法分析 ---
def run_perturbation_analysis(df_test, original_results, alr_cols_list, model, scaler):
    """
    通过向数据注入噪声来评估模型对输入数据变化的敏感性。
    """
    print("\n" + "="*20 + " 2. 开始数据扰动敏感性分析 " + "="*20)
    
    # 准备原始测试数据和预测
    X_test_original = create_features(df_test.copy(), alr_cols_list)
    original_predictions = (original_results['逻辑回归预测'] == '高钾').astype(int).values
    
    # 定义噪声等级
    epsilons = np.linspace(0, 0.5, 51) # 从0到50%的噪声，步长1%
    max_tolerated_noise = []

    print("正在为每个样本寻找其能容忍的最大噪声水平...")
    for i in tqdm(range(len(X_test_original))):
        sample = X_test_original.iloc[i:i+1]
        original_pred = original_predictions[i]
        
        max_eps = 0.0
        for eps in epsilons:
            if eps == 0:
                continue
            
            # 对特征添加高斯噪声: X_noisy = X * (1 + ε * N(0,1))
            noise = np.random.normal(0, 1, sample.shape)
            sample_noisy = sample * (1 + eps * noise)
            
            # 缩放并预测
            sample_noisy_scaled = scaler.transform(sample_noisy)
            noisy_pred = model.predict(sample_noisy_scaled)[0]
            
            if noisy_pred == original_pred:
                max_eps = eps
            else:
                # 一旦预测改变，就停止增加噪声
                break
        max_tolerated_noise.append(max_eps)

    perturbation_results = pd.DataFrame({
        '文物编号': original_results['文物编号'],
        '原始预测': original_results['逻辑回归预测'],
        '能容忍的最大噪声水平(ε)': max_tolerated_noise
    })
    
    print("数据扰动分析完成。")
    return perturbation_results


# --- 主函数 ---
if __name__ == '__main__':
    # --- 路径定义 ---
    TRAIN_DATA_PATH = '2/附件2_处理后ALR.csv'
    TEST_DATA_PATH = '3/附件3_处理后_ALR.csv'
    ORIGINAL_RESULTS_PATH = '3/prediction_results.csv'
    LR_MODEL_PATH = '2/logistic_regression/logistic_regression_model.joblib'
    LR_SCALER_PATH = '2/logistic_regression/scaler.joblib'

    # --- 加载数据 ---
    print("正在加载所需数据文件...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    original_results = pd.read_csv(ORIGINAL_RESULTS_PATH)
    
    # 识别ALR列 (以训练数据为准)
    alr_cols = [col for col in df_train.columns if col.startswith('ALR_')]

    # --- 执行自助法分析 ---
    bootstrap_report = run_bootstrap_analysis(df_train, df_test, original_results, alr_cols)
    print("\n--- 自助法敏感性分析报告 ---")
    print(bootstrap_report.to_string())

    # --- 执行数据扰动法分析 ---
    # 加载原始模型和缩放器
    lr_model = joblib.load(LR_MODEL_PATH)
    lr_scaler = joblib.load(LR_SCALER_PATH)
    perturbation_report = run_perturbation_analysis(df_test, original_results, alr_cols, lr_model, lr_scaler)
    print("\n--- 数据扰动法敏感性分析报告 ---")
    print(perturbation_report.to_string())

    # --- 保存报告 ---
    output_bootstrap_path = '3/sensitivity_bootstrap_report.csv'
    output_perturbation_path = '3/sensitivity_perturbation_report.csv'
    bootstrap_report.to_csv(output_bootstrap_path, index=False, encoding='utf-8-sig')
    perturbation_report.to_csv(output_perturbation_path, index=False, encoding='utf-8-sig')
    print(f"\n自助法分析报告已保存到: {output_bootstrap_path}")
    print(f"数据扰动法分析报告已保存到: {output_perturbation_path}") 