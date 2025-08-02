import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from scipy.stats import f, stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def ilr_transform(compositions):
    """
    等距对数比变换 (Isometric Log-Ratio, ILR)
    消除成分数据的定和约束影响
    
    Parameters:
    compositions: array-like, 成分数据 (各列为不同成分，行为样本)
    
    Returns:
    array: ILR变换后的数据
    """
    compositions = np.array(compositions)
    # 确保数据为正数且和为100（选择性数据）
    compositions = compositions + 1e-10  # 避免零值
    
    n_components = compositions.shape[1]
    n_samples = compositions.shape[0]
    
    # ILR变换
    ilr_data = np.zeros((n_samples, n_components - 1))
    
    for i in range(n_components - 1):
        # ILR_i = sqrt(i/(i+1)) * ln(geometric_mean(x_1,...,x_i) / x_{i+1})
        geometric_mean = np.exp(np.mean(np.log(compositions[:, :i+1]), axis=1))
        ilr_data[:, i] = np.sqrt((i+1)/(i+2)) * np.log(geometric_mean / compositions[:, i+1])
    
    return ilr_data

def inverse_ilr_transform(ilr_data):
    """
    ILR逆变换，将变换后数据转回原始成分空间
    
    Parameters:
    ilr_data: array-like, ILR变换后的数据
    
    Returns:
    array: 原始成分数据
    """
    ilr_data = np.array(ilr_data)
    n_samples, n_ilr = ilr_data.shape
    n_components = n_ilr + 1
    
    # 初始化成分矩阵
    compositions = np.zeros((n_samples, n_components))
    
    # 逆变换计算
    for i in range(n_components):
        if i == 0:
            compositions[:, i] = 1.0
        else:
            compositions[:, i] = compositions[:, i-1] * np.exp(-np.sqrt((i)/(i+1)) * ilr_data[:, i-1])
    
    # 标准化使和为100
    row_sums = np.sum(compositions, axis=1)
    compositions = compositions / row_sums[:, np.newaxis] * 100
    
    return compositions

def quadratic_regression_analysis(T, y, alpha=0.1):
    """
    二次回归分析，包括正则化和统计检验
    """
    if len(T) < 3:
        return {
            'coefficients': [np.nan, np.nan, np.nan],
            'r_squared': np.nan,
            'f_statistic': np.nan,
            'f_p_value': np.nan,
            'quadratic_t_stat': np.nan,
            'quadratic_p_value': np.nan,
            'is_significant': False,
            'has_quadratic_term': False,
            'extremum_temp': np.nan,
            'extremum_value': np.nan,
            'monotonicity': 'unknown',
            'fitted_values': np.full_like(T, np.nan)
        }
    
    # 温度中心化处理：T' = T - 300
    T_centered = T - 300
    
    # 构建设计矩阵 [1, T', T'^2]
    X = np.column_stack([np.ones(len(T_centered)), T_centered, T_centered**2])
    
    # 使用岭回归进行L2正则化
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, y)
    coefficients = ridge.coef_
    
    # 计算拟合值
    fitted_values = X @ coefficients
    
    # 计算R²
    ss_res = np.sum((y - fitted_values) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # F检验（模型整体显著性）
    n = len(y)
    p = 3  # 参数个数
    if n > p:
        msr = ss_tot - ss_res  # 回归平方和
        mse = ss_res / (n - p)  # 均方误差
        f_statistic = (msr / (p - 1)) / mse if mse > 0 else 0
        f_p_value = 1 - f.cdf(f_statistic, p - 1, n - p) if mse > 0 else 1
        is_significant = f_statistic > 19.0  # F_{0.05(2,2)} = 19.0
    else:
        f_statistic = np.nan
        f_p_value = np.nan
        is_significant = False
    
    # 二次项显著性t检验
    if n > p and mse > 0:
        try:
            XtX_inv = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1]))
            var_beta = mse * XtX_inv
            se_beta2 = np.sqrt(var_beta[2, 2]) if var_beta[2, 2] > 0 else np.inf
            
            quadratic_t_stat = coefficients[2] / se_beta2 if se_beta2 > 0 else 0
            quadratic_p_value = 2 * (1 - stats.t.cdf(abs(quadratic_t_stat), n - p))
            has_quadratic_term = quadratic_p_value < 0.2  # 放宽标准
        except:
            quadratic_t_stat = np.nan
            quadratic_p_value = np.nan
            has_quadratic_term = False
    else:
        quadratic_t_stat = np.nan
        quadratic_p_value = np.nan
        has_quadratic_term = False
    
    # 分析极值点和单调性
    extremum_temp = np.nan
    extremum_value = np.nan
    monotonicity = 'unknown'
    
    if has_quadratic_term and abs(coefficients[2]) > 1e-10:
        T_extremum_centered = -coefficients[1] / (2 * coefficients[2])
        extremum_temp = T_extremum_centered + 300
        extremum_value = (coefficients[0] + coefficients[1] * T_extremum_centered + 
                         coefficients[2] * T_extremum_centered**2)
        
        if coefficients[2] > 0:
            if extremum_temp < T.min():
                monotonicity = '递增'
            elif extremum_temp > T.max():
                monotonicity = '递减'
            else:
                monotonicity = 'U形'
        else:
            if extremum_temp < T.min():
                monotonicity = '递减'
            elif extremum_temp > T.max():
                monotonicity = '递增'
            else:
                monotonicity = '倒U形'
    else:
        if abs(coefficients[1]) > 1e-10:
            monotonicity = '递增' if coefficients[1] > 0 else '递减'
        else:
            monotonicity = '常数'
    
    return {
        'coefficients': coefficients,
        'r_squared': r_squared,
        'f_statistic': f_statistic,
        'f_p_value': f_p_value,
        'quadratic_t_stat': quadratic_t_stat,
        'quadratic_p_value': quadratic_p_value,
        'is_significant': is_significant,
        'has_quadratic_term': has_quadratic_term,
        'extremum_temp': extremum_temp,
        'extremum_value': extremum_value,
        'monotonicity': monotonicity,
        'fitted_values': fitted_values
    }

def analyze_catalyst_performance():
    """
    催化剂性能二次拟合分析（含ILR变换）
    """
    # 设置中文字体
    chinese_fonts = ['PingFang HK', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
    font_found = False
    
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            font_found = True
            print(f"成功设置中文字体: {font_name}")
            break
        except:
            continue
    
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    # 加载数据
    try:
        df = pd.read_csv('../附件1.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('附件1.csv')
        except FileNotFoundError:
            print("错误：未找到 '附件1.csv' 文件。")
            return
    
    # 提取产物选择性列（用于ILR变换）
    selectivity_columns = [
        '乙烯选择性（%）', 'C4烯烃选择性(%)', '乙醛选择性(%)', 
        '碳数为4-12脂肪醇选择性(%)', '甲基苯甲醛和甲基苯甲醇选择性(%)', '其他生成物的选择性(%)'
    ]
    
    # 数据分组
    grouped = df.groupby('催化剂组合编号')
    catalyst_names = df['催化剂组合编号'].unique()
    
    # 计算子图布局
    n_catalysts = len(catalyst_names)
    n_cols = 5
    n_rows = int(np.ceil(n_catalysts / n_cols))
    
    # 创建图表
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, n_rows * 4), constrained_layout=True)
    fig.suptitle('催化剂二次拟合分析（含ILR变换处理）', fontsize=20, y=1.02)
    
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    
    axes_flat = axes.flatten()
    
    analysis_results = []
    ilr_results = []
    
    for i, (name, group) in enumerate(grouped):
        ax1 = axes_flat[i]
        
        # 原始数据分析 - 乙醇转化率
        color1 = 'tab:blue'
        ax1.set_xlabel('温度 (°C)', fontsize=10)
        ax1.set_ylabel('乙醇转化率 (%)', color=color1, fontsize=10)
        ax1.plot(group['温度'], group['乙醇转化率(%)'], 'o-', color=color1, label='乙醇转化率', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title(f'催化剂: {name}', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 二次回归分析 - 乙醇转化率
        eth_analysis = quadratic_regression_analysis(group['温度'].values, group['乙醇转化率(%)'].values)
        
        if not np.isnan(eth_analysis['r_squared']):
            T_smooth = np.linspace(group['温度'].min(), group['温度'].max(), 100)
            T_smooth_centered = T_smooth - 300
            fitted_smooth = (eth_analysis['coefficients'][0] + 
                           eth_analysis['coefficients'][1] * T_smooth_centered + 
                           eth_analysis['coefficients'][2] * T_smooth_centered**2)
            ax1.plot(T_smooth, fitted_smooth, '--', color=color1, alpha=0.7, label='二次拟合')
        
        # ILR变换处理选择性数据
        selectivity_data = group[selectivity_columns].values
        ilr_data = ilr_transform(selectivity_data)
        
        # 对C4烯烃选择性（第二列，索引1）进行分析
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('C4烯烃选择性 (%)', color=color2, fontsize=10)
        ax2.plot(group['温度'], group['C4烯烃选择性(%)'], 's--', color=color2, label='C4烯烃选择性', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 二次回归分析 - C4烯烃选择性（原始数据）
        c4_analysis = quadratic_regression_analysis(group['温度'].values, group['C4烯烃选择性(%)'].values)
        
        if not np.isnan(c4_analysis['r_squared']):
            fitted_smooth_c4 = (c4_analysis['coefficients'][0] + 
                               c4_analysis['coefficients'][1] * T_smooth_centered + 
                               c4_analysis['coefficients'][2] * T_smooth_centered**2)
            ax2.plot(T_smooth, fitted_smooth_c4, ':', color=color2, alpha=0.7, label='二次拟合')
        
        # 对ILR变换后的第一个分量进行分析（对应C4烯烃相关的变换）
        ilr_c4_analysis = quadratic_regression_analysis(group['温度'].values, ilr_data[:, 1])  # 第二个ILR分量
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
        
        # 收集分析结果
        analysis_results.append({
            '催化剂组合编号': name,
            '乙醇转化率-β0': eth_analysis['coefficients'][0],
            '乙醇转化率-β1': eth_analysis['coefficients'][1], 
            '乙醇转化率-β2': eth_analysis['coefficients'][2],
            '乙醇转化率-R方': eth_analysis['r_squared'],
            '乙醇转化率-F统计量': eth_analysis['f_statistic'],
            '乙醇转化率-F检验p值': eth_analysis['f_p_value'],
            '乙醇转化率-模型显著': eth_analysis['is_significant'],
            '乙醇转化率-二次项t统计量': eth_analysis['quadratic_t_stat'],
            '乙醇转化率-二次项p值': eth_analysis['quadratic_p_value'],
            '乙醇转化率-保留二次项': eth_analysis['has_quadratic_term'],
            '乙醇转化率-极值温度': eth_analysis['extremum_temp'],
            '乙醇转化率-极值': eth_analysis['extremum_value'],
            '乙醇转化率-单调性': eth_analysis['monotonicity'],
            'C4烯烃选择性-β0': c4_analysis['coefficients'][0],
            'C4烯烃选择性-β1': c4_analysis['coefficients'][1],
            'C4烯烃选择性-β2': c4_analysis['coefficients'][2],
            'C4烯烃选择性-R方': c4_analysis['r_squared'],
            'C4烯烃选择性-F统计量': c4_analysis['f_statistic'],
            'C4烯烃选择性-F检验p值': c4_analysis['f_p_value'],
            'C4烯烃选择性-模型显著': c4_analysis['is_significant'],
            'C4烯烃选择性-二次项t统计量': c4_analysis['quadratic_t_stat'],
            'C4烯烃选择性-二次项p值': c4_analysis['quadratic_p_value'],
            'C4烯烃选择性-保留二次项': c4_analysis['has_quadratic_term'],
            'C4烯烃选择性-极值温度': c4_analysis['extremum_temp'],
            'C4烯烃选择性-极值': c4_analysis['extremum_value'],
            'C4烯烃选择性-单调性': c4_analysis['monotonicity']
        })
        
        # 收集ILR分析结果
        ilr_result = {
            '催化剂组合编号': name,
            'ILR-C4烯烃-β0': ilr_c4_analysis['coefficients'][0],
            'ILR-C4烯烃-β1': ilr_c4_analysis['coefficients'][1],
            'ILR-C4烯烃-β2': ilr_c4_analysis['coefficients'][2],
            'ILR-C4烯烃-R方': ilr_c4_analysis['r_squared'],
            'ILR-C4烯烃-F统计量': ilr_c4_analysis['f_statistic'],
            'ILR-C4烯烃-F检验p值': ilr_c4_analysis['f_p_value'],
            'ILR-C4烯烃-模型显著': ilr_c4_analysis['is_significant'],
            'ILR-C4烯烃-二次项t统计量': ilr_c4_analysis['quadratic_t_stat'],
            'ILR-C4烯烃-二次项p值': ilr_c4_analysis['quadratic_p_value'],
            'ILR-C4烯烃-保留二次项': ilr_c4_analysis['has_quadratic_term'],
            'ILR-C4烯烃-极值温度': ilr_c4_analysis['extremum_temp'],
            'ILR-C4烯烃-极值': ilr_c4_analysis['extremum_value'],
            'ILR-C4烯烃-单调性': ilr_c4_analysis['monotonicity']
        }
        
        # 添加所有ILR分量的数据
        for j in range(ilr_data.shape[1]):
            ilr_result[f'ILR分量{j+1}_温度250'] = ilr_data[0, j] if len(ilr_data) > 0 else np.nan
            ilr_result[f'ILR分量{j+1}_温度275'] = ilr_data[1, j] if len(ilr_data) > 1 else np.nan
            ilr_result[f'ILR分量{j+1}_温度300'] = ilr_data[2, j] if len(ilr_data) > 2 else np.nan
            ilr_result[f'ILR分量{j+1}_温度325'] = ilr_data[3, j] if len(ilr_data) > 3 else np.nan
            ilr_result[f'ILR分量{j+1}_温度350'] = ilr_data[4, j] if len(ilr_data) > 4 else np.nan
        
        ilr_results.append(ilr_result)
    
    # 隐藏多余的空子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    # 保存结果
    plt.savefig('quadratic_analysis_with_ilr.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'quadratic_analysis_with_ilr.png'")
    plt.show()
    
    # 保存分析结果
    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv('quadratic_regression_results.csv', index=False, encoding='utf-8')
    
    ilr_df = pd.DataFrame(ilr_results)
    ilr_df.to_csv('ilr_analysis_results.csv', index=False, encoding='utf-8')
    
    print("原始数据二次回归结果已保存为 'quadratic_regression_results.csv'")
    print("ILR变换分析结果已保存为 'ilr_analysis_results.csv'")

if __name__ == '__main__':
    analyze_catalyst_performance()