import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from scipy import stats
from scipy.stats import f, t
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
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

def optimize_alpha_loocv(T, y):
    """
    使用留一交叉验证优化正则化参数alpha
    适用于小样本数据（3-5个数据点）
    
    Parameters:
    T: array-like, 温度数据
    y: array-like, 响应变量数据
    
    Returns:
    float: 最优的alpha值
    """
    if len(T) < 3:
        return 0.1  # 样本太少，使用默认值
    
    # 测试alpha范围：从0.001到100的对数空间
    alphas = np.logspace(-3, 2, 20)
    best_alpha = 0.1
    best_mse = float('inf')
    
    # 温度中心化
    T_center = np.mean(T)
    T_centered = T - T_center
    X = np.column_stack([np.ones(len(T_centered)), T_centered, T_centered**2])
    
    loo = LeaveOneOut()
    
    for alpha in alphas:
        mse_scores = []
        
        try:
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                ridge = Ridge(alpha=alpha, fit_intercept=False)
                ridge.fit(X_train, y_train)
                y_pred = ridge.predict(X_test)
                
                mse = (y_pred[0] - y_test[0])**2
                mse_scores.append(mse)
            
            avg_mse = np.mean(mse_scores)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = alpha
                
        except Exception as e:
            # 如果某个alpha值导致数值问题，跳过
            continue
    
    return best_alpha

def quadratic_regression_analysis(T, y, alpha=None):
    """
    二次回归分析，包括正则化和统计检验
    
    Parameters:
    T: array-like, 温度数据
    y: array-like, 响应变量数据
    alpha: float or None, 正则化参数。如果为None，则自动优化
    
    Returns:
    dict: 包含回归分析结果的字典
    """
    # 如果alpha未指定，则使用LOOCV自动优化
    if alpha is None:
        alpha = optimize_alpha_loocv(T, y)
    
    if len(T) < 3:
        # 当样本数不足时，进行简单线性回归
        if len(T) == 2:
            T_center = np.mean(T)
            T_centered = T - T_center
            # 简单线性回归
            slope = np.sum((T_centered) * (y - np.mean(y))) / np.sum(T_centered**2) if np.sum(T_centered**2) > 0 else 0
            intercept = np.mean(y)
            fitted_values = intercept + slope * T_centered
            ss_res = np.sum((y - fitted_values) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            monotonicity = '递增' if slope > 0 else ('递减' if slope < 0 else '常数')
            
            return {
                'coefficients': [intercept, slope, 0.0],
                'r_squared': r_squared,
                'f_statistic': 0.0,
                'f_p_value': 1.0,
                'quadratic_t_stat': 0.0,
                'quadratic_p_value': 1.0,
                'is_significant': False,
                'has_quadratic_term': False,
                'extremum_temp': 0.0,
                'extremum_value': 0.0,
                'monotonicity': monotonicity,
                'fitted_values': fitted_values,
                'center_point': T_center,
                'optimal_alpha': alpha
            }
        else:
            # 只有一个数据点
            return {
                'coefficients': [np.mean(y), 0.0, 0.0],
                'r_squared': 0.0,
                'f_statistic': 0.0,
                'f_p_value': 1.0,
                'quadratic_t_stat': 0.0,
                'quadratic_p_value': 1.0,
                'is_significant': False,
                'has_quadratic_term': False,
                'extremum_temp': 0.0,
                'extremum_value': 0.0,
                'monotonicity': '常数',
                'fitted_values': np.full_like(T, np.mean(y)),
                'center_point': np.mean(T),
                'optimal_alpha': alpha
            }
    
    # 动态计算温度中心化点：使用温度数据的均值
    T_center = np.mean(T)
    T_centered = T - T_center
    
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
        f_statistic = 0.0
        f_p_value = 1.0
        is_significant = False
    
    # 二次项显著性t检验
    if n > p and mse > 0:
        try:
            XtX_inv = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1]))
            var_beta = mse * XtX_inv
            se_beta2 = np.sqrt(var_beta[2, 2]) if var_beta[2, 2] > 0 else np.inf
            
            if se_beta2 > 0 and se_beta2 != np.inf:
                quadratic_t_stat = coefficients[2] / se_beta2
                quadratic_p_value = 2 * (1 - t.cdf(abs(quadratic_t_stat), n - p))
                has_quadratic_term = quadratic_p_value < 0.2  # 放宽标准
            else:
                quadratic_t_stat = 0.0
                quadratic_p_value = 1.0
                has_quadratic_term = False
        except Exception as e:
            print(f"警告：二次项t检验计算失败: {e}")
            quadratic_t_stat = 0.0
            quadratic_p_value = 1.0
            has_quadratic_term = False
    else:
        quadratic_t_stat = 0.0
        quadratic_p_value = 1.0
        has_quadratic_term = False
    
    # 分析极值点和单调性
    extremum_temp = np.nan
    extremum_value = np.nan
    monotonicity = 'unknown'
    
    if has_quadratic_term and abs(coefficients[2]) > 1e-10:
        T_extremum_centered = -coefficients[1] / (2 * coefficients[2])
        extremum_temp = T_extremum_centered + T_center
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
        'fitted_values': fitted_values,
        'center_point': T_center,
        'optimal_alpha': alpha
    }

def analyze_catalyst_performance():
    """
    催化剂性能二次拟合分析
    - 乙醇转化率：直接分析（无约束数据）
    - C4烯烃选择性：ILR变换分析（成分数据，存在定和约束）
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
        df = pd.read_csv('/Users/Mac/Downloads/Math-Modeling-Practice/21B/附件1.csv')
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
    fig.suptitle('催化剂性能分析：乙醇转化率(直接分析) vs C4烯烃选择性(ILR变换)', fontsize=18, y=1.02)
    
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    
    axes_flat = axes.flatten()
    
    analysis_results = []
    
    for i, (name, group) in enumerate(grouped):
        ax1 = axes_flat[i]
        
        # 1. 乙醇转化率分析（直接分析，无约束）
        color1 = 'tab:blue'
        ax1.set_xlabel('温度 (°C)', fontsize=10)
        ax1.set_ylabel('乙醇转化率 (%)', color=color1, fontsize=10)
        ax1.plot(group['温度'], group['乙醇转化率(%)'], 'o-', color=color1, label='乙醇转化率', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title(f'催化剂: {name}', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 乙醇转化率二次回归分析
        eth_analysis = quadratic_regression_analysis(group['温度'].values, group['乙醇转化率(%)'].values)
        
        if not np.isnan(eth_analysis['r_squared']):
            T_smooth = np.linspace(group['温度'].min(), group['温度'].max(), 100)
            T_smooth_centered = T_smooth - eth_analysis['center_point']
            fitted_smooth = (eth_analysis['coefficients'][0] + 
                           eth_analysis['coefficients'][1] * T_smooth_centered + 
                           eth_analysis['coefficients'][2] * T_smooth_centered**2)
            ax1.plot(T_smooth, fitted_smooth, '--', color=color1, alpha=0.7, label='乙醇转化率拟合')
        
        # 2. C4烯烃选择性ILR变换分析
        selectivity_data = group[selectivity_columns].values
        try:
            # 检查数据有效性
            if np.any(selectivity_data <= 0):
                print(f"警告：催化剂 {name} 的选择性数据包含零值或负值，将进行处理")
                selectivity_data = np.maximum(selectivity_data, 1e-10)
            
            ilr_data = ilr_transform(selectivity_data)
            
            # 检查ILR变换结果
            if np.any(np.isnan(ilr_data)) or np.any(np.isinf(ilr_data)):
                print(f"警告：催化剂 {name} 的ILR变换产生了无效值")
                ilr_data = np.nan_to_num(ilr_data, nan=0.0, posinf=1.0, neginf=-1.0)
                
        except Exception as e:
            print(f"错误：催化剂 {name} 的ILR变换失败: {e}")
            ilr_data = np.zeros((len(selectivity_data), len(selectivity_columns)-1))
        
        # 创建右轴显示ILR变换后的C4烯烃相关分量
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('ILR-C4烯烃分量', color=color2, fontsize=10)
        ax2.plot(group['温度'], ilr_data[:, 1], 's--', color=color2, label='ILR-C4烯烃分量', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 对ILR变换后的C4烯烃分量进行二次回归分析
        ilr_c4_analysis = quadratic_regression_analysis(group['温度'].values, ilr_data[:, 1])
        
        if not np.isnan(ilr_c4_analysis['r_squared']):
            T_smooth_centered_ilr = T_smooth - ilr_c4_analysis['center_point']
            fitted_smooth_ilr = (ilr_c4_analysis['coefficients'][0] + 
                                ilr_c4_analysis['coefficients'][1] * T_smooth_centered_ilr + 
                                ilr_c4_analysis['coefficients'][2] * T_smooth_centered_ilr**2)
            ax2.plot(T_smooth, fitted_smooth_ilr, ':', color=color2, alpha=0.7, label='ILR-C4拟合')
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
        
        # 收集分析结果
        analysis_results.append({
            '催化剂组合编号': name,
            '温度范围': f"{group['温度'].min():.1f}-{group['温度'].max():.1f}°C",
            '温度中心点': f"{eth_analysis['center_point']:.1f}°C",
            # 乙醇转化率分析（直接分析）
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
            '乙醇转化率-最优alpha': eth_analysis['optimal_alpha'],
            # ILR-C4烯烃分量分析（处理定和约束）
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
            'ILR-C4烯烃-单调性': ilr_c4_analysis['monotonicity'],
            'ILR-C4烯烃-最优alpha': ilr_c4_analysis['optimal_alpha'],
            # 添加ILR分量的原始数值（动态获取温度点）
            **{f'ILR-C4分量_{temp:.0f}C': ilr_data[idx, 1] if idx < len(ilr_data) else np.nan 
               for idx, temp in enumerate(sorted(group['温度'].values))},
        })
    
    # 隐藏多余的空子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    # 保存结果
    plt.savefig('catalyst_analysis_optimized.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'catalyst_analysis_optimized.png'")
    plt.show()
    
    # 保存分析结果
    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv('catalyst_analysis_optimized_results.csv', index=False, encoding='utf-8')
    
    print("优化后的催化剂分析结果已保存为 'catalyst_analysis_optimized_results.csv'")
    
    # 显示分析总结
    print("\n=== 分析方法总结 ===")
    print("1. 乙醇转化率：直接二次回归分析（无约束数据）")
    print("2. C4烯烃选择性：ILR变换后二次回归分析（处理定和约束）")
    print("3. ILR变换：消除成分数据间的相关性和定和约束影响")
    print("4. 正则化优化：使用留一交叉验证(LOOCV)自动优化alpha参数")
    print("5. 小样本适应：针对3-5个数据点的专门优化策略")
    
    print("\n=== 各催化剂分析结果 ===")
    for result in analysis_results:
        print(f"催化剂 {result['催化剂组合编号']}:")
        print(f"  温度范围: {result['温度范围']}")
        print(f"  中心化点: {result['温度中心点']}")
        print(f"  乙醇转化率单调性: {result['乙醇转化率-单调性']} (α={result['乙醇转化率-最优alpha']:.4f})")
        print(f"  ILR-C4烯烃单调性: {result['ILR-C4烯烃-单调性']} (α={result['ILR-C4烯烃-最优alpha']:.4f})")
        if not np.isnan(result['乙醇转化率-极值温度']):
            print(f"  乙醇转化率极值温度: {result['乙醇转化率-极值温度']:.1f}°C")
        if not np.isnan(result['ILR-C4烯烃-极值温度']):
            print(f"  ILR-C4烯烃极值温度: {result['ILR-C4烯烃-极值温度']:.1f}°C")
        print()

if __name__ == '__main__':
    analyze_catalyst_performance()