import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

def ilr_transform(compositions):
    """等距对数比变换 (ILR)"""
    compositions = np.array(compositions)
    compositions = compositions + 1e-10  # 避免零值
    
    n_components = compositions.shape[1]
    n_samples = compositions.shape[0]
    
    ilr_data = np.zeros((n_samples, n_components - 1))
    
    for i in range(n_components - 1):
        geometric_mean = np.exp(np.mean(np.log(compositions[:, :i+1]), axis=1))
        ilr_data[:, i] = np.sqrt((i+1)/(i+2)) * np.log(geometric_mean / compositions[:, i+1])
    
    return ilr_data

def inverse_ilr_transform(ilr_data):
    """ILR逆变换"""
    ilr_data = np.array(ilr_data)
    n_samples, n_ilr = ilr_data.shape
    n_components = n_ilr + 1
    
    compositions = np.zeros((n_samples, n_components))
    
    for i in range(n_components):
        if i == 0:
            compositions[:, i] = 1.0
        else:
            compositions[:, i] = compositions[:, i-1] * np.exp(-np.sqrt((i)/(i+1)) * ilr_data[:, i-1])
    
    # 标准化使和为100
    row_sums = np.sum(compositions, axis=1)
    compositions = compositions / row_sums[:, np.newaxis] * 100
    
    return compositions

def deactivation_model(t, X0, kd):
    """催化剂失活动力学模型: ln(X) = ln(X0) - kd*t"""
    return X0 * np.exp(-kd * t)

def deactivation_linear_model(t, ln_X0, kd):
    """线性形式: ln(X) = ln(X0) - kd*t"""
    return ln_X0 - kd * t

def competition_model_single(t, A, B):
    """单一产物竞争模型: S = A * exp(B*t)"""
    return A * np.exp(B * t)

def analyze_time_series_data():
    """分析附件2的时间序列数据"""
    
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
        df = pd.read_csv('../附件2.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('附件2.csv')
        except FileNotFoundError:
            print("错误：未找到 '附件2.csv' 文件。")
            return
    
    # 清理数据
    df = df.dropna()
    
    # 提取时间和指标数据
    time = df['时间（min）'].values
    conversion = df['乙醇转化率(%)'].values
    
    # 计算C4烯烃收率 = 乙醇转化率 × C4烯烃选择性
    c4_yield = conversion * df['C4烯烃选择性'].values / 100
    
    # 提取选择性数据
    selectivity_columns = ['乙烯选择性', 'C4烯烃选择性', '乙醛选择性', 
                          '碳数为4-12脂肪醇选择性', '甲基苯甲醛和甲基苯甲醇选择性', '其他选择性']
    selectivity_data = df[selectivity_columns].values
    
    print("=== 附件2时间序列数据分析 ===\n")
    
    # =================== 1. 催化剂失活动力学模型 ===================
    print("1. 催化剂失活动力学模型分析")
    
    # 1.1 乙醇转化率失活分析
    try:
        # 非线性拟合
        popt_conv, pcov_conv = curve_fit(deactivation_model, time, conversion, 
                                        p0=[conversion[0], 0.001])
        X0_conv, kd_conv = popt_conv
        
        # 线性拟合 (对数形式)
        log_conv = np.log(conversion)
        popt_log_conv, pcov_log_conv = curve_fit(deactivation_linear_model, time, log_conv,
                                                p0=[log_conv[0], 0.001])
        ln_X0_conv, kd_conv_linear = popt_log_conv
        
        # 计算R²
        conv_fitted = deactivation_model(time, X0_conv, kd_conv)
        ss_res_conv = np.sum((conversion - conv_fitted) ** 2)
        ss_tot_conv = np.sum((conversion - np.mean(conversion)) ** 2)
        r2_conv = 1 - (ss_res_conv / ss_tot_conv)
        
        print(f"   乙醇转化率失活模型:")
        print(f"   - 初始转化率 X₀ = {X0_conv:.2f}%")
        print(f"   - 失活常数 kd = {kd_conv:.6f} min⁻¹")
        print(f"   - R² = {r2_conv:.4f}")
        
    except Exception as e:
        print(f"   乙醇转化率拟合失败: {e}")
        kd_conv = np.nan
    
    # 1.2 C4烯烃收率失活分析
    try:
        popt_yield, pcov_yield = curve_fit(deactivation_model, time, c4_yield,
                                          p0=[c4_yield[0], 0.001])
        X0_yield, kd_yield = popt_yield
        
        # 计算R²
        yield_fitted = deactivation_model(time, X0_yield, kd_yield)
        ss_res_yield = np.sum((c4_yield - yield_fitted) ** 2)
        ss_tot_yield = np.sum((c4_yield - np.mean(c4_yield)) ** 2)
        r2_yield = 1 - (ss_res_yield / ss_tot_yield)
        
        # 计算半衰期 t₁/₂ = ln(2)/kd
        half_life = np.log(2) / kd_yield if kd_yield > 0 else np.inf
        
        print(f"   C4烯烃收率失活模型:")
        print(f"   - 初始收率 X₀ = {X0_yield:.2f}%")
        print(f"   - 失活常数 kd = {kd_yield:.6f} min⁻¹")
        print(f"   - R² = {r2_yield:.4f}")
        print(f"   - 半衰期 t₁/₂ = {half_life:.1f} min")
        print(f"   - 建议再生时机: {half_life*0.8:.1f} min (80%半衰期)")
        
    except Exception as e:
        print(f"   C4烯烃收率拟合失败: {e}")
        kd_yield = np.nan
        half_life = np.nan
    
    # =================== 2. ILR变换处理选择性数据 ===================
    print(f"\n2. ILR变换处理选择性数据")
    
    # 进行ILR变换
    ilr_data = ilr_transform(selectivity_data)
    print(f"   - 原始选择性数据维度: {selectivity_data.shape}")
    print(f"   - ILR变换后维度: {ilr_data.shape}")
    print(f"   - 成功消除定和约束影响")
    
    # =================== 3. 产物选择性竞争模型 ===================
    print(f"\n3. 产物选择性竞争模型分析")
    
    # 选择主要产物进行竞争分析
    main_products = ['C4烯烃选择性', '乙烯选择性', '乙醛选择性']
    main_selectivities = df[main_products].values
    
    competition_results = {}
    
    for i, product in enumerate(main_products):
        try:
            selectivity = main_selectivities[:, i]
            
            # 拟合竞争模型 S = A * exp(B*t)
            popt_comp, pcov_comp = curve_fit(competition_model_single, time, selectivity,
                                           p0=[selectivity[0], 0.0])
            A_comp, B_comp = popt_comp
            
            # 计算R²
            fitted_comp = competition_model_single(time, A_comp, B_comp)
            ss_res_comp = np.sum((selectivity - fitted_comp) ** 2)
            ss_tot_comp = np.sum((selectivity - np.mean(selectivity)) ** 2)
            r2_comp = 1 - (ss_res_comp / ss_tot_comp)
            
            competition_results[product] = {
                'A': A_comp,
                'B': B_comp,
                'R2': r2_comp,
                'fitted': fitted_comp
            }
            
            trend = "增强" if B_comp > 0 else "减弱" if B_comp < 0 else "稳定"
            print(f"   {product}:")
            print(f"   - 初始选择性 A = {A_comp:.2f}%")
            print(f"   - 竞争系数 B = {B_comp:.6f} min⁻¹ ({trend})")
            print(f"   - R² = {r2_comp:.4f}")
            
        except Exception as e:
            print(f"   {product} 拟合失败: {e}")
            competition_results[product] = None
    
    # =================== 4. 可视化结果 ===================
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('350°C催化剂时间序列分析', fontsize=16)
    
    # 4.1 失活动力学图
    ax1 = axes[0, 0]
    ax1.scatter(time, conversion, color='blue', label='实验数据', s=50)
    if not np.isnan(kd_conv):
        time_smooth = np.linspace(time.min(), time.max(), 100)
        conv_smooth = deactivation_model(time_smooth, X0_conv, kd_conv)
        ax1.plot(time_smooth, conv_smooth, 'b-', label=f'失活模型 (kd={kd_conv:.6f})')
    ax1.set_xlabel('时间 (min)')
    ax1.set_ylabel('乙醇转化率 (%)')
    ax1.set_title('乙醇转化率失活动力学')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 4.2 C4收率失活图
    ax2 = axes[0, 1]
    ax2.scatter(time, c4_yield, color='red', label='实验数据', s=50)
    if not np.isnan(kd_yield):
        yield_smooth = deactivation_model(time_smooth, X0_yield, kd_yield)
        ax2.plot(time_smooth, yield_smooth, 'r-', label=f'失活模型 (kd={kd_yield:.6f})')
        ax2.axhline(y=X0_yield/2, color='orange', linestyle='--', alpha=0.7, label=f'半衰期={half_life:.1f}min')
    ax2.set_xlabel('时间 (min)')
    ax2.set_ylabel('C4烯烃收率 (%)')
    ax2.set_title('C4烯烃收率失活动力学')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 4.3 产物选择性竞争图
    ax3 = axes[1, 0]
    colors = ['green', 'purple', 'orange']
    for i, (product, color) in enumerate(zip(main_products, colors)):
        selectivity = main_selectivities[:, i]
        ax3.scatter(time, selectivity, color=color, label=f'{product} (实验)', s=40)
        
        if competition_results[product] is not None:
            fitted = competition_results[product]['fitted']
            ax3.plot(time, fitted, color=color, linestyle='-', alpha=0.7,
                    label=f'{product} (模型)')
    
    ax3.set_xlabel('时间 (min)')
    ax3.set_ylabel('选择性 (%)')
    ax3.set_title('产物选择性竞争模型')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4.4 ILR变换数据图
    ax4 = axes[1, 1]
    for i in range(ilr_data.shape[1]):
        ax4.plot(time, ilr_data[:, i], 'o-', label=f'ILR分量{i+1}', markersize=4)
    ax4.set_xlabel('时间 (min)')
    ax4.set_ylabel('ILR变换值')
    ax4.set_title('ILR变换后的选择性数据')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存为 'time_series_analysis.png'")
    plt.show()
    
    # =================== 5. 保存结果数据 ===================
    
    # 失活动力学结果
    deactivation_results = {
        '指标': ['乙醇转化率', 'C4烯烃收率'],
        '初始值_X0': [X0_conv if not np.isnan(kd_conv) else np.nan, 
                     X0_yield if not np.isnan(kd_yield) else np.nan],
        '失活常数_kd': [kd_conv, kd_yield],
        'R平方': [r2_conv if not np.isnan(kd_conv) else np.nan,
                r2_yield if not np.isnan(kd_yield) else np.nan],
        '半衰期_min': [np.log(2)/kd_conv if not np.isnan(kd_conv) and kd_conv > 0 else np.nan,
                     half_life if not np.isnan(half_life) else np.nan]
    }
    
    deactivation_df = pd.DataFrame(deactivation_results)
    deactivation_df.to_csv('deactivation_kinetics_results.csv', index=False, encoding='utf-8')
    
    # 竞争模型结果
    competition_data = []
    for product in main_products:
        if competition_results[product] is not None:
            result = competition_results[product]
            competition_data.append({
                '产物': product,
                '初始选择性_A': result['A'],
                '竞争系数_B': result['B'],
                'R平方': result['R2'],
                '趋势': "增强" if result['B'] > 0 else "减弱" if result['B'] < 0 else "稳定"
            })
    
    competition_df = pd.DataFrame(competition_data)
    competition_df.to_csv('competition_model_results.csv', index=False, encoding='utf-8')
    
    # ILR变换数据
    ilr_df = pd.DataFrame(ilr_data, columns=[f'ILR分量{i+1}' for i in range(ilr_data.shape[1])])
    ilr_df['时间'] = time
    ilr_df = ilr_df[['时间'] + [col for col in ilr_df.columns if col != '时间']]
    ilr_df.to_csv('ilr_transformed_data.csv', index=False, encoding='utf-8')
    
    print(f"\n分析结果已保存:")
    print(f"- deactivation_kinetics_results.csv: 失活动力学模型结果")
    print(f"- competition_model_results.csv: 产物竞争模型结果") 
    print(f"- ilr_transformed_data.csv: ILR变换后的数据")
    
    # =================== 6. 分析结论 ===================
    print(f"\n=== 分析结论 ===")
    print(f"1. 催化剂失活特征:")
    if not np.isnan(kd_conv) and not np.isnan(kd_yield):
        print(f"   - 乙醇转化率失活速率: {kd_conv:.6f} min⁻¹")
        print(f"   - C4烯烃收率失活速率: {kd_yield:.6f} min⁻¹")
        print(f"   - 建议催化剂再生时机: {half_life*0.8:.1f} min")
    
    print(f"2. 产物竞争关系:")
    for product in main_products:
        if competition_results[product] is not None:
            result = competition_results[product]
            trend = "增强" if result['B'] > 0 else "减弱" if result['B'] < 0 else "稳定"
            print(f"   - {product}: {trend} (B={result['B']:.6f})")
    
    print(f"3. ILR变换效果:")
    print(f"   - 成功消除选择性数据的定和约束")
    print(f"   - 转换{selectivity_data.shape[1]}维选择性数据为{ilr_data.shape[1]}维欧氏空间数据")

if __name__ == '__main__':
    analyze_time_series_data()