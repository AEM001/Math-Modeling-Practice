import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def set_chinese_font():
    """
    设置中文字体，以便在图表中正确显示中文。
    会尝试多种常见的的中文字体。
    """
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
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Fallback font
        plt.rcParams['axes.unicode_minus'] = False
        print("未找到指定中文字体，使用默认字体。")

def ilr_transform(compositions):
    """
    等距对数比变换 (Isometric Log-Ratio, ILR)
    消除成分数据的定和约束影响。
    """
    compositions = np.array(compositions)
    compositions = compositions + 1e-10  # 避免零值
    
    n_components = compositions.shape[1]
    n_samples = compositions.shape[0]
    
    ilr_data = np.zeros((n_samples, n_components - 1))
    
    for i in range(n_components - 1):
        geometric_mean = np.exp(np.mean(np.log(compositions[:, :i+1]), axis=1))
        ilr_data[:, i] = np.sqrt((i + 1) / (i + 2)) * np.log(geometric_mean / compositions[:, i + 1])
        
    return ilr_data

def main():
    """
    主函数，用于数据加载、预处理、变换和可视化。
    """
    # 设置字体
    set_chinese_font()
    
    # 1. 加载数据
    try:
        df_perf = pd.read_csv('/Users/Mac/Downloads/Math-Modeling-Practice/21B/附件1.csv')
        df_comp = pd.read_csv('/Users/Mac/Downloads/Math-Modeling-Practice/21B/2/每组指标.csv')
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 - {e}")
        return

    # 2. 合并数据
    df = pd.merge(df_perf, df_comp, on='催化剂组合编号')

    # 3. 数据预处理
    # 剔除A11异常数据
    df = df[df['催化剂组合编号'] != 'A11'].copy()
    
    # 将用量列中的'mg'和'无'替换，并转换为数值
    for col in ['Co/SiO2用量', 'HAP用量']:
        df[col] = df[col].astype(str).str.replace('mg', '').str.replace('无', '0').astype(float)

    # 将'Co负载量'从字符串（如 '1wt%'）转换为数值
    df['Co负载量(wt%)'] = df['Co负载量'].str.replace('wt%', '').astype(float)
    
    # 清理乙醇浓度列
    df['乙醇浓度'] = df['乙醇浓度'].str.replace('ml/min', '').astype(float)
    
    # 计算装料比 (Co/SiO2 / HAP)，处理HAP用量可能为0的情况
    # 对于HAP用量为0的，装料比设为0，表示没有HAP
    df['装料比'] = np.where(df['HAP用量'] > 0, df['Co/SiO2用量'] / df['HAP用量'], 0)

    # 4. ILR变换
    selectivity_columns = [
        '乙烯选择性（%）', 'C4烯烃选择性(%)', '乙醛选择性(%)', 
        '碳数为4-12脂肪醇选择性(%)', '甲基苯甲醛和甲基苯甲醇选择性(%)', '其他生成物的选择性(%)'
    ]
    # 确保用于变换的数据无缺失值
    df.dropna(subset=selectivity_columns, inplace=True)
    
    selectivity_data = df[selectivity_columns].values
    ilr_data = ilr_transform(selectivity_data)
    
    # C4烯烃选择性对应ILR变换后的第2个分量（索引为1）
    df['ILR_C4烯烃选择性'] = ilr_data[:, 1]
    
    # 5. 可视化
    # 准备用于可视化的数据
    features_to_visualize = [
        '温度', 
        'Co负载量(wt%)', 
        '装料比', 
        '乙醇浓度',
        '乙醇转化率(%)', 
        'ILR_C4烯烃选择性'
    ]
    df_vis = df[features_to_visualize]

    # 生成并保存配对图
    print("正在生成配对图 (Pair Plot)...")
    pair_plot = sns.pairplot(df_vis, diag_kind='kde')
    pair_plot.fig.suptitle('各变量间关系探索 (Pair Plot)', y=1.02, fontsize=16)
    plt.savefig('preliminary_pair_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("配对图已保存为 'preliminary_pair_plot.png'")

    # 生成并保存相关性热力图
    print("正在生成相关性热力图 (Correlation Heatmap)...")
    plt.figure(figsize=(10, 8))
    corr_matrix = df_vis.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('特征与目标变量相关性热力图', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('preliminary_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("相关性热力图已保存为 'preliminary_correlation_heatmap.png'")

if __name__ == '__main__':
    main()
