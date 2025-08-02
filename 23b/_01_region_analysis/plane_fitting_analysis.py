import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def fit_plane_to_regions(data_path='output.csv'):
    """
    对划分好的矩形区域进行平面拟合，计算地形参数并评估拟合优度。
    """
    print("--- 开始平面拟合、参数计算与优度检验 ---")

    # 1. 从报告中定义的区域
    regions = [
        {'id': 0, 'x_range': (0.00, 0.98), 'y_range': (0.00, 2.49)},
        {'id': 1, 'x_range': (0.98, 1.99), 'y_range': (0.00, 2.49)},
        {'id': 2, 'x_range': (0.00, 1.99), 'y_range': (2.49, 5.00)},
        {'id': 3, 'x_range': (1.99, 2.99), 'y_range': (0.00, 2.49)},
        {'id': 4, 'x_range': (2.99, 4.00), 'y_range': (0.00, 2.49)},
        {'id': 5, 'x_range': (1.99, 2.99), 'y_range': (2.49, 5.00)},
        {'id': 6, 'x_range': (2.99, 4.00), 'y_range': (2.49, 5.00)},
    ]

    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path)
    results = []

    for region in regions:
        x_min, x_max = region['x_range']
        y_min, y_max = region['y_range']
        
        sub_df = df[
            (df['横坐标'] >= x_min) & (df['横坐标'] <= x_max) &
            (df['纵坐标'] >= y_min) & (df['纵坐标'] <= y_max)
        ]
        
        if len(sub_df) < 3:
            print(f"警告: 区域 {region['id']} 数据点不足 ({len(sub_df)}个)，跳过。")
            continue

        X = sub_df[['横坐标', '纵坐标']]
        y = sub_df['深度']
        
        model = LinearRegression()
        model.fit(X, y)
        
        beta_0 = model.intercept_
        beta_1, beta_2 = model.coef_
        
        # 计算地形参数
        alpha = np.degrees(np.arccos(1 / np.sqrt(1 + beta_1**2 + beta_2**2)))
        phi = np.degrees(np.arctan2(-beta_2, -beta_1))

        # 检验拟合优度
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        results.append({
            '区域编号': region['id'],
            '数据点数量': len(sub_df),
            'beta_0 (截距)': beta_0,
            'beta_1 (x系数)': beta_1,
            'beta_2 (y系数)': beta_2,
            '坡度 (度)': alpha,
            '坡向 (度)': phi,
            'R2_Score': r2,
            'RMSE_m': rmse,
        })
        print(f"区域 {region['id']} 计算完成 (R²: {r2:.3f}, RMSE: {rmse:.3f}m)。")

    results_df = pd.DataFrame(results)
    output_csv_path = 'plane_fitting_results.csv'
    results_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"\n详细计算结果已保存至: '{output_csv_path}'")
    
    return results_df

def append_to_markdown_report(results_df, report_file='region_division_report.md'):
    """将平面拟合的结果和拟合优度追加到Markdown报告中。"""
    print(f"正在更新报告文件: '{report_file}'")
    
    # 查找并移除旧的"第三步"章节，以便重新生成
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_line = -1
        for i, line in enumerate(lines):
            if line.strip() == "## 第三步：各区域平面拟合与地形参数计算":
                start_line = i
                break
        
        if start_line != -1:
            lines = lines[:start_line]
            with open(report_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("已移除旧的第三步报告内容，准备重新生成。")
    except FileNotFoundError:
        pass # 如果文件不存在，则不执行任何操作

    new_section = """
## 第三步：各区域平面拟合、参数计算与优度检验

对第二步划分出的每个矩形区域，我们提取其内部的原始数据点，并拟合一个最佳近似平面 `z = β₀ + β₁x + β₂y`。根据拟合结果，我们计算出每个区域的平均坡度和坡向，并通过 **R²分数** 和 **均方根误差 (RMSE)** 来评估拟合的准确性。

- **R² 分数**: 范围0-1，越接近1，表示平面模型对地形的解释能力越强。
- **RMSE (m)**: 模型的预测误差，单位为米，值越小表示拟合偏差越小。

### 计算结果摘要

| 区域编号 | 数据点数 | β₀ | β₁ | β₂ | 坡度(α)/° | 坡向(φ)/° | R² 分数 | RMSE (m) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for index, row in results_df.iterrows():
        new_section += (
            f"| {int(row['区域编号'])} "
            f"| {int(row['数据点数量'])} "
            f"| {row['beta_0 (截距)']:.2f} "
            f"| {row['beta_1 (x系数)']:.2f} "
            f"| {row['beta_2 (y系数)']:.2f} "
            f"| {row['坡度 (度)']:.2f} "
            f"| {row['坡向 (度)']:.2f} "
            f"| {row['R2_Score']:.3f} "
            f"| {row['RMSE_m']:.3f} |\n"
        )
        
    try:
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write(new_section)
        print("报告更新成功。")
    except FileNotFoundError:
        print(f"错误: 报告文件 '{report_file}' 未找到。")

if __name__ == '__main__':
    results_dataframe = fit_plane_to_regions()
    
    if not results_dataframe.empty:
        append_to_markdown_report(results_dataframe) 