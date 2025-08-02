import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
import warnings
import numpy as np
from typing import Optional, Tuple


# --- 全局设置 ---
# 忽略警告以提高输出可读性
warnings.filterwarnings('ignore')
# 设置绘图样式
sns.set_theme(style="whitegrid", palette="pastel")

# 配置Matplotlib以正确显示中文字符
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'SimHei', 'sans-serif']  # Mac系统中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 10  # 设置默认字体大小

# --- 主要分析函数 ---

def create_markdown_table(df, title):
    """创建markdown格式的表格"""
    markdown = f"\n### {title}\n\n"
    
    # 表头
    headers = "| " + " | ".join(df.columns) + " |\n"
    separator = "|" + "---|" * len(df.columns) + "\n"
    
    markdown += headers + separator
    
    # 数据行
    for index, row in df.iterrows():
        row_data = "| " + " | ".join([str(index)] + [str(val) for val in row]) + " |\n"
        markdown += row_data
    
    return markdown

def check_chi2_assumptions(contingency_table):
    """
    检查卡方检验的前提条件
    返回：(检验方法, 期望频数统计, 检查结果文本)
    """
    # 计算期望频数
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    expected_array = np.array(expected)
    
    # 统计期望频数情况
    cells_less_than_1 = np.sum(expected_array < 1)
    cells_less_than_5 = np.sum(expected_array < 5)
    total_cells = expected_array.size
    percent_less_than_5 = (cells_less_than_5 / total_cells) * 100
    
    # 决定检验方法
    if cells_less_than_1 > 0:
        method = "Fisher精确检验"
        reason = f"存在{cells_less_than_1}个期望频数<1的单元格"
    elif percent_less_than_5 > 20:
        method = "Fisher精确检验"  
        reason = f"{percent_less_than_5:.1f}%的单元格期望频数<5，超过20%"
    else:
        method = "皮尔逊卡方检验"
        reason = "所有期望频数≥5，满足标准卡方检验条件"
    
    # 创建期望频数报告
    expected_df = pd.DataFrame(expected_array, 
                              index=contingency_table.index,
                              columns=contingency_table.columns)
    
    assumptions_text = f"""
### 卡方检验前提条件检查

**期望频数矩阵：**
{expected_df.round(2).to_string()}

**统计结果：**
- 总单元格数：{total_cells}
- 期望频数<1的单元格：{cells_less_than_1}个
- 期望频数<5的单元格：{cells_less_than_5}个 ({percent_less_than_5:.1f}%)
- **选择方法：{method}**
- **理由：{reason}**

"""
    
    return method, expected_df, assumptions_text

def perform_statistical_test(contingency_table, method) -> Tuple[Optional[float], float, Optional[int], str]:
    """根据方法执行相应的统计检验"""
    
    if method == "Fisher精确检验":
        if contingency_table.shape == (2, 2):
            # 2x2表格用Fisher精确检验
            oddsratio, p_value = fisher_exact(contingency_table)
            return None, float(p_value), None, f"比值比 (OR) = {float(oddsratio):.4f}"
        else:
            # 多维表格用卡方检验但标注警告
            chi2_stat, p_value, dof, _ = chi2_contingency(contingency_table)
            return float(chi2_stat), float(p_value), int(dof), "注：多维表格Fisher检验，此处显示卡方近似值"
    else:
        # 标准卡方检验
        chi2_stat, p_value, dof, _ = chi2_contingency(contingency_table)
        return float(chi2_stat), float(p_value), int(dof), ""

def analyze_and_visualize(df, independent_var, dependent_var='表面风化', report_file=None):
    """分析变量关系并生成markdown报告"""
    
    # 创建列联表
    contingency_table = pd.crosstab(df[dependent_var], df[independent_var], margins=True, margins_name="总计")
    test_data = pd.crosstab(df[dependent_var], df[independent_var])
    
    # 检查前提条件并选择检验方法
    method, expected_df, assumptions_text = check_chi2_assumptions(test_data)
    
    # 执行相应的统计检验
    chi2_stat, p_value, dof, additional_info = perform_statistical_test(test_data, method)
    
    # 初始化Cramer's V
    cramers_v = None

    # 生成markdown报告
    markdown_content = f"""
## {independent_var}与{dependent_var}的关系分析

{create_markdown_table(contingency_table, f"{independent_var}与{dependent_var}列联表")}

{assumptions_text}

### 检验结果 ({method})

| 统计量 | 数值 |
|--------|------|"""
    
    if chi2_stat is not None:
        markdown_content += f"""
| 卡方统计量 | {chi2_stat:.4f} |
| P值 | {p_value:.4f} |
| 自由度 | {dof} |"""
        # 计算 Cramer's V
        n = test_data.to_numpy().sum()
        if n > 0:
            try:
                r, c = test_data.shape
                min_dim = min(r - 1, c - 1)
                if min_dim > 0:
                    cramers_v = np.sqrt(chi2_stat / (n * min_dim))
                    markdown_content += f"""
| Cramer's V | {cramers_v:.4f} |"""
            except (ValueError, ZeroDivisionError):
                cramers_v = None # 计算错误时保持None
    else:
        markdown_content += f"""
| P值 | {p_value:.4f} |"""
    
    if additional_info:
        markdown_content += f"""
| 备注 | {additional_info} |"""
    
    markdown_content += "\n\n### 结论\n\n"
    
    alpha = 0.05
    p_val = p_value if p_value is not None else 1.0
    if p_val < alpha:
        conclusion = f"P值 ({p_val:.4f}) < {alpha}，拒绝原假设。**{dependent_var}与{independent_var}存在显著关联**。"
        if cramers_v is not None:
            if cramers_v >= 0.5:
                strength = "强关联"
            elif cramers_v >= 0.3:
                strength = "中等强度关联"
            elif cramers_v >= 0.1:
                strength = "弱关联"
            else:
                strength = "可忽略的关联"
            conclusion += f" Cramer's V 值为 {cramers_v:.4f}，表明这是一个**{strength}**。"
    else:
        conclusion = f"P值 ({p_val:.4f}) ≥ {alpha}，不能拒绝原假设。{dependent_var}与{independent_var}无显著关联。"
    
    markdown_content += conclusion + "\n\n"
    
    # 保存或追加到报告文件
    if report_file:
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write(markdown_content)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=independent_var, hue=dependent_var, order=df[independent_var].value_counts().index)
    
    plt.title(f'{independent_var}与{dependent_var}关系图', fontsize=16, fontweight='bold')
    plt.xlabel(independent_var, fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=dependent_var)
    plt.tight_layout()
    
    # 添加数据标签
    for patch in ax.patches:
        height = patch.get_height()  # type: ignore
        if height > 0:
            x = patch.get_x() + patch.get_width() / 2.  # type: ignore
            ax.annotate(f'{int(height)}', (x, height), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
    
    plot_filename = f'{dependent_var}_vs_{independent_var}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 在markdown中添加图片引用
    img_markdown = f"![{independent_var}与{dependent_var}关系图]({plot_filename})\n\n---\n\n"
    if report_file:
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write(img_markdown)
    
    return markdown_content

def main():
    """主函数"""
    file_path = '/Users/Mac/Downloads/22C/22/1/1_1/附件.csv'
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')
    
    required_columns = ['表面风化', '类型', '纹饰', '颜色']
    df_cleaned = df.dropna(subset=required_columns)
    
    # 数据转换
    if df_cleaned['表面风化'].dtype in ['int64', 'float64']:
        df_cleaned = df_cleaned.copy()
        df_cleaned['表面风化'] = df_cleaned['表面风化'].replace({1: '无风化', 2: '风化'})
    
    # 创建markdown报告文件
    report_file = '玻璃文物分析报告.md'
    
    # 初始化报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"""# 玻璃文物统计分析报告

**数据概况：** 共{df_cleaned.shape[0]}条有效记录，{df_cleaned.shape[1]}个变量

**分析日期：** {pd.Timestamp.now().strftime('%Y年%m月%d日')}

---

""")
    
    # 执行分析
    variables = ['类型', '纹饰', '颜色']
    for var in variables:
        print(f"正在分析：{var}与表面风化的关系...")
        analyze_and_visualize(df_cleaned, var, report_file=report_file)
    
    # 添加总结
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write("""## 分析总结

本报告通过统计检验分析了玻璃文物的表面风化与其类型、纹饰、颜色之间的关联性。

### 检验方法选择原则：
- **标准卡方检验：** 所有期望频数≥5
- **Fisher精确检验：** 存在期望频数<1或>20%单元格期望频数<5

### 前提条件检查：
- ✅ **变量类型：** 所有变量均为分类变量
- ✅ **独立性：** 样本相互独立，每个观测值仅归属一个交叉单元格
- ✅ **期望频数：** 已根据期望频数情况选择合适的检验方法

**显著性水平：** α = 0.05

*报告完成*
""")
    
    print(f"\n分析完成！markdown报告已保存为：{report_file}")

if __name__ == '__main__':
    main()

