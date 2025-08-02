import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings


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

def analyze_and_visualize(df, independent_var, dependent_var='表面风化', report_file=None):
    """分析变量关系并生成markdown报告"""
    
    # 创建列联表
    contingency_table = pd.crosstab(df[dependent_var], df[independent_var], margins=True, margins_name="总计")
    
    # 卡方检验
    test_data = pd.crosstab(df[dependent_var], df[independent_var])
    result = chi2_contingency(test_data)
    chi2: float = result[0]  # type: ignore
    p_value: float = result[1]  # type: ignore
    dof: int = result[2]  # type: ignore
    
    # 生成markdown报告
    markdown_content = f"""
## {independent_var}与{dependent_var}的关系分析

{create_markdown_table(contingency_table, f"{independent_var}与{dependent_var}列联表")}

### 卡方检验结果

| 统计量 | 数值 |
|--------|------|
| 卡方统计量 | {chi2:.4f} |
| P值 | {p_value:.4f} |
| 自由度 | {dof} |

### 结论

"""
    
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"P值 ({p_value:.4f}) < {alpha}，拒绝原假设。**{dependent_var}与{independent_var}存在显著关联**。"
    else:
        conclusion = f"P值 ({p_value:.4f}) ≥ {alpha}，不能拒绝原假设。{dependent_var}与{independent_var}无显著关联。"
    
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
    file_path = '附件.csv'
    
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

本报告通过卡方检验分析了玻璃文物的表面风化与其类型、纹饰、颜色之间的关联性。

- **显著性水平：** α = 0.05
- **检验方法：** 皮尔逊卡方检验
- **可视化：** 分组条形图

*报告完成*
""")
    
    print(f"\n分析完成！markdown报告已保存为：{report_file}")

if __name__ == '__main__':
    main()

