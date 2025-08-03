"""
乙醇偶合制备C4烯烃 - 催化剂组合及温度影响分析
问题3：探讨不同催化剂组合及温度对乙醇转化率以及C4烯烃选择性大小的影响

模块化分析包括：
1. 单因素影响分析
   1.1 可视化初步分析
   1.2 定量计算指标分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

def set_chinese_font():
    """
    设置中文字体，以便在图表中正确显示中文。
    会尝试多种常见的中文字体。
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
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("未找到指定中文字体，使用默认字体。")

class CatalystDataProcessor:
    """数据处理类"""
    
    def __init__(self, data_path, catalyst_info_path):
        """
        初始化数据处理器
        
        Args:
            data_path: 附件1数据路径
            catalyst_info_path: 每组指标数据路径
        """
        self.data_path = data_path
        self.catalyst_info_path = catalyst_info_path
        self.raw_data = None
        self.catalyst_info = None
        self.processed_data = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        self.catalyst_info = pd.read_csv(self.catalyst_info_path, encoding='utf-8')
        print(f"附件1数据形状: {self.raw_data.shape}")
        print(f"催化剂信息数据形状: {self.catalyst_info.shape}")
        
    def process_data(self):
        """处理数据：合并信息、剔除异常值、提取特征"""
        print("正在处理数据...")
        
        # 剔除A11异常样本
        self.raw_data = self.raw_data[self.raw_data['催化剂组合编号'] != 'A11'].copy()
        catalyst_info_clean = self.catalyst_info[self.catalyst_info['催化剂组合编号'] != 'A11'].copy()
        
        # 合并数据
        self.processed_data = pd.merge(self.raw_data, catalyst_info_clean, on='催化剂组合编号', how='left')
        
        # 提取Co负载量数值
        self.processed_data['Co负载量_数值'] = self.processed_data['Co负载量'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # 提取乙醇浓度数值
        self.processed_data['乙醇浓度_数值'] = self.processed_data['乙醇浓度'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # 计算装料质量比（Co/SiO2与HAP的比例）
        self.processed_data['装料质量比'] = self._calculate_loading_ratio()
        
        # 温度分组（每25摄氏度一组）
        self.processed_data['温度分组'] = ((self.processed_data['温度'] - 250) // 25 + 1) * 25 + 225
        
        print("数据处理完成")
        print(f"处理后数据形状: {self.processed_data.shape}")
        
    def _calculate_loading_ratio(self):
        """计算装料质量比"""
        ratios = []
        for _, row in self.processed_data.iterrows():
            ratio_str = row['Co/SiO2与HAP装料比']
            if pd.isna(ratio_str) or ratio_str == '50mg:无':
                ratios.append(np.nan)
            else:
                # 解析比例，如 "200mg:200mg" -> 1.0
                try:
                    parts = ratio_str.split(':')
                    if len(parts) == 2:
                        val1 = float(parts[0].replace('mg', ''))
                        val2 = float(parts[1].replace('mg', ''))
                        ratios.append(val1 / val2 if val2 != 0 else np.nan)
                    else:
                        ratios.append(np.nan)
                except:
                    ratios.append(np.nan)
        return ratios

class SingleFactorAnalyzer:
    """单因素影响分析类"""
    
    def __init__(self, data):
        """
        初始化分析器
        
        Args:
            data: 处理后的数据DataFrame
        """
        self.data = data
        self.target_vars = ['乙醇转化率(%)', 'C4烯烃选择性(%)']
        self.factor_vars = {
            '温度': '温度',
            'Co负载量': 'Co负载量_数值',
            '装料质量比': '装料质量比',
            '乙醇浓度': '乙醇浓度_数值'
        }
        
    def visualize_factors(self):
        """可视化初步分析"""
        print("开始可视化分析...")
        
        # 设置图形布局
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        
        for i, (factor_name, factor_col) in enumerate(self.factor_vars.items()):
            for j, target in enumerate(self.target_vars):
                ax = axes[i, j]
                
                if factor_name == '温度':
                    # 温度按25摄氏度分组绘制箱线图
                    sns.boxplot(data=self.data, x='温度分组', y=target, ax=ax)
                    ax.set_title(f'{target} vs {factor_name}（按25°C分组）')
                    ax.tick_params(axis='x', rotation=45)
                    
                elif factor_name == 'Co负载量':
                    # Co负载量分组绘制箱线图
                    sns.boxplot(data=self.data, x='Co负载量', y=target, ax=ax)
                    ax.set_title(f'{target} vs {factor_name}')
                    ax.tick_params(axis='x', rotation=45)
                    
                else:
                    # 装料质量比和乙醇浓度绘制散点图
                    clean_data = self.data.dropna(subset=[factor_col, target])
                    if len(clean_data) > 0:
                        sns.scatterplot(data=clean_data, x=factor_col, y=target, ax=ax, alpha=0.7)
                        # 添加趋势线
                        try:
                            z = np.polyfit(clean_data[factor_col], clean_data[target], 1)
                            p = np.poly1d(z)
                            ax.plot(clean_data[factor_col], p(clean_data[factor_col]), "r--", alpha=0.8)
                        except:
                            pass
                    ax.set_title(f'{target} vs {factor_name}')
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('single_factor_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def calculate_quantitative_indicators(self):
        """定量计算指标分析"""
        print("开始定量指标计算...")
        
        results = []
        
        for factor_name, factor_col in self.factor_vars.items():
            for target in self.target_vars:
                # 获取干净的数据
                clean_data = self.data.dropna(subset=[factor_col, target])
                
                if len(clean_data) < 3:
                    print(f"警告: {factor_name} vs {target} 的有效数据点不足")
                    continue
                
                x = clean_data[factor_col].values
                y = clean_data[target].values
                
                # 计算相关系数和p值
                corr_coef, p_value = stats.pearsonr(x, y)
                
                # 计算关系强度d（Cohen's d或效应量）
                # 这里使用相关系数的绝对值作为关系强度的度量
                effect_size = abs(corr_coef)
                
                # 计算非线性显著性（二次回归）
                try:
                    # 线性回归
                    linear_model = LinearRegression()
                    linear_model.fit(x.reshape(-1, 1), y)
                    linear_r2 = linear_model.score(x.reshape(-1, 1), y)
                    
                    # 二次回归
                    poly_features = PolynomialFeatures(degree=2, include_bias=False)
                    x_poly = poly_features.fit_transform(x.reshape(-1, 1))
                    quad_model = LinearRegression()
                    quad_model.fit(x_poly, y)
                    quad_r2 = quad_model.score(x_poly, y)
                    
                    # F检验计算非线性显著性
                    n = len(x)
                    f_stat = ((quad_r2 - linear_r2) / 1) / ((1 - quad_r2) / (n - 3))
                    p_quad = 1 - stats.f.cdf(f_stat, 1, n - 3) if f_stat > 0 else 1.0
                    
                except Exception as e:
                    print(f"二次回归计算失败 ({factor_name} vs {target}): {e}")
                    p_quad = np.nan
                
                results.append({
                    '影响因素': factor_name,
                    '目标变量': target,
                    '样本数量': len(clean_data),
                    '相关系数(ρ)': corr_coef,
                    'p值': p_value,
                    '关系强度(d)': effect_size,
                    '非线性显著性(p_quad)': p_quad,
                    '显著性评价': self._evaluate_significance(p_value),
                    '关系强度评价': self._evaluate_effect_size(effect_size),
                    '非线性评价': self._evaluate_nonlinearity(p_quad)
                })
        
        # 转换为DataFrame并保存
        results_df = pd.DataFrame(results)
        results_df = results_df.round(4)
        
        print("\n=== 定量指标分析结果 ===")
        print(results_df.to_string(index=False))
        
        # 保存结果
        results_df.to_csv('quantitative_analysis_results.csv', index=False, encoding='utf-8')
        
        return results_df
    
    def _evaluate_significance(self, p_value):
        """评价显著性水平"""
        if pd.isna(p_value):
            return "无法计算"
        elif p_value < 0.001:
            return "极显著***"
        elif p_value < 0.01:
            return "非常显著**"
        elif p_value < 0.05:
            return "显著*"
        else:
            return "不显著"
    
    def _evaluate_effect_size(self, effect_size):
        """评价关系强度"""
        if pd.isna(effect_size):
            return "无法计算"
        elif effect_size >= 0.8:
            return "强关系"
        elif effect_size >= 0.5:
            return "中等关系"
        elif effect_size >= 0.3:
            return "弱关系"
        else:
            return "很弱关系"
    
    def _evaluate_nonlinearity(self, p_quad):
        """评价非线性关系"""
        if pd.isna(p_quad):
            return "无法计算"
        elif p_quad < 0.05:
            return "存在显著非线性"
        else:
            return "线性关系充分"

class ResultSummarizer:
    """结果汇总类"""
    
    def __init__(self, quantitative_results):
        """
        初始化结果汇总器
        
        Args:
            quantitative_results: 定量分析结果DataFrame
        """
        self.results = quantitative_results
        
    def generate_summary_report(self):
        """生成汇总报告"""
        print("\n" + "="*80)
        print("催化剂组合及温度影响分析汇总报告")
        print("="*80)
        
        # 1. 乙醇转化率影响因素排序
        ethanol_results = self.results[self.results['目标变量'] == '乙醇转化率(%)'].copy()
        ethanol_results = ethanol_results.sort_values('关系强度(d)', ascending=False)
        
        print("\n1. 乙醇转化率影响因素强度排序:")
        print("-" * 50)
        for idx, row in ethanol_results.iterrows():
            print(f"   {row['影响因素']:10s}: {row['关系强度(d)']:.4f} ({row['关系强度评价']}) - {row['显著性评价']}")
        
        # 2. C4烯烃选择性影响因素排序
        c4_results = self.results[self.results['目标变量'] == 'C4烯烃选择性(%)'].copy()
        c4_results = c4_results.sort_values('关系强度(d)', ascending=False)
        
        print("\n2. C4烯烃选择性影响因素强度排序:")
        print("-" * 50)
        for idx, row in c4_results.iterrows():
            print(f"   {row['影响因素']:10s}: {row['关系强度(d)']:.4f} ({row['关系强度评价']}) - {row['显著性评价']}")
        
        # 3. 非线性关系分析
        nonlinear_cases = self.results[self.results['非线性评价'] == '存在显著非线性']
        if len(nonlinear_cases) > 0:
            print("\n3. 存在显著非线性关系的情况:")
            print("-" * 50)
            for idx, row in nonlinear_cases.iterrows():
                print(f"   {row['影响因素']} -> {row['目标变量']}: p_quad = {row['非线性显著性(p_quad)']:.4f}")
        else:
            print("\n3. 非线性关系分析:")
            print("-" * 50)
            print("   未发现显著的非线性关系，线性模型基本充分")
        
        # 4. 主要结论
        print("\n4. 主要结论:")
        print("-" * 50)
        
        # 找出最强的影响因素
        strongest_ethanol = ethanol_results.iloc[0]
        strongest_c4 = c4_results.iloc[0]
        
        print(f"   a) 对乙醇转化率影响最强的因素是 {strongest_ethanol['影响因素']} (r={strongest_ethanol['相关系数(ρ)']:.4f})")
        print(f"   b) 对C4烯烃选择性影响最强的因素是 {strongest_c4['影响因素']} (r={strongest_c4['相关系数(ρ)']:.4f})")
        
        # 温度的影响
        temp_ethanol = ethanol_results[ethanol_results['影响因素'] == '温度']
        temp_c4 = c4_results[c4_results['影响因素'] == '温度']
        
        if len(temp_ethanol) > 0 and len(temp_c4) > 0:
            print(f"   c) 温度对乙醇转化率的影响: {temp_ethanol.iloc[0]['显著性评价']} (r={temp_ethanol.iloc[0]['相关系数(ρ)']:.4f})")
            print(f"   d) 温度对C4烯烃选择性的影响: {temp_c4.iloc[0]['显著性评价']} (r={temp_c4.iloc[0]['相关系数(ρ)']:.4f})")
        
        print("\n" + "="*80)
        
        # 保存汇总报告
        summary_content = self._generate_markdown_report()
        with open('influence_analysis_summary_report.md', 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print("汇总报告已保存到: influence_analysis_summary_report.md")
    
    def _generate_markdown_report(self):
        """生成Markdown格式的报告"""
        report = """# 催化剂组合及温度影响分析报告

## 1. 分析目标
探讨不同催化剂组合及温度对乙醇转化率以及C4烯烃选择性大小的影响

## 2. 分析方法
- **可视化分析**: 温度箱线图（25°C分组）、Co负载量箱线图、装料比/乙醇浓度散点图
- **定量分析**: 相关系数(ρ)、显著性检验(p值)、关系强度(d)、非线性检验(p_quad)

## 3. 主要发现

### 3.1 乙醇转化率影响因素排序
"""
        
        ethanol_results = self.results[self.results['目标变量'] == '乙醇转化率(%)'].copy()
        ethanol_results = ethanol_results.sort_values('关系强度(d)', ascending=False)
        
        for idx, row in ethanol_results.iterrows():
            report += f"- **{row['影响因素']}**: r={row['相关系数(ρ)']:.4f}, {row['显著性评价']}, {row['关系强度评价']}\n"
        
        report += "\n### 3.2 C4烯烃选择性影响因素排序\n"
        
        c4_results = self.results[self.results['目标变量'] == 'C4烯烃选择性(%)'].copy()
        c4_results = c4_results.sort_values('关系强度(d)', ascending=False)
        
        for idx, row in c4_results.iterrows():
            report += f"- **{row['影响因素']}**: r={row['相关系数(ρ)']:.4f}, {row['显著性评价']}, {row['关系强度评价']}\n"
        
        report += "\n### 3.3 非线性关系分析\n"
        
        nonlinear_cases = self.results[self.results['非线性评价'] == '存在显著非线性']
        if len(nonlinear_cases) > 0:
            for idx, row in nonlinear_cases.iterrows():
                report += f"- **{row['影响因素']} -> {row['目标变量']}**: 存在显著非线性关系 (p_quad={row['非线性显著性(p_quad)']:.4f})\n"
        else:
            report += "- 未发现显著的非线性关系，线性模型基本充分\n"
        
        report += """
## 4. 结论与建议

基于单因素影响分析结果，为后续多因素建模和工艺优化提供了重要依据。温度作为关键工艺参数，对两个目标变量都有重要影响。催化剂组合参数的影响程度各有不同，需要在多因素联合分析中进一步探讨其协同效应。

## 5. 数据质量说明
- 已剔除A11异常样本
- 分析中处理了缺失值
- 所有统计检验均基于有效数据点
"""
        
        return report

def main():
    """主函数"""
    print("开始催化剂组合及温度影响分析...")
    
    # 设置中文字体
    set_chinese_font()
    
    # 1. 数据处理
    processor = CatalystDataProcessor(
        data_path='../附件1.csv',
        catalyst_info_path='每组指标.csv'
    )
    processor.load_data()
    processor.process_data()
    
    # 2. 单因素影响分析
    analyzer = SingleFactorAnalyzer(processor.processed_data)
    
    # 2.1 可视化分析
    analyzer.visualize_factors()
    
    # 2.2 定量指标计算
    quantitative_results = analyzer.calculate_quantitative_indicators()
    
    # 3. 结果汇总
    summarizer = ResultSummarizer(quantitative_results)
    summarizer.generate_summary_report()
    
    print("\n分析完成！")
    print("输出文件:")
    print("- single_factor_visualization.png")
    print("- quantitative_analysis_results.csv")
    print("- influence_analysis_summary_report.md")

if __name__ == "__main__":
    main()