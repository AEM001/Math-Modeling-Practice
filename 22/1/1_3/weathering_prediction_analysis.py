import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class WeatheringPredictionAnalyzer:
    def __init__(self):
        self.significant_changes = {}
        self.predicted_data = None
        self.clr_data = None
        self.gaojia_results = None
        self.qianbai_results = None
        
    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据文件...")
        
        # 加载分析结果
        self.gaojia_results = pd.read_csv('1/1_2/高钾分析结果.csv')
        self.qianbai_results = pd.read_csv('1/1_2/铅钡分析结果.csv')
        
        # 加载CLR处理后的数据
        self.clr_data = pd.read_csv('1/1_3/附件2_处理后_CLR.csv')
        
        print(f"高钾分析结果: {len(self.gaojia_results)} 个成分")
        print(f"铅钡分析结果: {len(self.qianbai_results)} 个成分")
        print(f"CLR数据: {len(self.clr_data)} 个样本")
        
    def extract_significant_changes(self):
        """提取显著变化的指标和变化规律"""
        print("\n正在提取显著变化的指标...")
        
        if self.gaojia_results is not None:
            # 高钾类型的显著变化
            gaojia_significant = self.gaojia_results[self.gaojia_results['是否显著'] == '是'].copy()
            self.significant_changes['高钾'] = {}
            
            for _, row in gaojia_significant.iterrows():
                component = row['component']
                change = row['mean_weathered'] - row['mean_unweathered']
                relative_change = change / abs(row['mean_unweathered']) if row['mean_unweathered'] != 0 else 0
                
                self.significant_changes['高钾'][component] = {
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'weathered_mean': row['mean_weathered'],
                    'unweathered_mean': row['mean_unweathered']
                }
        
        if self.qianbai_results is not None:
            # 铅钡类型的显著变化
            qianbai_significant = self.qianbai_results[self.qianbai_results['是否显著'] == '是'].copy()
            self.significant_changes['铅钡'] = {}
            
            for _, row in qianbai_significant.iterrows():
                component = row['component']
                change = row['mean_weathered'] - row['mean_unweathered']
                relative_change = change / abs(row['mean_unweathered']) if row['mean_unweathered'] != 0 else 0
                
                self.significant_changes['铅钡'][component] = {
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'weathered_mean': row['mean_weathered'],
                    'unweathered_mean': row['mean_unweathered']
                }
        
        print(f"高钾类型显著变化成分: {len(self.significant_changes.get('高钾', {}))} 个")
        print(f"铅钡类型显著变化成分: {len(self.significant_changes.get('铅钡', {}))} 个")
        
    def predict_unweathered_composition(self):
        """对风化样本进行未风化状态预测"""
        print("\n正在对风化样本进行未风化状态预测...")
        
        if self.clr_data is None:
            print("CLR数据未加载")
            return
            
        # 筛选风化样本
        weathered_samples = self.clr_data[self.clr_data['表面风化'] == '风化'].copy()
        print(f"找到 {len(weathered_samples)} 个风化样本")
        
        # 为每个风化样本预测其未风化状态
        predicted_samples = []
        
        for idx, sample in weathered_samples.iterrows():
            sample_type = sample['类型']
            if sample_type not in self.significant_changes:
                continue
                
            predicted_sample = sample.copy()
            predicted_sample['原始状态'] = '风化'
            predicted_sample['预测状态'] = '未风化(预测)'
            
            # 对每个显著变化的成分进行预测
            for component, change_info in self.significant_changes[sample_type].items():
                if component in sample.index:
                    current_value = sample[component]
                    predicted_value = current_value - change_info['absolute_change']
                    predicted_sample[component] = predicted_value
            
            predicted_samples.append(predicted_sample)
        
        if predicted_samples:
            self.predicted_data = pd.DataFrame(predicted_samples)
            print(f"成功预测 {len(self.predicted_data)} 个样本的未风化状态")
        else:
            self.predicted_data = pd.DataFrame()
            print("没有找到可预测的样本")
        
    def clr_inverse_transform(self, clr_data):
        """CLR逆变换"""
        component_cols = [col for col in clr_data.columns 
                        if '(' in col and ')' in col and col not in ['文物采样点', '类型', '表面风化', '原始状态', '预测状态']]
        
        # 提取成分数据
        clr_values = clr_data[component_cols].values
        
        # CLR逆变换: exp(y_i) / sum(exp(y_j))
        exp_values = np.exp(clr_values)
        row_sums = exp_values.sum(axis=1, keepdims=True)
        original_values = exp_values / row_sums * 100
        
        # 创建结果DataFrame
        result_df = clr_data[['文物采样点', '类型', '表面风化']].copy()
        if '原始状态' in clr_data.columns:
            result_df['原始状态'] = clr_data['原始状态']
        if '预测状态' in clr_data.columns:
            result_df['预测状态'] = clr_data['预测状态']
        
        for i, col in enumerate(component_cols):
            result_df[col] = original_values[:, i]
        
        return result_df
    
    def save_prediction_results(self):
        """保存预测结果"""
        if self.predicted_data is None or self.predicted_data.empty:
            print("没有预测数据可保存")
            return
        
        # 保存CLR格式的预测结果
        clr_output_file = '1/1_3/风化样本未风化状态预测_CLR.csv'
        self.predicted_data.to_csv(clr_output_file, index=False, encoding='utf-8-sig')
        print(f"CLR格式预测结果已保存: {clr_output_file}")
        
        # 进行CLR逆变换
        original_format_data = self.clr_inverse_transform(self.predicted_data)
        
        # 保存原始格式的预测结果
        original_output_file = '1/1_3/风化样本未风化状态预测_原始格式.csv'
        original_format_data.to_csv(original_output_file, index=False, encoding='utf-8-sig')
        print(f"原始格式预测结果已保存: {original_output_file}")
        
    def print_analysis_summary(self):
        """打印分析摘要"""
        print("\n=== 显著变化成分分析摘要 ===")
        
        for glass_type, changes in self.significant_changes.items():
            print(f"\n{glass_type}类型玻璃:")
            print("-" * 60)
            print(f"{'成分':<15} {'变化量':<10} {'变化趋势'}")
            print("-" * 60)
            
            for component, change_info in changes.items():
                abs_change = change_info['absolute_change']
                trend = "增加" if abs_change > 0 else "减少"
                print(f"{component:<15} {abs_change:>8.3f} {trend}")
    
    def generate_report(self):
        """生成分析报告"""
        report_content = f"""# 玻璃文物风化样本未风化状态预测分析报告

## 分析概述
本报告基于已有的风化与未风化样本统计分析结果，对风化样本进行了未风化状态的预测。

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据来源
- 高钾玻璃分析结果: {len(self.gaojia_results) if self.gaojia_results is not None else 0} 个化学成分
- 铅钡玻璃分析结果: {len(self.qianbai_results) if self.qianbai_results is not None else 0} 个化学成分  
- CLR转换后样本数据: {len(self.clr_data) if self.clr_data is not None else 0} 个样本

## 2. 显著变化成分识别

"""
        
        for glass_type, changes in self.significant_changes.items():
            report_content += f"### {glass_type}类型玻璃\n"
            report_content += f"显著变化成分数量: {len(changes)}\n\n"
            
            for component, change_info in changes.items():
                trend = "增加" if change_info['absolute_change'] > 0 else "减少"
                rel_change = change_info['relative_change'] * 100
                report_content += f"- **{component}**: {trend} {abs(change_info['absolute_change']):.3f} (相对变化: {rel_change:.1f}%)\n"
            report_content += "\n"
        
        predicted_count = 0 if self.predicted_data is None or self.predicted_data.empty else len(self.predicted_data)
        report_content += f"""## 3. 预测结果
- 总预测样本数: {predicted_count}

## 4. 预测方法
对于每个显著变化的化学成分，使用以下公式进行预测：
```
预测未风化值 = 当前风化值 - 平均风化变化量
```

## 5. 输出文件
- CLR格式预测结果: 风化样本未风化状态预测_CLR.csv
- 原始格式预测结果: 风化样本未风化状态预测_原始格式.csv

---
*本报告由自动化分析程序生成*
"""
        
        report_file = '1/1_3/风化样本预测分析报告.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n分析报告已保存: {report_file}")

def main():
    """主函数"""
    print("=== 玻璃文物风化样本未风化状态预测分析 ===\n")
    
    analyzer = WeatheringPredictionAnalyzer()
    
    try:
        # 1. 加载数据
        analyzer.load_data()
        
        # 2. 提取显著变化
        analyzer.extract_significant_changes()
        analyzer.print_analysis_summary()
        
        # 3. 进行预测
        analyzer.predict_unweathered_composition()
        
        # 4. 保存结果
        analyzer.save_prediction_results()
        
        # 5. 生成报告
        analyzer.generate_report()
        
        print("\n=== 分析完成 ===")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 