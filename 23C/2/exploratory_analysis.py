# -*- coding: utf-8 -*-
"""
探索性分析模块
检验周几效应、价格-销量关系的稳定性与非线性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ExploratoryAnalyzer:
    """探索性分析器"""
    
    def __init__(self, data_path='clean_data_full.csv'):
        """初始化分析器"""
        self.data_path = data_path
        self.data = None
        self.weekday_results = {}
        self.price_quantity_results = {}
        
    def load_data(self):
        """加载清洗后的数据"""
        print("正在加载清洗后的数据...")
        self.data = pd.read_csv(self.data_path)
        self.data['销售日期'] = pd.to_datetime(self.data['销售日期'])
        self.data['weekday'] = self.data['销售日期'].dt.dayofweek
        self.data['weekday_name'] = self.data['销售日期'].dt.day_name()
        
        # 添加对数变量
        self.data['ln_price'] = np.log(self.data['正常销售单价(元/千克)'])
        self.data['ln_quantity'] = np.log(self.data['正常销量(千克)'])
        
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        return self
        
    def analyze_weekday_effects(self):
        """分析周几效应"""
        print("\n=== 周几效应分析 ===")
        
        # 中文星期映射
        weekday_map = {
            0: '周一', 1: '周二', 2: '周三', 3: '周四', 
            4: '周五', 5: '周六', 6: '周日'
        }
        self.data['weekday_cn'] = self.data['weekday'].map(weekday_map)
        
        results = {}
        
        # 按品类分析周几效应
        for category in self.data['分类名称'].unique():
            cat_data = self.data[self.data['分类名称'] == category]
            
            if len(cat_data) < 50:  # 样本太少跳过
                continue
                
            print(f"\n--- {category} ---")
            
            # 按周几分组的销量数据
            weekday_groups = []
            weekday_labels = []
            
            for wd in range(7):
                wd_data = cat_data[cat_data['weekday'] == wd]['正常销量(千克)']
                if len(wd_data) > 5:  # 至少5个观测值
                    weekday_groups.append(wd_data.values)
                    weekday_labels.append(weekday_map[wd])
            
            if len(weekday_groups) < 3:  # 至少3个组才能比较
                continue
                
            # Kruskal-Wallis检验
            try:
                h_stat, p_value = kruskal(*weekday_groups)
                print(f"Kruskal-Wallis H统计量: {h_stat:.4f}")
                print(f"p值: {p_value:.6f}")
                
                is_significant = p_value < 0.05
                print(f"周几效应显著性: {'显著' if is_significant else '不显著'}")
                
                # 如果显著，进行简化的成对比较
                posthoc_results = None
                if is_significant and len(weekday_groups) > 2:
                    print("进行成对Mann-Whitney U检验...")
                    posthoc_results = {}
                    
                    # 成对比较（简化版）
                    for i in range(len(weekday_groups)):
                        for j in range(i+1, len(weekday_groups)):
                            label1, label2 = weekday_labels[i], weekday_labels[j]
                            group1, group2 = weekday_groups[i], weekday_groups[j]
                            
                            try:
                                u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                                # Bonferroni校正
                                n_comparisons = len(weekday_groups) * (len(weekday_groups) - 1) / 2
                                p_val_corrected = min(p_val * n_comparisons, 1.0)
                                
                                posthoc_results[f"{label1}_vs_{label2}"] = {
                                    'u_statistic': u_stat,
                                    'p_value': p_val,
                                    'p_value_corrected': p_val_corrected,
                                    'significant': p_val_corrected < 0.05
                                }
                            except:
                                continue
                
                # 描述性统计
                weekday_stats = {}
                for group, label in zip(weekday_groups, weekday_labels):
                    weekday_stats[label] = {
                        'mean': np.mean(group),
                        'median': np.median(group),
                        'std': np.std(group),
                        'count': len(group)
                    }
                
                results[category] = {
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'weekday_stats': weekday_stats,
                    'posthoc_results': posthoc_results
                }
                
            except Exception as e:
                print(f"分析失败: {e}")
                continue
        
        self.weekday_results = results
        
        # 可视化周几效应
        self._plot_weekday_effects()
        
    def _plot_weekday_effects(self):
        """绘制周几效应图"""
        n_categories = len(self.weekday_results)
        if n_categories == 0:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (category, results) in enumerate(self.weekday_results.items()):
            if i >= 6:  # 最多显示6个品类
                break
                
            ax = axes[i]
            
            # 准备数据
            weekdays = []
            means = []
            stds = []
            
            for wd_name, stats_dict in results['weekday_stats'].items():
                weekdays.append(wd_name)
                means.append(stats_dict['mean'])
                stds.append(stats_dict['std'])
            
            # 柱状图
            bars = ax.bar(weekdays, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{category}\n(p={results["p_value"]:.4f})')
            ax.set_ylabel('平均销量(千克)')
            ax.tick_params(axis='x', rotation=45)
            
            # 标记显著性
            if results['is_significant']:
                ax.set_facecolor('#f0f8ff')  # 浅蓝色背景表示显著
        
        # 隐藏多余的子图
        for i in range(n_categories, 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('weekday_effects_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("周几效应图已保存: weekday_effects_analysis.png")
        
    def analyze_price_quantity_relationship(self):
        """分析价格-销量关系的稳定性"""
        print("\n=== 价格-销量关系分析 ===")
        
        results = {}
        
        # 按品类分析
        for category in self.data['分类名称'].unique():
            cat_data = self.data[self.data['分类名称'] == category]
            
            if len(cat_data) < 30:  # 样本太少跳过
                continue
                
            print(f"\n--- {category} ---")
            
            # 基础双对数回归
            X = cat_data['ln_price'].values.reshape(-1, 1)
            y = cat_data['ln_quantity'].values
            
            # 过滤无效值
            valid_mask = np.isfinite(X.flatten()) & np.isfinite(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 10:
                continue
                
            # 线性回归
            reg = LinearRegression()
            reg.fit(X_clean, y_clean)
            
            y_pred = reg.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
            
            # 计算价格弹性（斜率）
            elasticity = reg.coef_[0]
            intercept = reg.intercept_
            
            print(f"价格弹性: {elasticity:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"样本量: {len(X_clean)}")
            
            # 分段分析：检查不同价格区间的弹性是否一致
            price_quantiles = np.quantile(cat_data['正常销售单价(元/千克)'], [0.33, 0.67])
            
            segment_results = {}
            for i, (low, high) in enumerate([(0, price_quantiles[0]), 
                                           (price_quantiles[0], price_quantiles[1]),
                                           (price_quantiles[1], np.inf)]):
                segment_name = f"价格段{i+1}"
                
                if i == 0:
                    segment_data = cat_data[cat_data['正常销售单价(元/千克)'] <= high]
                elif i == 1:
                    segment_data = cat_data[
                        (cat_data['正常销售单价(元/千克)'] > low) & 
                        (cat_data['正常销售单价(元/千克)'] <= high)
                    ]
                else:
                    segment_data = cat_data[cat_data['正常销售单价(元/千克)'] > low]
                
                if len(segment_data) < 10:
                    continue
                    
                X_seg = segment_data['ln_price'].values.reshape(-1, 1)
                y_seg = segment_data['ln_quantity'].values
                
                valid_mask_seg = np.isfinite(X_seg.flatten()) & np.isfinite(y_seg)
                X_seg_clean = X_seg[valid_mask_seg]
                y_seg_clean = y_seg[valid_mask_seg]
                
                if len(X_seg_clean) < 5:
                    continue
                
                reg_seg = LinearRegression()
                reg_seg.fit(X_seg_clean, y_seg_clean)
                
                y_pred_seg = reg_seg.predict(X_seg_clean)
                r2_seg = r2_score(y_seg_clean, y_pred_seg)
                
                segment_results[segment_name] = {
                    'elasticity': reg_seg.coef_[0],
                    'r2': r2_seg,
                    'n_samples': len(X_seg_clean),
                    'price_range': (segment_data['正常销售单价(元/千克)'].min(), 
                                  segment_data['正常销售单价(元/千克)'].max())
                }
                
                print(f"  {segment_name}: 弹性={reg_seg.coef_[0]:.4f}, R²={r2_seg:.4f}, n={len(X_seg_clean)}")
            
            # 时间稳定性分析：按季度分析弹性变化
            cat_data_time = cat_data.copy()
            cat_data_time['quarter'] = cat_data_time['销售日期'].dt.to_period('Q')
            
            quarterly_elasticity = {}
            for quarter in cat_data_time['quarter'].unique():
                q_data = cat_data_time[cat_data_time['quarter'] == quarter]
                
                if len(q_data) < 10:
                    continue
                    
                X_q = q_data['ln_price'].values.reshape(-1, 1)
                y_q = q_data['ln_quantity'].values
                
                valid_mask_q = np.isfinite(X_q.flatten()) & np.isfinite(y_q)
                X_q_clean = X_q[valid_mask_q]
                y_q_clean = y_q[valid_mask_q]
                
                if len(X_q_clean) < 5:
                    continue
                
                reg_q = LinearRegression()
                reg_q.fit(X_q_clean, y_q_clean)
                
                quarterly_elasticity[str(quarter)] = {
                    'elasticity': reg_q.coef_[0],
                    'r2': r2_score(y_q_clean, reg_q.predict(X_q_clean)),
                    'n_samples': len(X_q_clean)
                }
            
            # 弹性稳定性：计算季度弹性的标准差
            if len(quarterly_elasticity) > 1:
                elasticity_values = [v['elasticity'] for v in quarterly_elasticity.values()]
                elasticity_stability = np.std(elasticity_values)
                print(f"  弹性稳定性(季度标准差): {elasticity_stability:.4f}")
            else:
                elasticity_stability = None
            
            results[category] = {
                'overall_elasticity': elasticity,
                'overall_r2': r2,
                'n_samples': len(X_clean),
                'segment_results': segment_results,
                'quarterly_elasticity': quarterly_elasticity,
                'elasticity_stability': elasticity_stability
            }
        
        self.price_quantity_results = results
        
        # 可视化价格-销量关系
        self._plot_price_quantity_relationships()
        
    def _plot_price_quantity_relationships(self):
        """绘制价格-销量关系图"""
        n_categories = len(self.price_quantity_results)
        if n_categories == 0:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (category, results) in enumerate(self.price_quantity_results.items()):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # 获取该品类数据
            cat_data = self.data[self.data['分类名称'] == category]
            
            # 散点图
            ax.scatter(cat_data['ln_price'], cat_data['ln_quantity'], 
                      alpha=0.5, s=20, color='lightblue')
            
            # 拟合线
            X_range = np.linspace(cat_data['ln_price'].min(), 
                                cat_data['ln_price'].max(), 100)
            y_fit = results['overall_elasticity'] * X_range + \
                   (cat_data['ln_quantity'].mean() - 
                    results['overall_elasticity'] * cat_data['ln_price'].mean())
            
            ax.plot(X_range, y_fit, 'r-', linewidth=2, 
                   label=f'弹性={results["overall_elasticity"]:.3f}')
            
            ax.set_xlabel('ln(价格)')
            ax.set_ylabel('ln(销量)')
            ax.set_title(f'{category}\nR²={results["overall_r2"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_categories, 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('price_quantity_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("价格-销量关系图已保存: price_quantity_relationships.png")
        
    def detect_nonlinearity(self):
        """检测非线性关系"""
        print("\n=== 非线性检测 ===")
        
        nonlinearity_results = {}
        
        for category in self.data['分类名称'].unique():
            cat_data = self.data[self.data['分类名称'] == category]
            
            if len(cat_data) < 50:
                continue
                
            print(f"\n--- {category} ---")
            
            # 准备数据
            X = cat_data['ln_price'].values
            y = cat_data['ln_quantity'].values
            
            valid_mask = np.isfinite(X) & np.isfinite(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 20:
                continue
            
            # 线性模型
            reg_linear = LinearRegression()
            reg_linear.fit(X_clean.reshape(-1, 1), y_clean)
            y_pred_linear = reg_linear.predict(X_clean.reshape(-1, 1))
            r2_linear = r2_score(y_clean, y_pred_linear)
            
            # 二次模型
            X_poly = np.column_stack([X_clean, X_clean**2])
            reg_poly = LinearRegression()
            reg_poly.fit(X_poly, y_clean)
            y_pred_poly = reg_poly.predict(X_poly)
            r2_poly = r2_score(y_clean, y_pred_poly)
            
            # 分段线性模型（在中位数处分段）
            X_median = np.median(X_clean)
            X_seg1 = X_clean[X_clean <= X_median]
            y_seg1 = y_clean[X_clean <= X_median]
            X_seg2 = X_clean[X_clean > X_median]
            y_seg2 = y_clean[X_clean > X_median]
            
            r2_piecewise = None
            if len(X_seg1) > 5 and len(X_seg2) > 5:
                reg_seg1 = LinearRegression()
                reg_seg1.fit(X_seg1.reshape(-1, 1), y_seg1)
                reg_seg2 = LinearRegression()
                reg_seg2.fit(X_seg2.reshape(-1, 1), y_seg2)
                
                y_pred_seg1 = reg_seg1.predict(X_seg1.reshape(-1, 1))
                y_pred_seg2 = reg_seg2.predict(X_seg2.reshape(-1, 1))
                
                # 计算分段模型的整体R²
                ss_res = np.sum((y_seg1 - y_pred_seg1)**2) + np.sum((y_seg2 - y_pred_seg2)**2)
                ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
                r2_piecewise = 1 - (ss_res / ss_tot)
            
            print(f"线性模型 R²: {r2_linear:.4f}")
            print(f"二次模型 R²: {r2_poly:.4f}")
            if r2_piecewise is not None:
                print(f"分段线性 R²: {r2_piecewise:.4f}")
            
            # 判断是否存在显著非线性
            improvement_poly = r2_poly - r2_linear
            improvement_piecewise = (r2_piecewise - r2_linear) if r2_piecewise else 0
            
            has_nonlinearity = improvement_poly > 0.05 or improvement_piecewise > 0.05
            
            print(f"非线性特征: {'存在' if has_nonlinearity else '不明显'}")
            
            nonlinearity_results[category] = {
                'r2_linear': r2_linear,
                'r2_poly': r2_poly,
                'r2_piecewise': r2_piecewise,
                'improvement_poly': improvement_poly,
                'improvement_piecewise': improvement_piecewise,
                'has_nonlinearity': has_nonlinearity,
                'n_samples': len(X_clean)
            }
        
        self.nonlinearity_results = nonlinearity_results
        
    def generate_eda_report(self):
        """生成EDA报告"""
        print("\n=== 生成EDA报告 ===")
        
        report_content = []
        report_content.append("# 探索性数据分析报告")
        report_content.append("")
        
        # 周几效应总结
        report_content.append("## 周几效应分析")
        significant_categories = []
        for category, results in self.weekday_results.items():
            if results['is_significant']:
                significant_categories.append(category)
        
        report_content.append(f"- 分析品类数: {len(self.weekday_results)}")
        report_content.append(f"- 显著周几效应品类: {len(significant_categories)}")
        
        if significant_categories:
            report_content.append("- 显著品类列表:")
            for cat in significant_categories:
                p_val = self.weekday_results[cat]['p_value']
                report_content.append(f"  - {cat} (p={p_val:.4f})")
        
        report_content.append("")
        
        # 价格-销量关系总结
        report_content.append("## 价格-销量关系分析")
        report_content.append(f"- 分析品类数: {len(self.price_quantity_results)}")
        
        elasticity_summary = []
        r2_summary = []
        for category, results in self.price_quantity_results.items():
            elasticity_summary.append(results['overall_elasticity'])
            r2_summary.append(results['overall_r2'])
        
        if elasticity_summary:
            report_content.append(f"- 平均价格弹性: {np.mean(elasticity_summary):.3f}")
            report_content.append(f"- 弹性范围: [{np.min(elasticity_summary):.3f}, {np.max(elasticity_summary):.3f}]")
            report_content.append(f"- 平均R²: {np.mean(r2_summary):.3f}")
        
        report_content.append("")
        
        # 非线性检测总结
        if hasattr(self, 'nonlinearity_results'):
            report_content.append("## 非线性特征检测")
            nonlinear_categories = []
            for category, results in self.nonlinearity_results.items():
                if results['has_nonlinearity']:
                    nonlinear_categories.append(category)
            
            report_content.append(f"- 检测品类数: {len(self.nonlinearity_results)}")
            report_content.append(f"- 存在非线性特征品类: {len(nonlinear_categories)}")
            
            if nonlinear_categories:
                report_content.append("- 非线性品类:")
                for cat in nonlinear_categories:
                    results = self.nonlinearity_results[cat]
                    report_content.append(f"  - {cat} (二次改进: {results['improvement_poly']:.3f})")
        
        # 建议
        report_content.append("")
        report_content.append("## 建模建议")
        
        if significant_categories:
            report_content.append("- 建议在模型中加入周几哑变量")
        else:
            report_content.append("- 周几效应不显著，可不加入时间变量")
            
        if hasattr(self, 'nonlinearity_results') and nonlinear_categories:
            report_content.append("- 部分品类存在非线性特征，建议考虑:")
            report_content.append("  - 分段线性模型")
            report_content.append("  - 二次项或样条函数")
            report_content.append("  - GAM模型")
        
        # 保存报告
        report_text = "\n".join(report_content)
        with open('exploratory_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print("EDA报告已保存: exploratory_analysis_report.md")
        
    def run_full_analysis(self):
        """运行完整的探索性分析"""
        print("开始探索性分析...")
        
        self.load_data()
        self.analyze_weekday_effects()
        self.analyze_price_quantity_relationship()
        self.detect_nonlinearity()
        self.generate_eda_report()
        
        print("\n探索性分析完成！")
        return self

if __name__ == "__main__":
    analyzer = ExploratoryAnalyzer()
    analyzer.run_full_analysis()
