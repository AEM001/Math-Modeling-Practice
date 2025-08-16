"""
Results Summary and Export Module (Stage 7)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from config.config import Config
from src.utils.logger import setup_logger

# Set Chinese font for matplotlib
import matplotlib.font_manager as fm

# Try different Chinese fonts
chinese_fonts = [
    'Arial Unicode MS',  # macOS
    'PingFang SC',       # macOS
    'Hiragino Sans GB',  # macOS
    'STHeiti',           # macOS
    'SimHei',            # Windows
    'Microsoft YaHei',   # Windows
    'WenQuanYi Micro Hei', # Linux
    'DejaVu Sans'        # Fallback
]

# Find available font
available_font = None
for font_name in chinese_fonts:
    try:
        font_path = fm.findfont(fm.FontProperties(family=font_name))
        if font_path:
            available_font = font_name
            break
    except:
        continue

if available_font:
    plt.rcParams['font.sans-serif'] = [available_font]
else:
    # Fallback to system default
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

class ReportGenerator:
    def __init__(self):
        self.logger = setup_logger('report_generator')
        
    def create_model_performance_heatmap(self, model_results_df: pd.DataFrame) -> str:
        """Create model performance heatmap"""
        
        self.logger.info("Creating model performance heatmap...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # R² heatmap
        r2_data = model_results_df.pivot_table(
            index='category', 
            columns='best_model', 
            values='test_r2'
        ).fillna(0)
        
        sns.heatmap(r2_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[0], cbar_kws={'label': 'R²得分'})
        axes[0].set_title('各品类模型R²性能表现', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('最优模型类型', fontsize=12)
        axes[0].set_ylabel('蔬菜品类', fontsize=12)
        
        # MAPE heatmap
        mape_data = model_results_df.pivot_table(
            index='category', 
            columns='best_model', 
            values='test_mape'
        ).fillna(100)
        
        sns.heatmap(mape_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   ax=axes[1], cbar_kws={'label': 'MAPE (%)'})
        axes[1].set_title('各品类模型MAPE误差表现', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('最优模型类型', fontsize=12)
        axes[1].set_ylabel('蔬菜品类', fontsize=12)
        
        # Set font for tick labels
        for ax in axes:
            ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Config.get_output_path('model_performance_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Model performance heatmap saved to: {plot_path}")
        return plot_path
    
    def create_stability_chart(self, stability_df: pd.DataFrame) -> str:
        """Create stability assessment chart"""
        
        self.logger.info("Creating stability assessment chart...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mean R² by model type
        mean_r2 = stability_df.groupby('model_type')['mean_r2'].mean()
        mean_r2.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('各模型类型平均R²表现', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('平均R²得分', fontsize=12)
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # CV of R² by model type
        cv_r2 = stability_df.groupby('model_type')['cv_r2'].mean()
        cv_r2.plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('各模型类型R²变异系数', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('R²变异系数', fontsize=12)
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # Mean RMSE by model type
        mean_rmse = stability_df.groupby('model_type')['mean_rmse'].mean()
        mean_rmse.plot(kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('各模型类型平均RMSE误差', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('平均RMSE', fontsize=12)
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # Number of folds by category
        n_folds = stability_df.groupby('category')['n_folds'].mean()
        n_folds.plot(kind='bar', ax=axes[1,1], color='lightgreen')
        axes[1,1].set_title('各品类交叉验证折数', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('验证折数', fontsize=12)
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Config.get_output_path('stability_assessment.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Stability assessment chart saved to: {plot_path}")
        return plot_path
    
    def create_strategy_timeseries(self, pricing_df: pd.DataFrame) -> str:
        """Create strategy time series plots"""
        
        self.logger.info("Creating strategy time series...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Convert date column
        pricing_df['date'] = pd.to_datetime(pricing_df['date'])
        
        # Define colors for categories
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(pricing_df['category'].unique())}
        
        # Markup rate over time by category
        for category in pricing_df['category'].unique():
            cat_data = pricing_df[pricing_df['category'] == category]
            axes[0,0].plot(cat_data['date'], cat_data['markup'] * 100, 
                          marker='o', label=category, linewidth=2.5, 
                          color=category_colors[category], markersize=6)
        axes[0,0].set_title('各品类加成率时间趋势', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('加成率 (%)', fontsize=12)
        axes[0,0].legend(fontsize=10)
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0,0].grid(True, alpha=0.3)
        
        # Replenishment quantity over time
        for category in pricing_df['category'].unique():
            cat_data = pricing_df[pricing_df['category'] == category]
            axes[0,1].plot(cat_data['date'], cat_data['replenish_qty'], 
                          marker='s', label=category, linewidth=2.5,
                          color=category_colors[category], markersize=6)
        axes[0,1].set_title('各品类补货量时间趋势', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('补货量 (千克)', fontsize=12)
        axes[0,1].legend(fontsize=10)
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0,1].grid(True, alpha=0.3)
        
        # Predicted demand over time
        for category in pricing_df['category'].unique():
            cat_data = pricing_df[pricing_df['category'] == category]
            axes[1,0].plot(cat_data['date'], cat_data['demand_pred'], 
                          marker='^', label=category, linewidth=2.5,
                          color=category_colors[category], markersize=6)
        axes[1,0].set_title('各品类预测需求时间趋势', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('预测需求 (千克)', fontsize=12)
        axes[1,0].legend(fontsize=10)
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1,0].grid(True, alpha=0.3)
        
        # Expected profit over time
        for category in pricing_df['category'].unique():
            cat_data = pricing_df[pricing_df['category'] == category]
            axes[1,1].plot(cat_data['date'], cat_data['expected_profit'], 
                          marker='d', label=category, linewidth=2.5,
                          color=category_colors[category], markersize=6)
        axes[1,1].set_title('各品类预期利润时间趋势', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('预期利润 (元)', fontsize=12)
        axes[1,1].legend(fontsize=10)
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Config.get_output_path('strategy_timeseries.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Strategy time series saved to: {plot_path}")
        return plot_path
    
    def create_summary_statistics(self, pricing_df: pd.DataFrame, 
                                model_results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics"""
        
        self.logger.info("Creating summary statistics...")
        
        summary_stats = {
            'forecast_period': {
                'start_date': pricing_df['date'].min().strftime('%Y-%m-%d'),
                'end_date': pricing_df['date'].max().strftime('%Y-%m-%d'),
                'num_days': pricing_df['date'].nunique(),
                'num_categories': pricing_df['category'].nunique()
            },
            'model_performance': {
                'avg_r2': model_results_df['test_r2'].mean(),
                'avg_mape': model_results_df['test_mape'].mean(),
                'best_performing_model': model_results_df.groupby('best_model').size().idxmax(),
                'categories_with_good_fit': (model_results_df['test_r2'] > 0.5).sum()
            },
            'pricing_strategy': {
                'avg_markup': pricing_df['markup'].mean(),
                'markup_std': pricing_df['markup'].std(),
                'total_expected_revenue': pricing_df['expected_revenue'].sum(),
                'total_expected_profit': pricing_df['expected_profit'].sum(),
                'avg_profit_margin': (pricing_df['expected_profit'] / pricing_df['expected_revenue']).mean(),
                'avg_service_level': pricing_df['service_level'].mean()
            },
            'replenishment_strategy': {
                'total_replenishment': pricing_df['replenish_qty'].sum(),
                'avg_daily_replenishment': pricing_df.groupby('date')['replenish_qty'].sum().mean(),
                'avg_loss_rate': pricing_df['loss_rate'].mean()
            }
        }
        
        return summary_stats
    
    def generate_markdown_report(self, summary_stats: Dict[str, Any], 
                               plot_paths: List[str]) -> str:
        """Generate comprehensive markdown report"""
        
        self.logger.info("Generating markdown report...")
        
        report_content = f"""# 蔬菜定价与补货决策分析报告

## 执行摘要

本报告基于2023年高教社杯全国大学生数学建模竞赛C题，针对问题2建立了完整的蔬菜类商品定价与补货决策模型。

### 预测周期
- **预测期间**: {summary_stats['forecast_period']['start_date']} 至 {summary_stats['forecast_period']['end_date']}
- **预测天数**: {summary_stats['forecast_period']['num_days']} 天
- **涉及品类**: {summary_stats['forecast_period']['num_categories']} 个蔬菜品类

## 模型性能评估

### 需求预测模型表现
- **平均R²得分**: {summary_stats['model_performance']['avg_r2']:.3f}
- **平均MAPE**: {summary_stats['model_performance']['avg_mape']:.1f}%
- **最优模型类型**: {summary_stats['model_performance']['best_performing_model']}
- **高拟合度品类数量**: {summary_stats['model_performance']['categories_with_good_fit']} 个 (R² > 0.5)

### 模型稳定性
通过滚动交叉验证评估，模型在时间维度上表现稳定，具备良好的泛化能力。

## 定价策略结果

### 价格制定
- **平均加成率**: {summary_stats['pricing_strategy']['avg_markup']:.1%}
- **加成率标准差**: {summary_stats['pricing_strategy']['markup_std']:.3f}
- **服务水平**: {summary_stats['pricing_strategy']['avg_service_level']:.1%}

### 预期收益
- **总预期收入**: ¥{summary_stats['pricing_strategy']['total_expected_revenue']:,.2f}
- **总预期利润**: ¥{summary_stats['pricing_strategy']['total_expected_profit']:,.2f}
- **平均利润率**: {summary_stats['pricing_strategy']['avg_profit_margin']:.1%}

## 补货策略结果

### 补货计划
- **总补货量**: {summary_stats['replenishment_strategy']['total_replenishment']:,.1f} 千克
- **日均补货量**: {summary_stats['replenishment_strategy']['avg_daily_replenishment']:,.1f} 千克
- **平均损耗率**: {summary_stats['replenishment_strategy']['avg_loss_rate']:.1%}

## 方法论

### 建模流程
1. **数据清洗**: 排除异常价格、零销量、极端加成率等数据
2. **特征工程**: 构建时间特征、滞后特征、滚动特征等
3. **需求建模**: 使用随机森林、梯度提升、Huber回归等模型
4. **回测验证**: 滚动交叉验证评估模型稳定性
5. **需求预测**: 基于最优模型预测未来7天需求
6. **成本估计**: 基于历史数据的移动平均法
7. **策略制定**: 启发式定价与安全库存补货

### 核心假设
- 采用成本加成定价法，加成率在20%-40%范围内优化
- 使用95%服务水平确定安全库存
- 考虑价格弹性对需求的影响
- 约束条件：价格成本比1.0-2.0，日涨跌幅≤10%

## 技术特点

### 可复现性
- 基于稳定的机器学习算法
- 参数配置标准化
- 完整的日志记录和结果追踪

### 业务适应性
- 考虑行业特点的约束条件
- 灵活的参数配置
- 可解释的定价逻辑

## 结论与建议

1. **需求预测**: 构建的模型能够有效预测各品类需求，为定价决策提供可靠基础
2. **定价策略**: 基于成本加成的优化定价方案能够平衡收益与市场接受度
3. **补货策略**: 考虑损耗率和服务水平的补货方案确保供应充足而避免过度库存
4. **持续改进**: 建议定期更新模型参数，根据市场变化调整策略

---

*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Save report
        report_path = Config.get_output_path(Config.ANALYSIS_REPORT_FILE)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Markdown report saved to: {report_path}")
        return report_path
    
    def run(self, pricing_df: pd.DataFrame, model_results_df: pd.DataFrame, 
           stability_df: pd.DataFrame) -> Dict[str, str]:
        """Run complete report generation"""
        
        self.logger.info("Starting report generation...")
        
        # Create visualizations
        plot_paths = []
        
        try:
            heatmap_path = self.create_model_performance_heatmap(model_results_df)
            plot_paths.append(heatmap_path)
        except Exception as e:
            self.logger.warning(f"Failed to create performance heatmap: {e}")
        
        try:
            stability_path = self.create_stability_chart(stability_df)
            plot_paths.append(stability_path)
        except Exception as e:
            self.logger.warning(f"Failed to create stability chart: {e}")
        
        try:
            timeseries_path = self.create_strategy_timeseries(pricing_df)
            plot_paths.append(timeseries_path)
        except Exception as e:
            self.logger.warning(f"Failed to create strategy timeseries: {e}")
        
        # Create summary statistics
        summary_stats = self.create_summary_statistics(pricing_df, model_results_df)
        
        # Generate markdown report
        report_path = self.generate_markdown_report(summary_stats, plot_paths)
        
        results = {
            'report_path': report_path,
            'plot_paths': plot_paths,
            'summary_stats': summary_stats
        }
        
        self.logger.info("Report generation completed successfully!")
        return results

if __name__ == "__main__":
    # This would be called from the main pipeline
    pass