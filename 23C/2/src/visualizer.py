import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import json
import os
import platform
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """设置中文字体支持（更稳健的检测方式）"""
    system = platform.system()

    # 按平台准备候选字体（按优先级排序）
    if system == 'Darwin':  # macOS
        font_candidates = [
            'PingFang SC', 'Hiragino Sans GB', 'Songti SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS'
        ]
    elif system == 'Windows':
        font_candidates = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'NSimSun', 'FangSong'
        ]
    else:  # Linux
        font_candidates = [
            'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN'
        ]

    # 收集系统字体名称（包含变体名）
    try:
        available_names = [f.name for f in fm.fontManager.ttflist]
    except Exception:
        available_names = []

    selected = None
    # 先精确匹配，再模糊匹配（处理带权重/变体的命名）
    for cand in font_candidates:
        if cand in available_names:
            selected = cand
            break
        for name in available_names:
            if cand.lower() in name.lower():
                selected = name
                break
        if selected:
            break

    # 统一设置 Matplotlib 字体
    plt.rcParams['axes.unicode_minus'] = False
    if selected:
        plt.rcParams['font.family'] = 'sans-serif'
        current_list = [f for f in plt.rcParams.get('font.sans-serif', []) if f != selected]
        plt.rcParams['font.sans-serif'] = [selected] + current_list + ['DejaVu Sans']
        print(f"✓ 使用中文字体: {selected}")
        return selected
    else:
        # 保底：尽量让中文不至于乱码
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        print("⚠ 未找到中文字体，已尝试使用兼容字体（可能仍有异常）")
        return None

class OptimizationVisualizer:
    """优化结果可视化器"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化可视化器"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        with open(os.path.join(project_root, config_path), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.output_paths = {
            'results_dir': os.path.join(project_root, self.config['output_paths']['results_dir']),
            'figures_dir': os.path.join(project_root, self.config['output_paths']['figures_dir']),
            'reports_dir': os.path.join(project_root, self.config['output_paths']['reports_dir'])
        }
        
        # 设置中文字体
        self.font_name = setup_chinese_fonts()
        
        # 设置样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
    def load_results(self):
        """加载优化结果数据"""
        results_path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
        weekly_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Optimization results not found: {results_path}")
        
        self.daily_results = pd.read_csv(results_path)
        self.daily_results['date'] = pd.to_datetime(self.daily_results['date'])
        
        if os.path.exists(weekly_path):
            self.weekly_summary = pd.read_csv(weekly_path)
        else:
            print(f"Warning: Weekly summary not found at {weekly_path}")
            self.weekly_summary = None
            
    def plot_profit_by_category_and_date(self):
        """绘制各品类每日利润热力图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建透视表
        profit_pivot = self.daily_results.pivot(
            index='category', columns='date', values='expected_profit'
        )
        
        # 绘制热力图
        sns.heatmap(profit_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': '期望利润 (元)'})
        
        ax.set_title('各品类每日期望利润热力图 (2023-07-01至07-07)', fontsize=16, pad=20)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('品类', fontsize=12)
        
        # 旋转日期标签
        ax.set_xticklabels([d.strftime('%m-%d') for d in profit_pivot.columns], rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(self.output_paths['figures_dir'], exist_ok=True)
        fig.savefig(os.path.join(self.output_paths['figures_dir'], 'profit_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_pricing_strategy(self):
        """绘制定价策略图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        categories = self.daily_results['category'].unique()
        
        for i, category in enumerate(categories):
            if i >= len(axes):
                break
                
            cat_data = self.daily_results[self.daily_results['category'] == category]
            
            ax = axes[i]
            ax2 = ax.twinx()
            
            # 绘制价格和数量
            line1 = ax.plot(cat_data['date'], cat_data['optimal_price'], 
                           'b-o', label='最优价格', linewidth=2, markersize=6)
            line2 = ax2.plot(cat_data['date'], cat_data['optimal_quantity'], 
                            'r-s', label='最优订货量', linewidth=2, markersize=6)
            
            ax.set_xlabel('日期', fontsize=10)
            ax.set_ylabel('价格 (元/kg)', color='b', fontsize=10)
            ax2.set_ylabel('订货量 (kg)', color='r', fontsize=10)
            ax.set_title(f'{category} - 定价与订货策略', fontsize=12)
            
            # 设置日期格式
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
            
        # 隐藏多余的子图
        for i in range(len(categories), len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle('各品类一周定价与订货策略', fontsize=16, y=0.98)
        plt.tight_layout()
        
        fig.savefig(os.path.join(self.output_paths['figures_dir'], 'pricing_strategy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_service_metrics(self):
        """绘制服务水平指标图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 服务率对比
        service_by_cat = self.daily_results.groupby('category')['service_rate'].mean()
        bars1 = ax1.bar(range(len(service_by_cat)), service_by_cat.values, color='skyblue', alpha=0.8)
        ax1.set_xlabel('品类', fontsize=12)
        ax1.set_ylabel('平均服务率', fontsize=12)
        ax1.set_title('各品类平均服务率', fontsize=14)
        ax1.set_xticks(range(len(service_by_cat)))
        ax1.set_xticklabels(service_by_cat.index, rotation=45)
        ax1.set_ylim(0, 1.1)
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. 滞销量分析
        leftover_by_cat = self.daily_results.groupby('category')['expected_leftover'].mean()
        bars2 = ax2.bar(range(len(leftover_by_cat)), leftover_by_cat.values, color='orange', alpha=0.8)
        ax2.set_xlabel('品类', fontsize=12)
        ax2.set_ylabel('平均滞销量 (kg)', fontsize=12)
        ax2.set_title('各品类平均滞销量', fontsize=14)
        ax2.set_xticks(range(len(leftover_by_cat)))
        ax2.set_xticklabels(leftover_by_cat.index, rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(leftover_by_cat.values) * 0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 3. 缺货量分析
        stockout_by_cat = self.daily_results.groupby('category')['expected_stockout'].mean()
        bars3 = ax3.bar(range(len(stockout_by_cat)), stockout_by_cat.values, color='red', alpha=0.7)
        ax3.set_xlabel('品类', fontsize=12)
        ax3.set_ylabel('平均缺货量 (kg)', fontsize=12)
        ax3.set_title('各品类平均缺货量', fontsize=14)
        ax3.set_xticks(range(len(stockout_by_cat)))
        ax3.set_xticklabels(stockout_by_cat.index, rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(stockout_by_cat.values) * 0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 4. 加成比例分布
        markup_by_cat = self.daily_results.groupby('category')['markup_ratio'].mean()
        bars4 = ax4.bar(range(len(markup_by_cat)), markup_by_cat.values, color='green', alpha=0.7)
        ax4.set_xlabel('品类', fontsize=12)
        ax4.set_ylabel('平均加成比例', fontsize=12)
        ax4.set_title('各品类平均加成比例', fontsize=14)
        ax4.set_xticks(range(len(markup_by_cat)))
        ax4.set_xticklabels(markup_by_cat.index, rotation=45)
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='无加成线')
        
        # 添加数值标签
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax4.legend()
        
        plt.suptitle('服务水平与运营指标分析', fontsize=16, y=0.98)
        plt.tight_layout()
        
        fig.savefig(os.path.join(self.output_paths['figures_dir'], 'service_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_weekly_summary(self):
        """绘制周汇总指标图"""
        if self.weekly_summary is None:
            print("Warning: No weekly summary data available for plotting")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 总利润对比
        bars1 = ax1.bar(range(len(self.weekly_summary)), 
                       self.weekly_summary['total_expected_profit'], 
                       color=['green' if x > 0 else 'red' for x in self.weekly_summary['total_expected_profit']],
                       alpha=0.8)
        ax1.set_xlabel('品类', fontsize=12)
        ax1.set_ylabel('周总利润 (元)', fontsize=12)
        ax1.set_title('各品类周总利润', fontsize=14)
        ax1.set_xticks(range(len(self.weekly_summary)))
        ax1.set_xticklabels(self.weekly_summary['category'], rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    height + (max(self.weekly_summary['total_expected_profit']) * 0.01 if height > 0 else min(self.weekly_summary['total_expected_profit']) * 0.01),
                    f'{height:.0f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 2. 收入成本对比
        x = range(len(self.weekly_summary))
        width = 0.35
        bars2a = ax2.bar([i - width/2 for i in x], self.weekly_summary['total_revenue'], 
                        width, label='总收入', color='lightblue', alpha=0.8)
        bars2b = ax2.bar([i + width/2 for i in x], self.weekly_summary['total_cost'], 
                        width, label='总成本', color='lightcoral', alpha=0.8)
        ax2.set_xlabel('品类', fontsize=12)
        ax2.set_ylabel('金额 (元)', fontsize=12)
        ax2.set_title('各品类周收入成本对比', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.weekly_summary['category'], rotation=45)
        ax2.legend()
        
        # 3. 平均订货量
        bars3 = ax3.bar(range(len(self.weekly_summary)), 
                       self.weekly_summary['avg_optimal_quantity'], 
                       color='mediumpurple', alpha=0.8)
        ax3.set_xlabel('品类', fontsize=12)
        ax3.set_ylabel('平均订货量 (kg)', fontsize=12)
        ax3.set_title('各品类平均日订货量', fontsize=14)
        ax3.set_xticks(range(len(self.weekly_summary)))
        ax3.set_xticklabels(self.weekly_summary['category'], rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(self.weekly_summary['avg_optimal_quantity']) * 0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 4. 服务水平
        bars4 = ax4.bar(range(len(self.weekly_summary)), 
                       self.weekly_summary['avg_service_rate'], 
                       color='gold', alpha=0.8)
        ax4.set_xlabel('品类', fontsize=12)
        ax4.set_ylabel('平均服务率', fontsize=12)
        ax4.set_title('各品类平均服务率', fontsize=14)
        ax4.set_xticks(range(len(self.weekly_summary)))
        ax4.set_xticklabels(self.weekly_summary['category'], rotation=45)
        ax4.set_ylim(0, 1.1)
        
        # 添加数值标签
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('一周策略汇总指标', fontsize=16, y=0.98)
        plt.tight_layout()
        
        fig.savefig(os.path.join(self.output_paths['figures_dir'], 'weekly_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_dashboard(self):
        """生成汇总仪表板"""
        fig = plt.figure(figsize=(20, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 总利润趋势 (跨越两列)
        ax1 = fig.add_subplot(gs[0, :2])
        daily_profit = self.daily_results.groupby('date')['expected_profit'].sum()
        ax1.plot(daily_profit.index, daily_profit.values, 'b-o', linewidth=3, markersize=8)
        ax1.set_title('每日总利润趋势', fontsize=14, fontweight='bold')
        ax1.set_ylabel('总利润 (元)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 品类利润占比饼图
        ax2 = fig.add_subplot(gs[0, 2:])
        if self.weekly_summary is not None:
            positive_profits = self.weekly_summary[self.weekly_summary['total_expected_profit'] > 0]
            if not positive_profits.empty:
                ax2.pie(positive_profits['total_expected_profit'], 
                       labels=positive_profits['category'],
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('正利润品类贡献占比', fontsize=14, fontweight='bold')
        
        # 3. 价格分布箱线图
        ax3 = fig.add_subplot(gs[1, :2])
        price_data = []
        categories = []
        for cat in self.daily_results['category'].unique():
            cat_prices = self.daily_results[self.daily_results['category'] == cat]['optimal_price']
            price_data.append(cat_prices)
            categories.append(cat)
        
        ax3.boxplot(price_data, labels=categories)
        ax3.set_title('各品类价格分布', fontsize=14, fontweight='bold')
        ax3.set_ylabel('价格 (元/kg)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 订货量分布箱线图
        ax4 = fig.add_subplot(gs[1, 2:])
        quantity_data = []
        for cat in self.daily_results['category'].unique():
            cat_quantities = self.daily_results[self.daily_results['category'] == cat]['optimal_quantity']
            quantity_data.append(cat_quantities)
        
        ax4.boxplot(quantity_data, labels=categories)
        ax4.set_title('各品类订货量分布', fontsize=14, fontweight='bold')
        ax4.set_ylabel('订货量 (kg)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 关键指标表格
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        if self.weekly_summary is not None:
            # 创建表格数据
            table_data = []
            for _, row in self.weekly_summary.iterrows():
                table_data.append([
                    row['category'],
                    f"{row['avg_optimal_price']:.2f}",
                    f"{row['avg_optimal_quantity']:.1f}",
                    f"{row['total_expected_profit']:.0f}",
                    f"{row['avg_service_rate']:.3f}"
                ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['品类', '平均价格(元/kg)', '平均订货量(kg)', '周总利润(元)', '服务率'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # 设置表头样式
            for i in range(5):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('一周策略汇总表', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('2023年7月1-7日 蔬菜定价与补货策略分析仪表板', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        fig.savefig(os.path.join(self.output_paths['figures_dir'], 'summary_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("正在生成可视化图表...")
        
        self.load_results()
        
        visualizations = {
            'profit_heatmap': self.plot_profit_by_category_and_date,
            'pricing_strategy': self.plot_pricing_strategy,
            'service_metrics': self.plot_service_metrics,
            'weekly_summary': self.plot_weekly_summary,
            'summary_dashboard': self.generate_summary_dashboard
        }
        
        results = {}
        for name, func in visualizations.items():
            try:
                func()
                results[name] = True
                print(f"  ✓ {name} 生成成功")
            except Exception as e:
                results[name] = False
                print(f"  ✗ {name} 生成失败: {e}")
        
        print(f"\n可视化完成: {sum(results.values())}/{len(results)} 个图表生成成功")
        return results

if __name__ == '__main__':
    visualizer = OptimizationVisualizer()
    visualizer.generate_all_visualizations()
