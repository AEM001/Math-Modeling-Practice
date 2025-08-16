# -*- coding: utf-8 -*-
"""
可视化模块
支持中文字体显示的图表生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json

# 导入字体配置模块
from font_config import setup_chinese_font, get_chinese_font

class Visualizer:
    """可视化器"""
    
    def __init__(self, config_path='config/config.json'):
        """初始化可视化器"""
        self.config = self.load_config(config_path)
        self.output_paths = self.config['output_paths']
        
        # 确保字体设置正确
        setup_chinese_font()
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        plt.style.use('default')
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_demand_model_performance(self, save_path=None):
        """绘制需求模型性能对比图"""
        try:
            # 加载建模结果
            results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            if not os.path.exists(results_path):
                print("需求模型结果文件不存在，跳过绘图")
                return None
                
            df = pd.read_csv(results_path)
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('需求建模性能分析', fontsize=16, fontweight='bold')
            
            # 1. 各品类模型性能对比
            ax1 = axes[0, 0]
            best_models = df[df['is_best'] == True]
            
            categories = best_models['category'].tolist()
            test_r2 = best_models['test_r2'].tolist()
            
            bars = ax1.bar(categories, test_r2, color='skyblue', alpha=0.7)
            ax1.set_title('各品类最佳模型测试R²', fontweight='bold')
            ax1.set_ylabel('测试R²')
            ax1.tick_params(axis='x', rotation=45)
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, test_r2):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 2. 模型类型性能分布
            ax2 = axes[0, 1]
            model_performance = df.groupby('model')['test_r2'].agg(['mean', 'std']).reset_index()
            
            x_pos = range(len(model_performance))
            means = model_performance['mean']
            stds = model_performance['std']
            
            bars2 = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                           color='lightcoral', alpha=0.7)
            ax2.set_title('不同模型类型性能对比', fontweight='bold')
            ax2.set_ylabel('平均测试R²')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(model_performance['model'], rotation=45)
            
            # 3. 价格弹性分析
            ax3 = axes[1, 0]
            elastic_data = df[df['price_elasticity'].notna() & (df['is_best'] == True)]
            
            if len(elastic_data) > 0:
                categories_elastic = elastic_data['category'].tolist()
                elasticities = elastic_data['price_elasticity'].tolist()
                
                bars3 = ax3.bar(categories_elastic, elasticities, 
                               color='lightgreen', alpha=0.7)
                ax3.set_title('各品类价格弹性', fontweight='bold')
                ax3.set_ylabel('价格弹性')
                ax3.tick_params(axis='x', rotation=45)
                ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # 添加数值标签
                for bar, value in zip(bars3, elasticities):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, '无价格弹性数据', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('各品类价格弹性', fontweight='bold')
            
            # 4. 训练vs测试性能散点图
            ax4 = axes[1, 1]
            train_r2 = df['train_r2']
            test_r2 = df['test_r2']
            
            # 按模型类型着色
            models = df['model'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = df[df['model'] == model]
                ax4.scatter(model_data['train_r2'], model_data['test_r2'], 
                           c=[colors[i]], label=model, alpha=0.7, s=60)
            
            # 添加对角线（理想情况）
            max_val = max(train_r2.max(), test_r2.max())
            ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='理想线')
            
            ax4.set_title('训练vs测试性能', fontweight='bold')
            ax4.set_xlabel('训练R²')
            ax4.set_ylabel('测试R²')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            if save_path is None:
                save_path = os.path.join(self.output_paths['figures_dir'], 'demand_model_performance.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"需求模型性能图已保存: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"绘制需求模型性能图失败: {e}")
            return None
    
    def plot_optimization_results(self, save_path=None):
        """绘制优化结果图"""
        try:
            # 加载优化结果
            results_path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
            weekly_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
            
            if not os.path.exists(results_path):
                print("优化结果文件不存在，跳过绘图")
                return None
                
            df_daily = pd.read_csv(results_path)
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('优化策略分析', fontsize=16, fontweight='bold')
            
            # 1. 各品类平均定价策略
            ax1 = axes[0, 0]
            avg_prices = df_daily.groupby('category')['optimal_price'].mean()
            
            bars1 = ax1.bar(avg_prices.index, avg_prices.values, 
                           color='gold', alpha=0.7)
            ax1.set_title('各品类平均定价', fontweight='bold')
            ax1.set_ylabel('平均价格 (元/千克)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars1, avg_prices.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 2. 各品类加成率分析
            ax2 = axes[0, 1]
            avg_markup = df_daily.groupby('category')['markup_ratio'].mean()
            
            bars2 = ax2.bar(avg_markup.index, avg_markup.values, 
                           color='lightblue', alpha=0.7)
            ax2.set_title('各品类平均加成率', fontweight='bold')
            ax2.set_ylabel('加成率')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars2, avg_markup.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 3. 各品类期望利润
            ax3 = axes[1, 0]
            total_profit = df_daily.groupby('category')['expected_profit'].sum()
            
            bars3 = ax3.bar(total_profit.index, total_profit.values, 
                           color='lightcoral', alpha=0.7)
            ax3.set_title('各品类周期望利润', fontweight='bold')
            ax3.set_ylabel('期望利润 (元)')
            ax3.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars3, total_profit.values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.0f}', ha='center', va='bottom')
            
            # 4. 日期趋势分析（选择一个代表性品类）
            ax4 = axes[1, 1]
            sample_category = df_daily['category'].iloc[0]
            cat_data = df_daily[df_daily['category'] == sample_category].copy()
            cat_data['date'] = pd.to_datetime(cat_data['date'])
            cat_data = cat_data.sort_values('date')
            
            # 双y轴
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(cat_data['date'], cat_data['optimal_price'], 
                           'o-', color='blue', label='最优价格', linewidth=2)
            line2 = ax4_twin.plot(cat_data['date'], cat_data['optimal_quantity'], 
                                's-', color='red', label='最优数量', linewidth=2)
            
            ax4.set_title(f'{sample_category} 定价与补货趋势', fontweight='bold')
            ax4.set_xlabel('日期')
            ax4.set_ylabel('价格 (元/千克)', color='blue')
            ax4_twin.set_ylabel('数量 (千克)', color='red')
            
            # 设置图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 格式化日期显示
            ax4.tick_params(axis='x', rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            if save_path is None:
                save_path = os.path.join(self.output_paths['figures_dir'], 'optimization_results.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"优化结果图已保存: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"绘制优化结果图失败: {e}")
            return None
    
    def create_summary_dashboard(self, save_path=None):
        """创建汇总仪表板"""
        try:
            # 创建一个综合性的仪表板
            fig = plt.figure(figsize=(20, 12))
            
            # 设置主标题
            fig.suptitle('蔬菜定价与补货策略分析仪表板', fontsize=20, fontweight='bold', y=0.95)
            
            # 创建网格布局
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. 数据概览 (左上角大块)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_data_overview(ax1)
            
            # 2. 模型性能概览 (右上角)
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_model_summary(ax2)
            
            # 3. 优化策略概览 (中间行)
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_optimization_summary(ax3)
            
            # 4. 利润分析 (中间右)
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_profit_analysis(ax4)
            
            # 5. 品类对比 (底部)
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_category_comparison(ax5)
            
            # 保存图像
            if save_path is None:
                save_path = os.path.join(self.output_paths['figures_dir'], 'summary_dashboard.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"汇总仪表板已保存: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"创建汇总仪表板失败: {e}")
            return None
    
    def _plot_data_overview(self, ax):
        """绘制数据概览"""
        try:
            # 这里可以显示一些基本的数据统计信息
            info_text = """
数据处理概览:
• 原始记录: 46,599 条
• 清洗后记录: 42,336 条  
• 数据保留率: 90.9%
• 品类数量: 6 个
• 单品数量: 251 个
• 时间跨度: 约 2 个月
            """
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
            ax.set_title('数据处理概览', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'数据概览加载失败: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_model_summary(self, ax):
        """绘制模型性能汇总"""
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                best_models = df[df['is_best'] == True]
                avg_r2 = best_models['test_r2'].mean()
                
                performance_text = f"""
模型性能汇总:
• 建模品类: {len(best_models)} 个
• 平均测试R²: {avg_r2:.3f}
• 最佳算法: RandomForest
• 交叉验证: 3折时间序列CV
• 特征数量: 28 个
                """
                ax.text(0.05, 0.95, performance_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
            else:
                ax.text(0.5, 0.5, '模型结果文件未找到', ha='center', va='center', transform=ax.transAxes)
                
            ax.set_title('模型性能汇总', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)  
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'模型汇总加载失败: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_optimization_summary(self, ax):
        """绘制优化策略汇总"""
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                avg_markup = df['markup_ratio'].mean()
                total_profit = df['expected_profit'].sum()
                
                opt_text = f"""
优化策略汇总:
• 优化时间窗口: 7 天
• 平均加成率: {avg_markup:.2f}
• 总期望利润: {total_profit:.0f} 元
• 优化算法: 启发式算法
• 服务水平: 80%
                """
                ax.text(0.05, 0.95, opt_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
            else:
                ax.text(0.5, 0.5, '优化结果文件未找到', ha='center', va='center', transform=ax.transAxes)
                
            ax.set_title('优化策略汇总', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'优化汇总加载失败: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_profit_analysis(self, ax):
        """绘制利润分析"""
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                
                # 绘制各品类利润分布饼图
                categories = df['category']
                profits = df['total_expected_profit']
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                wedges, texts, autotexts = ax.pie(profits, labels=categories, autopct='%1.1f%%',
                                                 colors=colors, startangle=90)
                
                ax.set_title('各品类利润占比', fontweight='bold')
            else:
                ax.text(0.5, 0.5, '周策略文件未找到', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'利润分析加载失败: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_category_comparison(self, ax):
        """绘制品类对比"""
        try:
            weekly_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
            if os.path.exists(weekly_path):
                df = pd.read_csv(weekly_path)
                
                categories = df['category']
                x_pos = np.arange(len(categories))
                
                # 多指标对比（标准化）
                avg_prices = df['avg_weekly_price'] / df['avg_weekly_price'].max()
                avg_markups = df['avg_markup_ratio'] / df['avg_markup_ratio'].max()  
                profits = df['total_expected_profit'] / df['total_expected_profit'].max()
                
                width = 0.25
                ax.bar(x_pos - width, avg_prices, width, label='平均价格 (标准化)', alpha=0.7)
                ax.bar(x_pos, avg_markups, width, label='加成率 (标准化)', alpha=0.7)
                ax.bar(x_pos + width, profits, width, label='利润 (标准化)', alpha=0.7)
                
                ax.set_title('各品类指标对比 (标准化)', fontweight='bold')
                ax.set_ylabel('标准化值')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '品类对比数据未找到', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'品类对比加载失败: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("=== 开始生成可视化图表 ===")
        
        results = {}
        
        # 1. 需求模型性能图
        print("生成需求模型性能图...")
        results['model_performance'] = self.plot_demand_model_performance()
        
        # 2. 优化结果图  
        print("生成优化结果图...")
        results['optimization_results'] = self.plot_optimization_results()
        
        # 3. 汇总仪表板
        print("生成汇总仪表板...")
        results['dashboard'] = self.create_summary_dashboard()
        
        print("=== 可视化图表生成完成 ===")
        return results

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.generate_all_visualizations()
