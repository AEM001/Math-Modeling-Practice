# -*- coding: utf-8 -*-
"""
English Visualization Module
Generate charts with English text to avoid font issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json

class EnglishVisualizer:
    """English Visualizer"""
    
    def __init__(self, config_path='config/config.json'):
        """Initialize visualizer"""
        self.config = self.load_config(config_path)
        self.output_paths = self.config['output_paths']
        
        # Set matplotlib style
        sns.set_style("whitegrid")
        plt.style.use('default')
        plt.rcParams['axes.unicode_minus'] = False
        
        # Category name translations
        self.category_translations = {
            '水生根茎类': 'Aquatic Root Vegetables',
            '花叶类': 'Leafy Vegetables',
            '花菜类': 'Cauliflower', 
            '茄类': 'Eggplant',
            '辣椒类': 'Chili Pepper',
            '食用菌': 'Mushroom'
        }
        
    def load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def translate_categories(self, categories):
        """Translate Chinese categories to English"""
        translated = []
        for cat in categories:
            translated.append(self.category_translations.get(cat, cat))
        return translated
    
    def plot_demand_model_performance(self, save_path=None):
        """Plot demand model performance comparison"""
        try:
            # Load modeling results
            results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            if not os.path.exists(results_path):
                print("Demand model results file not found, skipping plot")
                return None
                
            df = pd.read_csv(results_path)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Demand Model Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Best model performance by category
            ax1 = axes[0, 0]
            best_models = df[df['is_best'] == True]
            
            categories = self.translate_categories(best_models['category'].tolist())
            test_r2 = best_models['test_r2'].tolist()
            
            bars = ax1.bar(categories, test_r2, color='skyblue', alpha=0.7)
            ax1.set_title('Best Model Test R² by Category', fontweight='bold')
            ax1.set_ylabel('Test R²')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, test_r2):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 2. Model type performance distribution
            ax2 = axes[0, 1]
            model_performance = df.groupby('model')['test_r2'].agg(['mean', 'std']).reset_index()
            
            x_pos = range(len(model_performance))
            means = model_performance['mean']
            stds = model_performance['std']
            
            bars2 = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                           color='lightcoral', alpha=0.7)
            ax2.set_title('Performance Comparison by Model Type', fontweight='bold')
            ax2.set_ylabel('Average Test R²')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(model_performance['model'], rotation=45)
            
            # 3. Price elasticity analysis
            ax3 = axes[1, 0]
            elastic_data = df[df['price_elasticity'].notna() & (df['is_best'] == True)]
            
            if len(elastic_data) > 0:
                categories_elastic = self.translate_categories(elastic_data['category'].tolist())
                elasticities = elastic_data['price_elasticity'].tolist()
                
                bars3 = ax3.bar(categories_elastic, elasticities, 
                               color='lightgreen', alpha=0.7)
                ax3.set_title('Price Elasticity by Category', fontweight='bold')
                ax3.set_ylabel('Price Elasticity')
                ax3.tick_params(axis='x', rotation=45)
                ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Add value labels
                for bar, value in zip(bars3, elasticities):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No Price Elasticity Data', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Price Elasticity by Category', fontweight='bold')
            
            # 4. Training vs Testing performance scatter plot
            ax4 = axes[1, 1]
            train_r2 = df['train_r2']
            test_r2 = df['test_r2']
            
            # Color by model type
            models = df['model'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = df[df['model'] == model]
                ax4.scatter(model_data['train_r2'], model_data['test_r2'], 
                           c=[colors[i]], label=model, alpha=0.7, s=60)
            
            # Add diagonal line (ideal case)
            max_val = max(train_r2.max(), test_r2.max())
            ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Ideal Line')
            
            ax4.set_title('Training vs Testing Performance', fontweight='bold')
            ax4.set_xlabel('Training R²')
            ax4.set_ylabel('Testing R²')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_path is None:
                save_path = os.path.join(self.output_paths['figures_dir'], 'demand_model_performance_en.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Demand model performance chart saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Failed to plot demand model performance: {e}")
            return None
    
    def plot_optimization_results(self, save_path=None):
        """Plot optimization results"""
        try:
            # Load optimization results
            results_path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
            
            if not os.path.exists(results_path):
                print("Optimization results file not found, skipping plot")
                return None
                
            df_daily = pd.read_csv(results_path)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Optimization Strategy Analysis', fontsize=16, fontweight='bold')
            
            # 1. Average pricing strategy by category
            ax1 = axes[0, 0]
            avg_prices = df_daily.groupby('category')['optimal_price'].mean()
            categories = self.translate_categories(avg_prices.index.tolist())
            
            bars1 = ax1.bar(categories, avg_prices.values, 
                           color='gold', alpha=0.7)
            ax1.set_title('Average Pricing by Category', fontweight='bold')
            ax1.set_ylabel('Average Price (Yuan/kg)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars1, avg_prices.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 2. Markup ratio analysis by category
            ax2 = axes[0, 1]
            avg_markup = df_daily.groupby('category')['markup_ratio'].mean()
            categories = self.translate_categories(avg_markup.index.tolist())
            
            bars2 = ax2.bar(categories, avg_markup.values, 
                           color='lightblue', alpha=0.7)
            ax2.set_title('Average Markup Ratio by Category', fontweight='bold')
            ax2.set_ylabel('Markup Ratio')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars2, avg_markup.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 3. Expected profit by category
            ax3 = axes[1, 0]
            total_profit = df_daily.groupby('category')['expected_profit'].sum()
            categories = self.translate_categories(total_profit.index.tolist())
            
            bars3 = ax3.bar(categories, total_profit.values, 
                           color='lightcoral', alpha=0.7)
            ax3.set_title('Weekly Expected Profit by Category', fontweight='bold')
            ax3.set_ylabel('Expected Profit (Yuan)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars3, total_profit.values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.0f}', ha='center', va='bottom')
            
            # 4. Date trend analysis (representative category)
            ax4 = axes[1, 1]
            sample_category = df_daily['category'].iloc[0]
            cat_data = df_daily[df_daily['category'] == sample_category].copy()
            cat_data['date'] = pd.to_datetime(cat_data['date'])
            cat_data = cat_data.sort_values('date')
            
            # Dual y-axis
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(cat_data['date'], cat_data['optimal_price'], 
                           'o-', color='blue', label='Optimal Price', linewidth=2)
            line2 = ax4_twin.plot(cat_data['date'], cat_data['optimal_quantity'], 
                                's-', color='red', label='Optimal Quantity', linewidth=2)
            
            translated_category = self.category_translations.get(sample_category, sample_category)
            ax4.set_title(f'{translated_category} Pricing & Inventory Trend', fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Price (Yuan/kg)', color='blue')
            ax4_twin.set_ylabel('Quantity (kg)', color='red')
            
            # Set legend
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format date display
            ax4.tick_params(axis='x', rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_path is None:
                save_path = os.path.join(self.output_paths['figures_dir'], 'optimization_results_en.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Optimization results chart saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Failed to plot optimization results: {e}")
            return None
    
    def create_summary_dashboard(self, save_path=None):
        """Create summary dashboard"""
        try:
            # Create comprehensive dashboard
            fig = plt.figure(figsize=(20, 12))
            
            # Set main title
            fig.suptitle('Vegetable Pricing & Inventory Strategy Analysis Dashboard', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            # Create grid layout
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Data overview (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_data_overview(ax1)
            
            # 2. Model performance overview (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_model_summary(ax2)
            
            # 3. Optimization strategy overview (middle row)
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_optimization_summary(ax3)
            
            # 4. Profit analysis (middle right)
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_profit_analysis(ax4)
            
            # 5. Category comparison (bottom)
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_category_comparison(ax5)
            
            # Save figure
            if save_path is None:
                save_path = os.path.join(self.output_paths['figures_dir'], 'summary_dashboard_en.png')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Summary dashboard saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Failed to create summary dashboard: {e}")
            return None
    
    def _plot_data_overview(self, ax):
        """Plot data overview"""
        try:
            info_text = """
Data Processing Overview:
• Original records: 46,599 entries
• Cleaned records: 42,336 entries  
• Data retention rate: 90.9%
• Categories: 6 types
• Products: 251 items
• Time span: ~2 months
            """
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
            ax.set_title('Data Processing Overview', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Data overview loading failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_model_summary(self, ax):
        """Plot model performance summary"""
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'demand_model_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                best_models = df[df['is_best'] == True]
                avg_r2 = best_models['test_r2'].mean()
                
                performance_text = f"""
Model Performance Summary:
• Modeled categories: {len(best_models)}
• Average test R²: {avg_r2:.3f}
• Best algorithm: RandomForest
• Cross validation: 3-fold time series CV
• Feature count: 28
                """
                ax.text(0.05, 0.95, performance_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'Model results file not found', ha='center', va='center', 
                       transform=ax.transAxes)
                
            ax.set_title('Model Performance Summary', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)  
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Model summary loading failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_optimization_summary(self, ax):
        """Plot optimization strategy summary"""
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'optimization_results.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                avg_markup = df['markup_ratio'].mean()
                total_profit = df['expected_profit'].sum()
                
                opt_text = f"""
Optimization Strategy Summary:
• Optimization window: 7 days
• Average markup ratio: {avg_markup:.2f}
• Total expected profit: {total_profit:.0f} Yuan
• Algorithm: Heuristic optimization
• Service level: 80%
                """
                ax.text(0.05, 0.95, opt_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'Optimization results file not found', ha='center', va='center', 
                       transform=ax.transAxes)
                
            ax.set_title('Optimization Strategy Summary', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Optimization summary loading failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_profit_analysis(self, ax):
        """Plot profit analysis"""
        try:
            results_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                
                # Plot profit distribution pie chart by category
                categories = self.translate_categories(df['category'].tolist())
                profits = df['total_expected_profit']
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                wedges, texts, autotexts = ax.pie(profits, labels=categories, autopct='%1.1f%%',
                                                 colors=colors, startangle=90)
                
                ax.set_title('Profit Share by Category', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Weekly strategy file not found', ha='center', va='center', 
                       transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Profit analysis loading failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_category_comparison(self, ax):
        """Plot category comparison"""
        try:
            weekly_path = os.path.join(self.output_paths['results_dir'], 'weekly_strategy.csv')
            if os.path.exists(weekly_path):
                df = pd.read_csv(weekly_path)
                
                categories = self.translate_categories(df['category'].tolist())
                x_pos = np.arange(len(categories))
                
                # Multi-metric comparison (normalized)
                avg_prices = df['avg_weekly_price'] / df['avg_weekly_price'].max()
                avg_markups = df['avg_markup_ratio'] / df['avg_markup_ratio'].max()  
                profits = df['total_expected_profit'] / df['total_expected_profit'].max()
                
                width = 0.25
                ax.bar(x_pos - width, avg_prices, width, label='Avg Price (normalized)', alpha=0.7)
                ax.bar(x_pos, avg_markups, width, label='Markup Ratio (normalized)', alpha=0.7)
                ax.bar(x_pos + width, profits, width, label='Profit (normalized)', alpha=0.7)
                
                ax.set_title('Category Metrics Comparison (Normalized)', fontweight='bold')
                ax.set_ylabel('Normalized Value')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Category comparison data not found', ha='center', va='center', 
                       transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Category comparison loading failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("=== Starting English visualization generation ===")
        
        results = {}
        
        # 1. Demand model performance chart
        print("Generating demand model performance chart...")
        results['model_performance'] = self.plot_demand_model_performance()
        
        # 2. Optimization results chart  
        print("Generating optimization results chart...")
        results['optimization_results'] = self.plot_optimization_results()
        
        # 3. Summary dashboard
        print("Generating summary dashboard...")
        results['dashboard'] = self.create_summary_dashboard()
        
        print("=== English visualization generation completed ===")
        return results

if __name__ == "__main__":
    visualizer = EnglishVisualizer()
    visualizer.generate_all_visualizations()
