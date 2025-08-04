import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def set_chinese_font():
    """设置中文字体"""
    chinese_fonts = ['PingFang HK', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except:
            continue
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def create_model_results_summary():
    """创建模型结果总结图表"""
    set_chinese_font()
    
    # 创建图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 模型性能结果
    ax1 = plt.subplot(2, 3, 1)
    models = ['转化率模型', '选择性模型']
    test_scores = [0.760, 0.696]
    train_scores = [0.826, 0.794]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_scores, width, label='训练集', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_scores, width, label='测试集', color='lightgreen', alpha=0.8)
    
    # 添加数值标签
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        ax1.text(i - width/2, train + 0.005, f'{train:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, test + 0.005, f'{test:.3f}', ha='center', va='bottom')
    
    ax1.set_xlabel('模型类型')
    ax1.set_ylabel('R² 得分')
    ax1.set_title('模型性能结果')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 特征重要性对比
    ax2 = plt.subplot(2, 3, 2)
    features = ['Co/SiO2用量', 'HAP用量', '温度', '装料质量比', '投料方式', '乙醇浓度', 'Co负载量']
    conv_importance = [0.491, 0.397, 0.098, 0.011, 0.002, 0.002, 0.000]
    sel_importance = [0.566, 0.420, 0.008, 0.004, 0.001, 0.000, 0.001]
    
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, conv_importance, width, label='转化率模型', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, sel_importance, width, label='选择性模型', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('特征')
    ax2.set_ylabel('重要性权重')
    ax2.set_title('特征重要性对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 配置组合测试结果
    ax3 = plt.subplot(2, 3, 3)
    configs = ['极简+保守', '极简+标准', '极简+激进', '极简+精细', '简单+保守', '简单+标准', '简单+激进', '简单+精细']
    conv_scores = [0.502, 0.511, 0.508, 0.507, 0.504, 0.504, 0.504, 0.504]
    sel_scores = [0.463, 0.434, 0.421, 0.513, 0.257, 0.104, 0.032, -0.011]
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, conv_scores, width, label='转化率模型', color='lightblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, sel_scores, width, label='选择性模型', color='lightpink', alpha=0.8)
    
    ax3.set_xlabel('配置组合')
    ax3.set_ylabel('综合得分')
    ax3.set_title('配置组合测试结果')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 数据分割策略对比
    ax4 = plt.subplot(2, 3, 4)
    strategies = ['标准', '保守', '激进', '随机1', '随机2']
    similarity_scores = [0.947, 0.880, 0.974, 0.931, 0.936]
    colors = ['lightgray', 'lightgray', 'lightgreen', 'lightgray', 'lightgray']
    
    bars = ax4.bar(strategies, similarity_scores, color=colors, alpha=0.8)
    ax4.set_xlabel('分割策略')
    ax4.set_ylabel('分布相似性')
    ax4.set_title('数据分割策略对比')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, similarity_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 5. 过拟合程度分析
    ax5 = plt.subplot(2, 3, 5)
    models = ['转化率模型', '选择性模型']
    overfitting_scores = [0.067, 0.098]
    
    bars = ax5.bar(models, overfitting_scores, color='lightgreen', alpha=0.8)
    ax5.set_xlabel('模型类型')
    ax5.set_ylabel('过拟合程度')
    ax5.set_title('过拟合程度分析')
    ax5.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, overfitting_scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 6. 模型复杂度vs性能
    ax6 = plt.subplot(2, 3, 6)
    complexity_levels = ['极简', '简单', '中等', '复杂', '极复杂']
    spline_counts = [3, 5, 8, 12, 15]
    performance_scores = [0.511, 0.504, 0.502, 0.499, 0.370]  # 转化率模型综合得分
    
    bars = ax6.bar(complexity_levels, performance_scores, color='lightsteelblue', alpha=0.8)
    ax6.set_xlabel('模型复杂度')
    ax6.set_ylabel('综合得分')
    ax6.set_title('模型复杂度vs性能')
    ax6.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, performance_scores):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('GAM_模型分析总结.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_parameter_importance_chart():
    """创建参数重要性图表"""
    set_chinese_font()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 参数影响程度
    parameters = ['样条数量', '正则化强度', '数据分割比例', '样条阶数', '交叉验证折数', '特征重要性权重', '随机种子', '异常值阈值', '数据缩放方法']
    impact_levels = ['高', '高', '高', '中', '中', '中', '低', '低', '低']
    impact_scores = [9, 9, 9, 6, 6, 6, 3, 3, 3]
    
    colors = ['red' if level == '高' else 'orange' if level == '中' else 'green' for level in impact_levels]
    
    bars = ax1.barh(parameters, impact_scores, color=colors, alpha=0.7)
    ax1.set_xlabel('影响程度 (1-10)')
    ax1.set_title('参数影响程度分析')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, impact_scores):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score}', ha='left', va='center')
    
    # 建模策略效果
    strategies = ['数据分割优化', '模型配置优化', '评分策略优化', '特征工程优化', '正则化优化']
    effectiveness = [0.85, 0.90, 0.80, 0.75, 0.88]
    
    bars = ax2.bar(strategies, effectiveness, color='lightblue', alpha=0.8)
    ax2.set_ylabel('效果评分 (0-1)')
    ax2.set_title('建模策略效果评估')
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, effectiveness):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('参数重要性分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("生成GAM模型分析总结图表...")
    
    # 创建主要总结图表
    fig1 = create_model_results_summary()
    print("✓ 主要总结图表已生成: GAM_模型分析总结.png")
    
    # 创建参数重要性图表
    fig2 = create_parameter_importance_chart()
    print("✓ 参数重要性图表已生成: 参数重要性分析.png")
    
    print("\n图表生成完成！") 