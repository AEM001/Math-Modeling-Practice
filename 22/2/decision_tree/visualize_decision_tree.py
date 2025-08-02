import os
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# --- 解决matplotlib中文显示问题 ---
# 指定macOS系统中的中文字体文件路径
# 通过 system_profiler 找到的 PingFang SC 字体精确路径
font_path = '/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc' 
try:
    # 使用FontProperties加载字体
    cn_font = FontProperties(fname=font_path, size=12)
    cn_font_title = FontProperties(fname=font_path, size=18)
    cn_font_legend_title = FontProperties(fname=font_path, size=14)
    print(f"中文字体 '{font_path}' 加载成功。")
except FileNotFoundError:
    print(f"错误: 字体文件未在 '{font_path}' 找到。")
    print("请确认您的macOS字体路径，或尝试 'STHeiti.ttf' 等其他中文字体。")
    # 如果找不到字体，则提供一个备用方案或退出
    cn_font = FontProperties(size=12) # 使用默认字体
    cn_font_title = FontProperties(size=18)
    cn_font_legend_title = FontProperties(size=14)


print("--- 开始根据决策规则生成可视化 ---")

# --- 1. 定义路径和参数 ---
output_base_dir = '2/decision_tree/'
output_plot_path = os.path.join(output_base_dir, 'decision_tree_visualization_from_rules.png')
os.makedirs(output_base_dir, exist_ok=True)

# 定义颜色
colors = {
    'lead_barium': '#a6d9b3',  # 铅钡 (Class 0)
    'high_potassium': '#f2cda0', # 高钾 (Class 1)
    'node': '#c7d9f7'          # 决策节点
}

# --- 2. 使用Graphviz构建决策树 ---
# 创建一个有向图
dot = Digraph(comment='Decision Tree from Rules')
dot.attr('node', shape='box', style='rounded,filled', fontname='PingFang SC')
dot.attr('edge', fontname='PingFang SC')
dot.attr('graph', fontname='PingFang SC')

# 定义节点
# 决策规则:
# |--- ALR_氧化钡(BaO) <= 0.90
# |   |--- ALR_氧化铅(PbO) <= -1.45
# |   |   |--- class: 1 (高钾)
# |   |--- ALR_氧化铅(PbO) >  -1.45
# |   |   |--- class: 0 (铅钡)
# |--- ALR_氧化钡(BaO) >  0.90
# |   |--- ALR_氧化铅(PbO)_x_Weathering <= 0.22
# |   |   |--- class: 1 (高钾)
# |   |--- ALR_氧化铅(PbO)_x_Weathering >  0.22
# |   |   |--- class: 0 (铅钡)

dot.node('root', 'ALR_氧化钡(BaO) <= 0.90', fillcolor=colors['node'])
dot.node('L', 'ALR_氧化铅(PbO) <= -1.45', fillcolor=colors['node'])
dot.node('R', 'ALR_氧化铅(PbO)_x_Weathering <= 0.22', fillcolor=colors['node'])

dot.node('LL', '类别: 高钾', fillcolor=colors['high_potassium'])
dot.node('LR', '类别: 铅钡', fillcolor=colors['lead_barium'])
dot.node('RL', '类别: 高钾', fillcolor=colors['high_potassium'])
dot.node('RR', '类别: 铅钡', fillcolor=colors['lead_barium'])

# 定义边
dot.edge('root', 'L', label='是')
dot.edge('root', 'R', label='否')

dot.edge('L', 'LL', label='是')
dot.edge('L', 'LR', label='否')

dot.edge('R', 'RL', label='是')
dot.edge('R', 'RR', label='否')

# --- 3. 保存Graphviz图 ---
# Graphviz会生成一个.png文件和一个源文件
try:
    dot.render(os.path.join(output_base_dir, 'decision_tree_temp'), format='png', cleanup=True)
    # 重命名以匹配我们的路径变量
    os.rename(os.path.join(output_base_dir, 'decision_tree_temp.png'), output_plot_path)
    print(f"Graphviz图已初步生成。")
except Exception as e:
    print(f"Graphviz渲染失败: {e}")
    print("请确保已安装Graphviz (https://graphviz.org/download/) 并且其bin目录在系统的PATH中。")
    exit()


# --- 4. 添加图例并最终保存 ---
print("正在为图像添加图例...")
# 读取由Graphviz生成的图像
img = plt.imread(output_plot_path)

fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
ax.imshow(img)
ax.axis('off') # 不显示坐标轴

# 创建图例
legend_patches = [
    mpatches.Patch(color=colors['lead_barium'], label='类别 0: 铅钡'),
    mpatches.Patch(color=colors['high_potassium'], label='类别 1: 高钾'),
    mpatches.Patch(color=colors['node'], label='决策节点')
]
ax.legend(handles=legend_patches, loc='upper right', prop=cn_font, title="图例", frameon=True, fancybox=True, shadow=True)
plt.setp(ax.get_legend().get_title(), fontproperties=cn_font_legend_title) # 单独设置图例标题字体

plt.title("决策树可视化 (基于规则)", fontproperties=cn_font_title, pad=20)
plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n--- 可视化完成 ---")
print(f"带图例的最终决策树图已保存到: {output_plot_path}") 