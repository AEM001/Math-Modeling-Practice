#!/usr/bin/env python3
"""
Test script to check Chinese font display in matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

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
            print(f"Found available font: {available_font}")
            break
    except:
        continue

if available_font:
    plt.rcParams['font.sans-serif'] = [available_font]
else:
    # Fallback to system default
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
    print("Using fallback fonts")

plt.rcParams['axes.unicode_minus'] = False

# Test plot
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data
categories = ['花叶类', '花菜类', '茄类', '辣椒类', '水生根茎类', '食用菌']
values = [0.866, 0.667, 0.812, 0.787, 0.837, 0.848]

bars = ax.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

ax.set_title('各品类模型R²性能测试', fontsize=16, fontweight='bold')
ax.set_ylabel('R²得分', fontsize=14)
ax.set_xlabel('蔬菜品类', fontsize=14)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{value:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12)

ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/Mac/Downloads/Math-Modeling-Practice/23C/2/output/font_test.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Font test completed! Check output/font_test.png")
print(f"Current font setting: {plt.rcParams['font.sans-serif']}")

# List all available Chinese fonts
print("\nAvailable Chinese fonts on this system:")
chinese_font_names = []
for font in fm.fontManager.ttflist:
    if any(chinese in font.name.lower() for chinese in ['simhei', 'simsun', 'pingfang', 'hiragino', 'arial unicode', 'yahei', 'heiti']):
        chinese_font_names.append(font.name)

chinese_font_names = list(set(chinese_font_names))
for font_name in sorted(chinese_font_names):
    print(f"  - {font_name}")