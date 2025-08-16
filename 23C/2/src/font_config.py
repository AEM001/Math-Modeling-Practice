# -*- coding: utf-8 -*-
"""
中文字体配置模块
解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_chinese_font():
    """
    设置中文字体，解决matplotlib中文显示问题
    """
    system = platform.system()
    
    # macOS系统的中文字体选择
    if system == 'Darwin':  # macOS
        chinese_fonts = [
            'PingFang SC',
            'PingFang HK', 
            'Heiti TC',
            'STHeiti',
            'Kaiti SC',
            'Arial Unicode MS',
            'Songti SC',
            'SimSong'
        ]
    # Windows系统的中文字体选择
    elif system == 'Windows':
        chinese_fonts = [
            'Microsoft YaHei',
            'SimHei',
            'SimSun',
            'KaiTi',
            'FangSong'
        ]
    # Linux系统的中文字体选择
    else:
        chinese_fonts = [
            'DejaVu Sans',
            'Noto Sans CJK SC',
            'WenQuanYi Micro Hei',
            'AR PL UMing CN'
        ]
    
    # 查找系统中可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 如果没有找到专门的中文字体，使用系统默认
    if selected_font is None:
        # 尝试一些通用的支持中文的字体
        fallback_fonts = ['Arial Unicode MS', 'DejaVu Sans']
        for font in fallback_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font is None:
            selected_font = 'sans-serif'
    
    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = [selected_font, 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    print(f"已设置中文字体: {selected_font}")
    return selected_font

def get_chinese_font():
    """
    获取当前设置的中文字体
    """
    return plt.rcParams['font.sans-serif'][0]

# 在模块导入时自动设置中文字体
setup_chinese_font()
