# -*- coding: utf-8 -*-
"""
中文字体配置模块
解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import platform

def setup_chinese_font():
    """
    设置中文字体，解决matplotlib中文显示问题
    """
    system = platform.system()
    
    # macOS系统的中文字体选择
    if system == 'Darwin':  # macOS
        chinese_fonts = [
            'PingFang HK',
            'PingFang SC',
            'STHeiti',
            'Heiti TC',
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
            'Noto Sans CJK SC',
            'WenQuanYi Micro Hei',
            'AR PL UMing CN',
            'DejaVu Sans'
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
    
    # 强制设置所有字体相关参数
    mpl.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial', 'sans-serif']
    mpl.rcParams['font.family'] = [selected_font]
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置所有文本相关的字体
    mpl.rcParams['font.serif'] = [selected_font, 'serif']
    mpl.rcParams['font.monospace'] = [selected_font, 'monospace']
    
    # 清除并重新加载字体管理器
    try:
        fm._rebuild()
        fm._load_fontmanager(try_read_cache=False)
    except:
        try:
            import matplotlib.pyplot as plt
            plt.rcdefaults()
            mpl.rcParams['font.sans-serif'] = [selected_font]
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['axes.unicode_minus'] = False
        except:
            pass
    
    print(f"已设置中文字体: {selected_font}")
    return selected_font

def get_chinese_font():
    """
    获取当前设置的中文字体
    """
    return plt.rcParams['font.sans-serif'][0]

# 在模块导入时自动设置中文字体
setup_chinese_font()
