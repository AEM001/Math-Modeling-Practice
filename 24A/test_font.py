import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 获取系统所有字体
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = []

# 查找可能的中文字体
for font in fonts:
    if any(keyword in font.lower() for keyword in ['chinese', 'china', 'han', 'gb', 'song', 'hei', 'kai', 'unicode', 'arial', 'hiragino']):
        chinese_fonts.append(font)

print("Mac系统中可用的中文相关字体:")
for font in sorted(set(chinese_fonts)):
    print(f"- {font}")

# 测试字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title('中文字体测试 - 这是标题')
ax.set_xlabel('横轴标签 (x轴)')
ax.set_ylabel('纵轴标签 (y轴)')
plt.show() 