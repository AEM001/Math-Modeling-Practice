import pandas as pd
import numpy as np
def get_width(B):
    # 初始化参数
    D_0 = 120  # 海底深度（单位：m）
    alpha = 1.5  # 坡度（单位：度）
    D = D_0 - distances_km * np.tan(np.radians(alpha)) * np.cos(np.radians(180 - B))
    theta = 120  # 换能器的开角（单位：度）
    alpha= np.arctan(abs(np.sin(np.radians(B))) * np.tan(np.radians(alpha))) * 180 / np.pi
    print(D)
    W = D * np.sin(np.radians(theta / 2)) * (
            1 / np.sin(np.radians((180 - theta) / 2 + alpha)) + 1 / np.sin(np.radians((180 - theta) / 2 - alpha)))
    print(W)
    return W
distances = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1])
distances_km = distances * 1852  # 转换为米用于计算
print(distances_km)

angle=[0,45,90,135,180,225,270,315]
W=[]
for i in angle:
    W.append(get_width(i))

# 创建按照要求格式的 DataFrame
# 第一行：距离值作为标题
data = []
data.append(['测量船距海域中心点处的距离/海里'] + distances.tolist())

# 添加每个测线方向角对应的覆盖宽度数据
for i, ang in enumerate(angle):
    data.append([f'测线方向夹角/{ang}°'] + W[i].tolist())

# 创建 DataFrame，不使用列名和索引
df = pd.DataFrame(data)

# 将 DataFrame 保存为 Excel 文件
path = '/Users/Mac/Downloads/23b/1_2/result2.xlsx'
df.to_excel(path, index=False, header=False)
print(f"结果已保存到: {path}")