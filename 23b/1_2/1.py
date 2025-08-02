import pandas as pd
import numpy as np
# 初始化参数
D_0 = 70
theta = 120
alpha = 1.5
d = 200
d = d*np.sin(np.radians(90-theta/2))/np.sin(np.radians(90-alpha+theta/2))

distances = np.array([-800, -600, -400, -200, 0, 200, 400, 600, 800])
D= D_0 - distances * np.tan(np.radians(alpha))

print(D)

W=D*np.sin(np.radians(theta/2))*( 1/np.sin(np.radians((180-theta)/2+alpha)) +1/np.sin(np.radians((180-theta)/2-alpha)))
print(W)
n=1-d/W
# 将重叠率转换为百分比，并将第一个设为空（因为没有前一条测线）
n_percent = n * 100
n_percent[0] = np.nan  # 第一个测线没有重叠率
print(n_percent)
# 创建按照要求格式的 DataFrame
# 第一行：距离值作为列名
columns = [''] + [str(int(d)) for d in distances]

# 创建数据行
data = []
data.append(['测线距中心点处的距离/m'] + distances.tolist())
data.append(['海水深度/m'] + D.tolist())
data.append(['覆盖宽度/m'] + W.tolist())
data.append(['与前一条测线的重叠率/%'] + n_percent.tolist())

# 创建 DataFrame，不使用列名（因为第一行就是标题）
df = pd.DataFrame(data)

## 将 DataFrame 保存为 Excel 文件
path = '/Users/Mac/Downloads/23b/1_2/result1.xlsx'
df.to_excel(path, index=False, header=False)
print(f"结果已保存到: {path}")