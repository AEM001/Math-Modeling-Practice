
import pandas as pd
import numpy as np

# Load the data
try:
    df_item = pd.read_csv('单品级每日汇总表.csv')
except FileNotFoundError:
    print("错误：请确保CSV文件与脚本位于同一目录中。")
    exit()

# --- 数据整合与清洗 ---

# 过滤掉打折销售的记录 (打折销量 > 0)
df_normal_sales = df_item[df_item['打折销量(千克)'] == 0].copy()

# 处理缺失或无效值
key_columns = ['正常销量(千克)', '正常销售单价(元/千克)', '批发价格(元/千克)']
df_normal_sales.dropna(subset=key_columns, inplace=True)

# 过滤掉价格或销量为零或负数的情况，这对于对数模型是无效的
df_normal_sales = df_normal_sales[df_normal_sales['正常销量(千克)'] > 0]
df_normal_sales = df_normal_sales[df_normal_sales['正常销售单价(元/千克)'] > 0]
df_normal_sales = df_normal_sales[df_normal_sales['批发价格(元/千克)'] > 0]

# --- 特征工程 ---

# 将'销售日期'转换为datetime对象
df_normal_sales['销售日期'] = pd.to_datetime(df_normal_sales['销售日期'])

# --- 数据集划分 ---

# 按日期对数据框进行排序
df_normal_sales.sort_values(by='销售日期', inplace=True)

# 将数据分割为训练集和测试集
train_size = int(0.7 * len(df_normal_sales))
train_df = df_normal_sales.iloc[:train_size]
test_df = df_normal_sales.iloc[train_size:]

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

# 将准备好的数据保存到新的CSV文件，以备下一阶段使用
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("\n数据准备完成。已创建 `train_data.csv` 和 `test_data.csv`。")
