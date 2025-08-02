import pandas as pd
import numpy as np

# --- 1. 加载数据 ---
input_path = '3/附件3.csv'
try:
    df = pd.read_csv(input_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding='gbk')

print(f"数据已从 {input_path} 加载。")

# --- 2. 准备数据 ---
# 识别信息列和化学成分列
info_cols = ['文物编号', '表面风化']
chem_cols = [col for col in df.columns if col not in info_cols]

# 将化学成分列强制转换为数值类型，无法转换的将变为NaN
# 这会处理原始数据中的空白或非数值条目
df[chem_cols] = df[chem_cols].apply(pd.to_numeric, errors='coerce')

print("化学成分数据已准备好，空值已转换为NaN。")

# --- 3. ALR变换 ---
reference_col = '二氧化硅(SiO2)'
print(f"开始执行ALR变换，参照列为: {reference_col}")

# 检查参照列是否存在
if reference_col not in df.columns:
    print(f"错误：参照列 '{reference_col}' 不存在于数据中。")
    exit()

alr_data = {}
# 遍历除参照列外的所有化学成分列
for col in chem_cols:
    if col != reference_col:
        # 计算ALR变换: log(x_i / x_ref)
        # 如果任一值为NaN，结果自动为NaN
        alr_col_name = f"ALR_{col}"
        alr_data[alr_col_name] = np.log(df[col] / df[reference_col])

# 创建包含ALR变换结果的DataFrame
alr_df = pd.DataFrame(alr_data)
print("ALR变换完成，结果中包含NaN。")

# --- 4. 将NaN替换为0 ---
alr_df.fillna(0, inplace=True)
print("已将所有NaN值替换为0。")

# --- 5. 合并与保存 ---
# 合并原始信息列和处理后的ALR数据
final_df = pd.concat([df[info_cols], alr_df], axis=1)

# 保存结果
output_path = '3/附件3_处理后_ALR.csv'
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"处理完成，结果已保存到 {output_path}")
print("\n处理后的数据预览：")
print(final_df.head()) 