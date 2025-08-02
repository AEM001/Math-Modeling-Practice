import pandas as pd
import numpy as np

# --- 1. 加载数据 ---
# 加载文物信息表
try:
    df1 = pd.read_csv('2/附件.csv', encoding='utf-8')
except UnicodeDecodeError:
    df1 = pd.read_csv('2/附件.csv', encoding='gbk')

# 加载文物化学成分表
try:
    df2 = pd.read_csv('2/附件2.csv', encoding='utf-8')
except UnicodeDecodeError:
    df2 = pd.read_csv('2/附件2.csv', encoding='gbk')

print("数据加载完成。")

# --- 2. 处理附件2数据 ---
df2_processed = df2.copy()

# 将"附件2.csv"中空缺的值用0填充
df2_processed.fillna(0, inplace=True)
print("缺失值已用0填充。")

# 提取文物编号
# 从'文物采样点'列中提取前面的数字作为文物编号
df2_processed['文物编号'] = df2_processed['文物采样点'].astype(str).str.extract(r'(\d+)').iloc[:, 0]
df2_processed['文物编号'] = df2_processed['文物编号'].str.zfill(2)

# 识别化学成分列
chem_cols = [col for col in df2.columns if col not in ['文物采样点', '类型', '表面风化']]

# 转换化学成分列为数值类型
for col in chem_cols:
    df2_processed[col] = df2_processed[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)

# 按文物编号分组并取均值
numeric_cols = df2_processed.select_dtypes(include=np.number).columns.tolist()
df2_agg = df2_processed.groupby('文物编号')[numeric_cols].mean().reset_index()
print("已按文物编号合并数据并计算平均值。")

# --- 3. 合并信息 ---
# 准备附件1中的信息用于合并
df1['文物编号'] = df1['文物编号'].astype(str).str.zfill(2)
df1_info = df1[['文物编号', '纹饰', '类型', '颜色', '表面风化']]

# 在合并前，从 df2_agg 中删除 '类型' 和 '表面风化' 列，以避免因列名冲突导致合并失败
# 这些列将在下一步从 df1_info 中正确添加
df2_agg.drop(columns=['类型', '表面风化'], inplace=True, errors='ignore')

# 合并处理后的数据和文物信息
# 使用left join确保处理后的数据行都被保留
final_df = pd.merge(df2_agg, df1_info, on='文物编号', how='left')

# 重新排列列的顺序，将基本信息列放在前面
info_cols = ['文物编号', '纹饰', '类型', '颜色', '表面风化']
final_df = final_df[info_cols + [col for col in final_df.columns if col not in info_cols]]
print("已补全类型和表面风化信息。")

# --- 4. ALR变换 ---
reference_col = '二氧化硅(SiO2)'
# 使用 final_df 的列来获取最新的化学成分列列表，以防万一
chem_cols_for_alr = [col for col in chem_cols if col in final_df.columns]

print("开始执行ALR变换(新逻辑：忽略0值)...")

alr_cols = {}
# 执行ALR变换
# np.log(a/b) 当 a 或 b 为 NaN 时，结果为 NaN，这正好满足我们的要求
for col in chem_cols_for_alr:
    if col != reference_col:
        alr_col_name = f'ALR_{col}'
        # Replace 0 with NaN for the current column and the reference column before division
        numerator = final_df[col].replace(0, np.nan)
        denominator = final_df[reference_col].replace(0, np.nan)
        alr_cols[alr_col_name] = np.log(numerator / denominator)

alr_df = pd.DataFrame(alr_cols, index=final_df.index)

# 将ALR结果合并到最终的DataFrame
final_df_with_alr = pd.concat([final_df, alr_df], axis=1)
print("ALR变换完成。")

# --- 5. 保存结果 ---
output_path_pre_alr = '2/附件2_处理前ALR.csv'
final_df.to_csv(output_path_pre_alr, index=False, encoding='utf-8-sig')
print(f"处理完成，ALR变换前的数据已保存到 {output_path_pre_alr}")

output_path_post_alr = '2/附件2_处理后ALR.csv'
final_df_with_alr.to_csv(output_path_post_alr, index=False, encoding='utf-8-sig')

print(f"处理完成，ALR变换后的数据已保存到 {output_path_post_alr}")
print("\n处理后的数据预览：")
print(final_df_with_alr.head())
