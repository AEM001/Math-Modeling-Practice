import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 1. 加载和准备数据
file_path = '附件2_处理后_CLR.csv'
data = pd.read_csv(file_path)

# 提取化学成分数据用于聚类
chemical_columns = data.columns[1:15]
X_cluster = data[chemical_columns]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 2. K-Means 聚类
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

# 3. 分析并排序簇的风化程度
cluster_weathering_ratio = {}
for i in range(k):
    cluster_data = data[data['cluster'] == i]
    weathering_count = (cluster_data['表面风化'] == '风化').sum()
    total_count = len(cluster_data)
    ratio = weathering_count / total_count if total_count > 0 else 0
    cluster_weathering_ratio[i] = ratio

sorted_clusters = sorted(cluster_weathering_ratio.items(), key=lambda item: item[1])
least_weathered_cluster_id = sorted_clusters[0][0]
most_weathered_cluster_id = sorted_clusters[-1][0]

least_weathered_group = data[data['cluster'] == least_weathered_cluster_id].copy()
most_weathered_group = data[data['cluster'] == most_weathered_cluster_id].copy()

# 4. 样本一一配对
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(least_weathered_group[chemical_columns])
distances, indices = nbrs.kneighbors(most_weathered_group[chemical_columns])

paired_data = []
weathered_paired_samples = []
for i in range(len(most_weathered_group)):
    weathered_sample = most_weathered_group.iloc[i]
    unweathered_sample = least_weathered_group.iloc[indices[i][0]]
    paired_data.append((weathered_sample, unweathered_sample))
    weathered_paired_samples.append(weathered_sample)

weathered_paired_df = pd.DataFrame(weathered_paired_samples)

# 5. 回归建模 (跳过0值成分)
regression_models = {}
# 识别在所有配对风化样本中都为0的列
zero_value_columns = [col for col in chemical_columns if weathered_paired_df[col].sum() == 0]
print(f"\n检测到始终为0的成分，将跳过回归: {zero_value_columns}")

non_zero_columns = [col for col in chemical_columns if col not in zero_value_columns]

for col in non_zero_columns:
    X_reg = np.array([pair[0][col] for pair in paired_data]).reshape(-1, 1)
    y_reg = np.array([pair[1][col] for pair in paired_data]).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    regression_models[col] = reg

# 6. 预测所有风化样本的化学成分
weathered_samples = data[data['表面风化'] == '风化'].copy()
predicted_clr_df = weathered_samples[['文物采样点']].copy()

for col in chemical_columns:
    if col in zero_value_columns:
        predicted_clr_df[col] = 0
    else:
        model = regression_models[col]
        current_values = weathered_samples[col].values.reshape(-1, 1)
        predicted_values = model.predict(current_values)
        predicted_clr_df[col] = predicted_values.flatten()

# 7. CLR逆变换函数
def inverse_clr(clr_data):
    exp_data = np.exp(clr_data)
    sum_exp_data = np.sum(exp_data, axis=1)
    original_data = exp_data.div(sum_exp_data, axis=0) * 100
    return original_data

# 提取预测的CLR值进行逆变换
predicted_clr_values = predicted_clr_df[chemical_columns]

# 执行逆变换
real_values_df = inverse_clr(predicted_clr_values)
real_values_df.columns = [col + '_预测_真实值' for col in chemical_columns]

# 8. 准备并保存最终结果
final_result_df = pd.concat([
    weathered_samples[['文物采样点']].reset_index(drop=True),
    real_values_df.reset_index(drop=True)
], axis=1)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n--- 预测的风化前成分真实值 ---")
print(final_result_df.head())

# 保存到CSV
output_filename = '最终预测风化前成分_真实值.csv'
final_result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n最终预测结果已保存到 '{output_filename}'")