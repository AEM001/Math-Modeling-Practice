import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import joblib # 新增导入

# 定义输出目录
output_base_dir = '2/logistic_regression/'
os.makedirs(output_base_dir, exist_ok=True)

# --- 1. 加载和准备数据 ---
console_output = [] # 用于收集控制台输出
report_content = [] # 用于收集markdown报告内容

def print_to_console_and_report(message, is_heading=False, is_code_block=False):
    print(message)
    if is_heading:
        report_content.append(f"\n## {message.strip(' =')}\n")
    elif is_code_block:
        report_content.append(f"\n```\n{message}\n```\n")
    else:
        report_content.append(f"{message}\n")

print_to_console_and_report("正在加载已处理的数据...")
try:
    # 加载上一步处理好的数据
    df = pd.read_csv('2/附件2_处理后.csv')
except FileNotFoundError:
    print_to_console_and_report("错误：未找到 '2/附件2_处理后.csv'。请先运行 data_processor.py 脚本。")
    exit()

print_to_console_and_report("数据加载完成。")

# 筛选出高钾和铅钡玻璃
df_filtered = df[df['类型'].isin(['高钾', '铅钡'])].copy()
print_to_console_and_report(f"筛选出 {len(df_filtered)} 条高钾和铅钡玻璃的数据。")

# --- 2. 特征工程 ---
# 创建因变量 y
df_filtered['target'] = (df_filtered['类型'] == '高钾').astype(int)

# 识别ALR特征列
alr_cols = [col for col in df.columns if col.startswith('ALR_')]

# 处理ALR特征中的NaN值（源于原始数据中的0），用0填充
alr_features_subset = df_filtered[alr_cols]
df_filtered[alr_cols] = alr_features_subset.fillna(0)
print_to_console_and_report("ALR特征中的NaN值已用0填充。")

# 创建风化特征
df_filtered['Weathering'] = (df_filtered['表面风化'] == '风化').astype(int)

# 创建交互项：化学成分 * 风化
interaction_features = pd.DataFrame()
for col in alr_cols:
    interaction_features[f'{col}_x_Weathering'] = df_filtered[col] * df_filtered['Weathering']
print_to_console_and_report("已创建化学成分与风化的交互项。")

# 合并所有特征
base_features = df_filtered[alr_cols + ['Weathering']]
features = pd.concat([base_features, interaction_features], axis=1)
X = features
y = df_filtered['target']

# --- 3. 数据划分 ---
print_to_console_and_report("="*20 + " 3. 数据划分 " + "="*20, is_heading=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print_to_console_and_report(f"数据已划分为训练集 ({len(X_train)}条) 和测试集 ({len(X_test)}条)。")

# --- 4. 数据标准化 ---
print_to_console_and_report("="*20 + " 4. 数据标准化 " + "="*20, is_heading=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print_to_console_and_report("特征数据已完成标准化 (fit on train, transform train/test)。")

# --- 5. 逻辑回归建模 (主模型) ---
print_to_console_and_report("="*20 + " 5. 逻辑回归建模 " + "="*20, is_heading=True)
# 使用L1正则化，有助于特征选择
log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
log_reg.fit(X_train_scaled, y_train)
print_to_console_and_report("主逻辑回归模型训练完成。")

# --- 6. 模型评估 (基于测试集) ---
print_to_console_and_report("="*20 + " 6. 模型评估 " + "="*20, is_heading=True)

# 6.1. 在训练集上的性能 (用于对比)
y_train_pred = log_reg.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print_to_console_and_report("--- 6.1. 在训练集上的性能 (用于判断过拟合) ---")
print_to_console_and_report(f"训练集准确率 (Accuracy): {train_accuracy:.4f}")

# 6.2. 在测试集上的性能 (关键指标)
y_test_pred = log_reg.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print_to_console_and_report("\n--- 6.2. 在测试集上的性能 (关键评估) ---")
print_to_console_and_report(f"测试集准确率 (Accuracy): {test_accuracy:.4f}")

# 分类报告
print_to_console_and_report("测试集分类报告 (Classification Report):")
class_report_str = classification_report(y_test, y_test_pred, target_names=['铅钡 (0)', '高钾 (1)'])
print_to_console_and_report(class_report_str, is_code_block=True)

# 混淆矩阵
print_to_console_and_report("测试集混淆矩阵 (Confusion Matrix):")
conf_matrix = confusion_matrix(y_test, y_test_pred)
print_to_console_and_report(str(conf_matrix), is_code_block=True)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测为铅钡', '预测为高钾'],
            yticklabels=['实际为铅钡', '实际为高钾'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
# 设置中文字体以避免乱码
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
output_path_cm = os.path.join(output_base_dir, 'confusion_matrix.png')
plt.savefig(output_path_cm)
print_to_console_and_report(f"\n混淆矩阵热力图已保存到: {os.path.basename(output_path_cm)}")
plt.close() # 关闭图形，避免在终端显示

# 6.3. 模型系数解读
print_to_console_and_report("--- 6.3. 模型系数 ---")
print_to_console_and_report(f"模型截距 (Intercept): {float(log_reg.intercept_):.4f}")

# 获取回归系数
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
})

# 按系数绝对值降序排列
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)

# 保存系数到CSV
output_path_coeffs = os.path.join(output_base_dir, 'logistic_regression_coefficients.csv')
coefficients.to_csv(output_path_coeffs, index=False, encoding='utf-8-sig')
print_to_console_and_report(f"\n回归系数已保存到: {os.path.basename(output_path_coeffs)}")
print_to_console_and_report("回归系数表 (绝对值排名前10):")
print_to_console_and_report(coefficients.head(10).to_markdown(index=False), is_code_block=True)

# --- 7. 可视化与PCA分析 ---
print_to_console_and_report("="*20 + " 7. 可视化与PCA分析 " + "="*20, is_heading=True)
pca = PCA(n_components=2)
# PCA应在训练集上拟合
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 打印PCA解释方差
print_to_console_and_report(f"\nPCA降维分析:")
print_to_console_and_report(f"前两个主成分解释的方差比例: {pca.explained_variance_ratio_}")
print_to_console_and_report(f"总解释方差: {sum(pca.explained_variance_ratio_):.4f}")

# 在2D PCA数据上训练一个新的逻辑回归模型用于可视化
log_reg_pca = LogisticRegression(random_state=42)
log_reg_pca.fit(X_train_pca, y_train)
pca_test_accuracy = log_reg_pca.score(X_test_pca, y_test)
print_to_console_and_report(f"基于PCA的2D简化模型在测试集上的准确率: {pca_test_accuracy:.4f}")

# 创建网格来绘制决策边界
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格中每个点的类别
Z = log_reg_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和数据点
plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# 绘制散点图
# 训练点
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=80, marker='o', label='训练数据')
# 测试点
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k', s=150, marker='*', label='测试数据')

plt.title('简化模型决策边界 (基于PCA降维数据)', fontsize=16)
plt.xlabel(f'主成分 1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'主成分 2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
plt.legend()

# 设置中文字体以避免乱码
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei'] # 尝试使用macOS自带的苹方字体，并提供备选
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

output_path_plot = os.path.join(output_base_dir, 'classification_boundary_pca.png')
plt.savefig(output_path_plot)
print_to_console_and_report(f"分类边界图已保存到: {os.path.basename(output_path_plot)}")
plt.close() # 关闭图形，避免在终端显示

# --- 8. 保存模型 ---
print_to_console_and_report("="*20 + " 8. 保存模型 " + "="*20, is_heading=True)
model_path = os.path.join(output_base_dir, 'logistic_regression_model.joblib')
scaler_path = os.path.join(output_base_dir, 'scaler.joblib')
pca_path = os.path.join(output_base_dir, 'pca.joblib')

joblib.dump(log_reg, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(pca, pca_path)

print_to_console_and_report(f"主模型已保存到: {os.path.basename(model_path)}")
print_to_console_and_report(f"数据缩放器已保存到: {os.path.basename(scaler_path)}")
print_to_console_and_report(f"PCA转换器已保存到: {os.path.basename(pca_path)}")

# --- 保存最终报告 ---
markdown_report_path = os.path.join(output_base_dir, 'model_analysis_report.md')
with open(markdown_report_path, 'w', encoding='utf-8') as f:
    f.write("# 逻辑回归模型分析报告\n\n")
    for line in report_content:
        f.write(line)
    # 在报告末尾添加图片链接
    f.write(f"\n## 附加图表\n")
    f.write(f"\n### 混淆矩阵热力图\n")
    f.write(f"![Confusion Matrix]({os.path.basename(output_path_cm)})\n\n")
    f.write(f"\n### 分类边界图\n")
    f.write(f"![Classification Boundary]({os.path.basename(output_path_plot)})\n\n")

print("\n" + "="*20 + " 报告生成完成 " + "="*20 + "\n")
print(f"模型分析报告已保存到: {markdown_report_path}") 