import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- 0. 环境设置 ---
# 定义输出目录
output_base_dir = '2/decision_tree/'
os.makedirs(output_base_dir, exist_ok=True)

report_content = []

def print_to_console_and_report(message, is_heading=False, is_code_block=False):
    """辅助函数，同时打印到控制台并收集内容到报告列表"""
    print(message)
    if is_heading:
        report_content.append(f"\n## {message.strip(' =')}\n")
    elif is_code_block:
        report_content.append(f"\n```\n{message}\n```\n")
    else:
        report_content.append(f"{message}\n")

# --- 1. 数据加载与准备 ---
print_to_console_and_report("="*20 + " 1. 数据加载与准备 " + "="*20, is_heading=True)
try:
    df = pd.read_csv('2/decision_tree/附件2_处理后.csv')
    print_to_console_and_report("数据 '2/decision_tree/附件2_处理后.csv' 加载完成。")
except FileNotFoundError:
    print_to_console_and_report("错误：未找到 '2/decision_tree/附件2_处理后.csv'。请先运行初始的数据处理脚本。")
    exit()

# --- 2. 特征工程 ---
print_to_console_and_report("="*20 + " 2. 特征工程 " + "="*20, is_heading=True)
df_filtered = df[df['类型'].isin(['高钾', '铅钡'])].copy()
df_filtered['target'] = (df_filtered['类型'] == '高钾').astype(int)
alr_cols = [col for col in df.columns if col.startswith('ALR_')]
df_filtered.loc[:, alr_cols] = df_filtered[alr_cols].fillna(0)
df_filtered['Weathering'] = (df_filtered['表面风化'] == '风化').astype(int)
interaction_features = pd.DataFrame()
for col in alr_cols:
    interaction_features[f'{col}_x_Weathering'] = df_filtered[col] * df_filtered['Weathering']

base_features = df_filtered[alr_cols + ['Weathering']]
features = pd.concat([base_features, interaction_features], axis=1)
X = features
y = df_filtered['target']
print_to_console_and_report("特征工程完成。")


# --- 3. 数据划分与标准化 ---
print_to_console_and_report("="*20 + " 3. 数据划分与标准化 " + "="*20, is_heading=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print_to_console_and_report(f"数据已划分为训练集({len(X_train)}条)和测试集({len(X_test)}条)并完成标准化。")


# --- 4. 决策树建模 ---
print_to_console_and_report("="*20 + " 4. 决策树建模 " + "="*20, is_heading=True)
# 设置max_depth以防止过拟合，并保证树的可解释性
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)
print_to_console_and_report("决策树模型训练完成 (max_depth=5)。")


# --- 5. 模型评估 (基于测试集) ---
print_to_console_and_report("="*20 + " 5. 模型评估 " + "="*20, is_heading=True)
y_test_pred = dt_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print_to_console_and_report(f"测试集准确率 (Accuracy): {test_accuracy:.4f}")

print_to_console_and_report("\n测试集分类报告:")
class_report_str = classification_report(y_test, y_test_pred, target_names=['铅钡 (0)', '高钾 (1)'])
print_to_console_and_report(class_report_str, is_code_block=True)

print_to_console_and_report("\n测试集混淆矩阵:")
conf_matrix = confusion_matrix(y_test, y_test_pred)
print_to_console_and_report(str(conf_matrix), is_code_block=True)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=['预测为铅钡', '预测为高钾'], yticklabels=['实际为铅钡', '实际为高钾'])
plt.title('Decision Tree Confusion Matrix')
plt.ylabel('Actual Class'); plt.xlabel('Predicted Class')
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
output_path_cm = os.path.join(output_base_dir, 'decision_tree_confusion_matrix.png')
plt.savefig(output_path_cm)
plt.close()
print_to_console_and_report(f"\n混淆矩阵热力图已保存到: {os.path.basename(output_path_cm)}")


# --- 6. 决策规则与特征重要性 ---
print_to_console_and_report("="*20 + " 6. 决策规则与特征重要性 " + "="*20, is_heading=True)

# 提取文本格式的决策规则
print_to_console_and_report("\n决策规则 (文本格式):")
rules = export_text(dt_model, feature_names=list(X.columns))
print_to_console_and_report(rules, is_code_block=True)

# 提取并展示特征重要性
print_to_console_and_report("\n特征重要性:")
importances = dt_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
# 只显示重要性>0的特征
feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
print_to_console_and_report(feature_importance_df.to_markdown(index=False), is_code_block=True)


# --- 7. 决策树可视化 ---
print_to_console_and_report("="*20 + " 7. 决策树可视化 " + "="*20, is_heading=True)
plt.figure(figsize=(40, 20))

# 设置中文字体以避免乱码 (再次确保应用到此图)
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

plot_tree(dt_model,
          feature_names=X.columns,
          class_names=['铅钡', '高钾'],
          filled=True,
          rounded=True,
          fontsize=8) # 减小字体大小，使文本不那么拥挤
output_path_tree = os.path.join(output_base_dir, 'decision_tree_visualization.png')
plt.savefig(output_path_tree, dpi=300)
plt.close()
print_to_console_and_report(f"决策树可视化图已保存到: {os.path.basename(output_path_tree)}")


# --- 8. 保存模型 ---
print_to_console_and_report("="*20 + " 8. 保存模型 " + "="*20, is_heading=True)
model_path = os.path.join(output_base_dir, 'decision_tree_model.joblib')
scaler_path = os.path.join(output_base_dir, 'scaler.joblib')
joblib.dump(dt_model, model_path)
joblib.dump(scaler, scaler_path)
print_to_console_and_report(f"决策树模型已保存到: {os.path.basename(model_path)}")
print_to_console_and_report(f"数据缩放器已保存到: {os.path.basename(scaler_path)}")


# --- 9. 生成最终报告 ---
markdown_report_path = os.path.join(output_base_dir, 'decision_tree_analysis_report.md')
with open(markdown_report_path, 'w', encoding='utf-8') as f:
    f.write("# 决策树模型分析报告\n\n")
    for line in report_content:
        f.write(line)
    # 在报告末尾添加图片链接
    f.write(f"\n## 附加图表\n")
    f.write(f"\n### 混淆矩阵热力图\n")
    f.write(f"![Confusion Matrix]({os.path.basename(output_path_cm)})\n\n")
    f.write(f"\n### 决策树可视化图\n")
    f.write(f"![Decision Tree]({os.path.basename(output_path_tree)})\n\n")

print("\n" + "="*20 + " 分析报告生成完成 " + "="*20)
print(f"决策树模型分析报告已保存到: {markdown_report_path}") 