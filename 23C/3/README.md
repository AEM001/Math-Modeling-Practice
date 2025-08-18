# 单品补货与定价策略优化

基于历史数据与短期序列，预测2023-07-01的单品销量和批发价，并建立优化模型求解最优补货量与定价策略。

## 项目结构

```
├── data/                           # 数据文件
│   ├── 过滤后单品级汇总表.csv        # 历史销售数据
│   ├── 可售品种时间序列表_20230624-30.csv  # 短期时间序列
│   └── 品种周统计表_20230624-30.csv  # 周统计数据
├── src/                            # 源代码
│   ├── config.py                   # 配置参数
│   ├── io_utils.py                 # 数据加载清洗
│   ├── features.py                 # 特征工程
│   ├── forecast.py                 # 销量价格预测
│   ├── screen.py                   # 候选产品筛选
│   ├── optimize.py                 # 优化建模求解
│   ├── pricing.py                  # 结果整合
│   ├── visualize.py                # 可视化分析
│   └── main.py                     # 主程序入口
├── outputs/                        # 输出目录
│   ├── results/                    # 结果文件
│   │   └── plan_2023-07-01.csv     # 最终补货计划
│   └── figs/                       # 图表文件
├── requirements.txt                # Python依赖
├── README.md                       # 使用说明
└── plan.md                         # 详细实现计划
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本运行
```bash
python src/main.py
```

### 完整参数示例
```bash
python src/main.py \
    --date 2023-07-01 \
    --data-dir ./data \
    --output-dir ./outputs \
    --solver CBC \
    --max-candidates 40 \
    --min-shelf 27 \
    --max-shelf 33
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--date` | 目标优化日期 (YYYY-MM-DD) | 2023-07-01 |
| `--data-dir` | 数据目录路径 | data |
| `--output-dir` | 输出目录路径 | outputs |
| `--use-elasticity` | 使用价格弹性模型 | False |
| `--solver` | 优化求解器 (CBC/GLPK) | CBC |
| `--quick-mode` | 快速基线预测模式 | False |
| `--no-viz` | 跳过可视化生成 | False |
| `--max-candidates` | 最大候选产品数 | 40 |
| `--min-shelf` | 最小上架数量 | 27 |
| `--max-shelf` | 最大上架数量 | 33 |

## 输出文件

### 主要结果文件
- `plan_2023-07-01.csv` - 最终补货与定价计划
- `detailed_results_20230701.csv` - 详细结果数据
- `summary_report_20230701.txt` - 摘要报告

### 中间结果文件
- `forecast_results.csv` - 预测结果
- `candidates.csv` - 候选产品
- `screening_report.txt` - 筛选报告
- `category_analysis_20230701.csv` - 品类分析

### 可视化图表
- `sales_distribution.png` - 销量分布分析
- `price_analysis.png` - 价格策略分析
- `profit_analysis.png` - 利润分析
- `category_overview.png` - 品类概览
- `optimization_summary.png` - 优化结果摘要

## 算法流程

### 1. 数据准备
- 加载历史销售数据和时间序列数据
- 数据清洗：去重、异常值处理、缺失值填补
- 获取可售产品列表

### 2. 特征工程
- 时间特征：周几、是否周末、月份等
- 移动平均特征：7/14/28天移动均值、标准差
- 滞后特征：1/2/3/7天滞后值
- 品类特征：品类统计信息
- 价格弹性特征：价格变化率等

### 3. 销量与价格预测
- **销量预测**：随机森林回归模型，结合周六效应修正
- **批发价预测**：移动平均法或回归模型
- **快速模式**：基于移动平均和周六修正的基线方法

### 4. 候选产品筛选
- 销量阈值筛选：预测销量 ≥ 2.5kg
- 模型质量筛选：R²评分阈值
- 综合评分：模型质量 + 稳定性 + 利润潜力
- 品类多样性平衡
- 选择前N个候选产品

### 5. 优化建模
#### 决策变量
- $x_i ∈ \\{0,1\\}$：是否上架
- $P_i ≥ 0$：进货量(kg)
- $A_i ∈ [A_{min}, A_{max}]$：加成率

#### 目标函数
最大化利润：$\\max \\sum_i x_i \\cdot (Q_i \\cdot C_i \\cdot (1 + A_i) - P_i \\cdot C_i)$

#### 约束条件
- 上架数量：$27 ≤ \\sum_i x_i ≤ 33$
- 最小陈列量：$P_i ≥ 2.5 \\cdot x_i$
- 加成率范围：$A_{min} ≤ A_i ≤ A_{max}$
- 库存合理性：$P_i ≤ 2Q_i \\cdot x_i$

### 6. 结果整合与导出
- 计算定价指标：毛利率、库存周转率等
- 生成最终补货计划
- 导出多格式结果文件
- 品类分析汇总

### 7. 可视化分析
- 销量分布对比分析
- 价格策略可视化
- 利润贡献分析
- 品类组合概览
- 优化结果摘要

## 核心配置

可在 `src/config.py` 中修改关键参数：

```python
# 约束参数
MIN_SHELF_COUNT = 27        # 最小上架数量
MAX_SHELF_COUNT = 33        # 最大上架数量
MIN_DISPLAY_QTY = 2.5       # 最小陈列量

# 加成率范围
MARKUP_BOUNDS = {
    'min': 0.1,             # 最小加成率 10%
    'max': 0.6              # 最大加成率 60%
}

# 随机森林参数
RF_PARAMS = {
    'n_estimators': 50,
    'max_depth': 8,
    'min_samples_split': 10,
    'min_samples_leaf': 5
}
```

## 使用建议

### 快速测试
```bash
# 使用快速模式进行测试
python src/main.py --quick-mode --no-viz
```

### 完整分析
```bash
# 生成完整的分析报告和图表
python src/main.py --use-elasticity
```

### 参数调优
```bash
# 调整候选数量和上架范围
python src/main.py --max-candidates 50 --min-shelf 25 --max-shelf 35
```

## 注意事项

1. **数据要求**：确保数据文件格式正确，日期列为标准格式
2. **内存使用**：大数据集时建议使用`--quick-mode`
3. **求解器选择**：CBC为开源求解器，GLPK速度较慢但更稳定
4. **可视化**：生成图表需要中文字体支持

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查数据文件路径和格式
   - 确认CSV文件编码为UTF-8

2. **优化求解失败**
   - 降低候选产品数量
   - 放宽约束条件
   - 尝试不同的求解器

3. **可视化错误**
   - 安装中文字体
   - 使用`--no-viz`跳过可视化

### 获取帮助
```bash
python src/main.py --help
```

## 许可证

本项目仅供学术研究使用。