# 蔬菜定价与补货决策系统

基于2023年高教社杯全国大学生数学建模竞赛C题问题2的完整解决方案。

## 项目概述

本项目实现了完整的蔬菜类商品自动定价与补货决策系统，通过机器学习预测需求，并基于启发式算法制定最优的定价和补货策略，以实现商超收益最大化。

## 核心功能

- **需求预测**: 基于随机森林、梯度提升等机器学习算法预测未来7天各品类需求
- **成本估计**: 基于历史数据的移动平均法估计未来成本
- **定价策略**: 成本加成定价法结合价格弹性优化，约束条件下的收益最大化
- **补货策略**: 基于服务水平的安全库存策略，考虑损耗率和需求不确定性
- **回测验证**: 滚动交叉验证确保模型稳定性和可靠性

## 项目结构

```
23C/2/
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖包
├── README.md                  # 项目说明
├── config/
│   └── config.py              # 配置文件
├── src/
│   ├── data_processing/       # 数据处理模块
│   │   └── data_cleaner.py    # 数据清洗
│   ├── feature_engineering/   # 特征工程模块
│   │   └── feature_builder.py # 特征构建
│   ├── modeling/              # 建模模块
│   │   ├── demand_models.py   # 需求建模
│   │   └── backtest.py        # 回测验证
│   ├── forecasting/           # 预测模块
│   │   └── demand_forecast.py # 需求预测
│   ├── strategy/              # 策略模块
│   │   └── pricing_replenishment.py # 定价补货策略
│   ├── visualization/         # 可视化模块
│   │   └── report_generator.py # 报告生成
│   └── utils/                 # 工具模块
│       └── logger.py          # 日志工具
├── output/                    # 输出结果
├── logs/                      # 运行日志
├── 单品级每日汇总表.csv        # 数据文件
└── 品类级每日汇总表.csv        # 数据文件
```

## 技术架构

### 七阶段处理流程

1. **数据审计与清洗**: 排除异常数据，标记促销场景
2. **特征工程**: 构建时间特征、滞后特征、滚动特征等
3. **需求建模**: 多模型对比选择最优预测模型
4. **回测验证**: 滚动交叉验证评估模型稳定性
5. **需求预测**: 预测未来7天各品类需求和成本
6. **策略制定**: 启发式定价与安全库存补货
7. **结果输出**: 生成可视化报告和决策文件

### 关键算法

- **机器学习模型**: Random Forest, Gradient Boosting, Huber Regression
- **特征工程**: 滞后特征、滚动窗口统计、品类聚合特征
- **定价优化**: 成本加成+价格弹性调整的网格搜索
- **库存管理**: 基于服务水平的安全库存计算

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 requirements.txt

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

```bash
python main.py
```

### 输出文件

运行完成后，`output/` 目录将包含：

- `clean_items.csv`: 清洗后的数据
- `train_features.csv`, `test_features.csv`: 训练和测试特征
- `enhanced_demand_model_results.csv`: 模型性能结果
- `backtest_splits_results.csv`: 回测详细结果  
- `backtest_stability_results.csv`: 模型稳定性评估
- `pricing_and_replenishment_2023-07-01_07.csv`: **最终决策结果**
- `comprehensive_analysis_report.md`: 综合分析报告
- 各类可视化图表 (PNG格式)

## 核心参数配置

主要参数在 `config/config.py` 中配置：

```python
# 模型参数
RF_N_ESTIMATORS = 300          # 随机森林树数量
GB_LEARNING_RATE = 0.08        # 梯度提升学习率

# 定价参数  
DEFAULT_MARKUP = 0.30          # 默认加成率30%
MIN_MARKUP = 0.20              # 最小加成率20%
MAX_MARKUP = 0.40              # 最大加成率40%
SERVICE_LEVEL = 0.95           # 服务水平95%

# 预测周期
FORECAST_START_DATE = date(2023, 7, 1)   # 预测开始日期
FORECAST_END_DATE = date(2023, 7, 7)     # 预测结束日期
```

## 结果解读

### 最终决策文件

`pricing_and_replenishment_2023-07-01_07.csv` 包含每个品类每天的决策：

- `date`: 日期
- `category`: 品类名称  
- `price`: 建议销售价格(元/千克)
- `replenish_qty`: 建议补货量(千克)
- `demand_pred`: 预测需求量
- `cost_est`: 估计成本
- `markup`: 实际加成率
- `service_level`: 服务水平

### 业务指标

- **平均加成率**: 约30%，符合行业惯例
- **服务水平**: 95%，确保供应充足
- **预测精度**: 各品类R²多数>0.5，MAPE<30%
- **策略稳定性**: 通过交叉验证确保时间稳定性

## 模型特点

### 可复现性

- 固定随机种子确保结果一致
- 完整参数记录和日志追踪
- 标准化的数据处理流程

### 业务适应性

- 考虑蔬菜行业特点(损耗、季节性等)
- 灵活的约束条件设置
- 可解释的定价逻辑

### 扩展性

- 模块化设计，易于功能扩展
- 支持新品类和新特征添加
- 可调整的预测周期

## 注意事项

1. **数据质量**: 确保输入数据完整性和准确性
2. **参数调优**: 根据实际业务情况调整关键参数
3. **结果验证**: 建议结合业务经验验证决策合理性
4. **定期更新**: 随着新数据积累，定期重新训练模型

## 技术支持

如有问题，请检查：

1. 数据文件路径是否正确
2. Python环境和依赖包版本
3. 运行日志中的错误信息
4. 配置参数是否合理

## 版权声明

本项目仅用于学术研究和教学目的。