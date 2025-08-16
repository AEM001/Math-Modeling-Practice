# 蔬菜定价与补货策略分析系统

## 项目概述

这是一个基于机器学习的蔬菜定价与补货策略优化系统，采用精简化设计，专注核心功能。

## 项目结构

```
.
├── main_pipeline.py          # 主执行管道
├── config/
│   └── config.json          # 统一配置文件
├── src/                     # 源代码模块
│   ├── data_auditor.py      # 数据质量审计
│   ├── feature_engineer.py  # 特征工程
│   ├── demand_modeler.py    # 需求建模
│   └── optimizer.py         # 优化算法
├── data/                    # 数据文件
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
├── outputs/                # 输出文件
│   ├── figures/           # 图表文件
│   └── results/           # 分析结果
└── reports/               # 分析报告
```

## 核心功能模块

### 1. 数据质量审计 (data_auditor.py)
- 数据清洗与预处理
- 异常值检测和处理
- 数据质量评估报告

### 2. 特征工程 (feature_engineer.py)  
- 时间特征构建
- 价格相对特征
- 滞后与滚动特征
- 交互特征创建

### 3. 需求建模 (demand_modeler.py)
- 多模型对比（线性回归、随机森林、梯度提升等）
- 时间序列交叉验证
- 价格弹性估计

### 4. 优化算法 (optimizer.py)
- 启发式定价策略
- 基于弹性的补货优化
- 风险评估与利润预测

## 使用方法

### 完整运行
```bash
python main_pipeline.py
```

### 单模块运行
```bash
# 只运行数据审计
python main_pipeline.py --modules audit

# 运行特征工程和建模
python main_pipeline.py --modules features modeling

# 查看帮助
python main_pipeline.py --help
```

### 自定义配置
```bash
python main_pipeline.py --config custom_config.json
```

## 配置说明

所有配置参数集中在 `config/config.json` 中，包括：

- **data_paths**: 数据文件路径配置
- **output_paths**: 输出文件路径配置  
- **data_cleaning**: 数据清洗参数
- **feature_engineering**: 特征工程配置
- **modeling**: 建模参数配置
- **optimization**: 优化算法参数

## 输出结果

### 报告文件 (reports/)
- `data_audit_report.md` - 数据质量审计报告
- `feature_engineering_report.md` - 特征工程报告
- `demand_modeling_report.md` - 需求建模报告  
- `optimization_report.md` - 优化策略报告

### 结果文件 (outputs/results/)
- `demand_model_results.csv` - 需求模型评估结果
- `optimization_results.csv` - 详细优化结果
- `weekly_strategy.csv` - 周策略汇总

## 系统特点

### 精简设计
- 移除冗余功能，专注核心价值
- 统一配置管理，简化参数调优
- 模块化设计，支持独立运行

### 性能优化  
- 减少计算复杂度，提升运行速度
- 优化数据处理流程
- 智能特征选择，降低维度

### 易用性
- 命令行友好界面
- 详细的执行日志
- 自动化的报告生成

## 技术架构

- **数据处理**: Pandas + NumPy
- **机器学习**: Scikit-learn  
- **配置管理**: JSON配置文件
- **报告生成**: Markdown格式

## 运行环境

- Python 3.7+
- 主要依赖包:
  - pandas
  - numpy  
  - scikit-learn
  - scipy

## 性能指标

在标准硬件上的运行时间（以46,599条记录为例）：
- 数据质量审计: ~0.3秒
- 特征工程: ~1.1秒  
- 需求建模: ~50秒
- 优化算法: ~0.02秒
- **总计: ~56秒**

## 项目优势

1. **功能完整**: 涵盖数据处理到策略优化的完整流程
2. **运行高效**: 精简设计，快速执行
3. **配置灵活**: 统一的配置管理系统
4. **结果可解释**: 详细的分析报告和指标输出
5. **模块化**: 支持按需运行特定功能模块

---

**最后更新**: 2025-08-16
