# 需求建模报告（SARIMAX）

执行概览
- 模块：`VegetablePricingPipeline.run_demand_modeling()`
- 运行记录：`pipeline_execution_log_20250819_153920.json`
- 产出文件：
  - 结果明细：`outputs/results/demand_model_results.csv`
  - 模型文件：`outputs/models/*.pkl`
  - 可视化：`outputs/figures/demand_model_performance.png`
- 使用外生变量（来自 `config/config.json`）：`["ln_price", "is_weekend", "time_trend"]`

总体结果
- 训练完成 6 个品类的 SARIMAX 模型。
- 测试集 R2 值均为负，说明当前模型对波动的解释力不足（在测试集上不如简单均值基线）。
- 价格弹性存在正值与负值，部分为正（价格越高销量越高），提示数据或建模设定存在偏差需复核。

各品类建模摘要（来自 `outputs/results/demand_model_results.csv`）
- 花叶类
  - Test R2: -0.2861，MAE: 0.9762
  - 价格弹性: -0.9374
  - 模型阶数: (2, 0, 2)，季节阶数: (2, 1, 0, 7)
  - 模型文件: `outputs/models/花叶类_sarimax.pkl`
- 辣椒类
  - Test R2: -0.4480，MAE: 1.2965
  - 价格弹性: +0.9146
  - 模型阶数: (1, 0, 1)，季节阶数: (2, 1, 0, 7)
  - 模型文件: `outputs/models/辣椒类_sarimax.pkl`
- 花菜类
  - Test R2: -0.3179，MAE: 0.6358
  - 价格弹性: +0.1245
  - 模型阶数: (2, 0, 0)，季节阶数: (2, 1, 0, 7)
  - 模型文件: `outputs/models/花菜类_sarimax.pkl`
- 食用菌
  - Test R2: -0.9028，MAE: 1.0778
  - 价格弹性: -0.3690
  - 模型阶数: (0, 0, 0)，季节阶数: (2, 1, 0, 7)
  - 模型文件: `outputs/models/食用菌_sarimax.pkl`
- 水生根茎类
  - Test R2: -0.2888，MAE: 1.1701
  - 价格弹性: +0.1313
  - 模型阶数: (0, 0, 3)，季节阶数: (1, 1, 1, 7)
  - 模型文件: `outputs/models/水生根茎类_sarimax.pkl`
- 茄类
  - Test R2: -0.0466，MAE: 0.9085
  - 价格弹性: +0.1610
  - 模型阶数: (0, 0, 2)，季节阶数: (0, 1, 1, 7)
  - 模型文件: `outputs/models/茄类_sarimax.pkl`

可视化与健壮性
- `src/visualizer.py` 已针对结果文件中缺失 `model`、`train_r2` 等列做了兼容：
  - 若无 `train_r2`，改绘制 `test_r2` 分布直方图。
  - 若无 `model` 列，填充默认值 `SARIMAX`，避免 KeyError。
- 相关图表：`outputs/figures/demand_model_performance.png`。
