# 优化策略报告（定价与补货）

执行概览
- 模块：`VegetablePricingPipeline.run_optimization()` + `Visualizer.generate_all_visualizations()`
- 运行记录：`pipeline_execution_log_20250819_153920.json`
- 产出文件：
  - 日度优化结果：`outputs/results/optimization_results.csv`
  - 周策略汇总：`outputs/results/weekly_strategy.csv`
  - 可视化：
    - `outputs/figures/optimization_results.png`
    - `outputs/figures/summary_dashboard.png`

关键配置（来自 `config/config.json`）
- 优化周期：7 天（`optimization_horizon`）
- 目标服务水平：0.80（决定安全库存）
- 加价率搜索区间：1.6 ~ 1.8（`min_markup_ratio` ~ `max_markup_ratio`）
- 惩罚项：断货惩罚 8.0，损耗惩罚 0.5
- 蒙特卡洛样本数：50（用于利润期望与波动评估）

周度关键指标（`weekly_strategy.csv`）
- 总期望利润：-4698.99（负值，详见诊断）
- 平均利润率（(均价-均成本)/均成本）：0.7997（接近统一加价率 ≈ 1.80x）
- 均值：
  - 平均成本：11.64
  - 平均售价：20.95
  - 平均订货量：15.54
- 各品类期望利润（高→低）：
  - 水生根茎类：-236.22
  - 茄类：-308.30
  - 食用菌：-501.17
  - 花叶类：-627.60
  - 辣椒类：-1108.19
  - 花菜类：-1917.51

日度与区间统计（`optimization_results.csv`）
- 售价范围：10.05 ~ 35.57
- 订货量范围：4.64 ~ 37.54
- 单日最佳记录（亏损最少）：
  - 2025-08-25 水生根茎类，期望利润 -9.65
- 单日最差记录：
  - 2025-08-21 花菜类，期望利润 -410.03
- 各日总利润：
  - 2025-08-25：-171.98（最佳）
  - 2025-08-23：-382.51
  - 2025-08-20：-648.51
  - 2025-08-22：-820.08
  - 2025-08-24：-824.07
  - 2025-08-19：-828.25
  - 2025-08-21：-1023.60（最差）

可视化
- 优化结果图：`outputs/figures/optimization_results.png` 展示价格、数量与利润的整体走势与分布。
- 汇总仪表板：`outputs/figures/summary_dashboard.png` 汇总周策略、利润分布与品类对比。

诊断与建议
- 现象：尽管售价相对成本的加价率较高（≈1.80x），总期望利润仍为负。
- 可能原因：
  1) 需求模型对价格敏感度估计异常（多个品类价格弹性为正）。
  2) 断货惩罚权重较大（8.0）+ 需求不确定性放大，导致惩罚吞噬利润。
  3) 加价率上限限制与当前需求曲线组合，导致在边界附近选择仍无法覆盖风险成本。
- 建议：
  - 调整建模与参数：
    - 在特征中加入促销、天气、节假日等，约束或先验引导价格弹性为负。
    - 放宽或重新设定加价率搜索区间（如 1.2~2.2），并进行网格/贝叶斯搜索对比。
    - 降低 `stockout_penalty_weight` 或提高 `service_level` 的同时缩小不确定性估计来源。
    - 增大 `monte_carlo_samples` 提升期望估计稳定性。
  - 业务规则：
    - 设置最低毛利率或最低单日利润阈值的约束，避免负利润解被选择。
    - 对高波动品类单独设定价格上/下限与订货量缓冲。

可复现性
- 随机性：需求情景模拟以日期播种，结果可复现。
- 运行方式：`python3 main_pipeline.py -m optimization visualization` 或运行全流程。
