

# 问题二改进方案（Plan v2）

基于你在 `23C/2/C题.md` 的现状记录与 [23C/2/README.md](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/README.md:0:0-0:0) 的既有流程与结果文件（如 [validation_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/validation_results.csv:0:0-0:0)、[weekly_category_strategy.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/weekly_category_strategy.csv:0:0-0:0)、[wholesale_forecasts.json](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/wholesale_forecasts.json:0:0-0:0)），以下给出针对“测试集效果不佳”的系统性改进方案与可执行计划。

## 现状与痛点

- 现用模型：单品级双对数 OLS，聚合至品类，网格搜索优化（参考 [README.md](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/README.md:0:0-0:0)）。
- 痛点：测试集表现不稳（你在 `C题.md` 中提出），平均拟合度 R²≈0.28，可能存在价格内生性、异方差、非线性、样本少导致高方差、时间效应不确定等。

## 改进目标与衡量指标

- 模型侧：提升测试集稳定性（更小的MAPE/sMAPE、log-RMSE），弹性估计更稳（置信区间更窄）。
- 业务侧：提高一周总利润与利润稳定性，降低缺货与损耗风险。
- 可用性：价格/补货策略日间更平滑、在历史合理区间内。

## 总体策略

- 单品建模为主，品类聚合；但引入更鲁棒的估计（RLM/2SLS/层次模型）与非线性（GAM/分段）。
- 加强数据质量与特征工程（周几、趋势、相对价格、滞后项）。
- 优化器加入风险与平滑约束，面向利润-风险均衡，而非“点预测最大化”。

---

## 详细改进措施与落地计划

1) 数据质量审计与清洗（高优先）

- 排除/标注异常样本：零价/异常低价（如 P/C<1）、销量或价格极端值（MAD/Z-score on lnP, lnQ）。
- 促销/折扣处理：保留并加促销哑变量；若无法稳定识别则从建模样本中剔除，另做敏感性分析。
- “潜在售罄”识别（无库存列时）：销量接近该单品历史高分位（如 P95），且次日销量显著回升、当日价格并不高 → 标注为可能受库存约束样本，保守处理或剔除。
- 输出：数据审计报告与剔除规则清单；更新 [train_data.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/train_data.csv:0:0-0:0) / [test_data.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/test_data.csv:0:0-0:0)。

2) 探索性分析与时间效应（中优先）

- 周几效应：Kruskal-Wallis 检验销量在周几上的差异；若显著，用成组事后检验（scipy + 多重比较校正）确认差异组。
- 非线性与分段性：对 lnP–lnQ 绘制分位数散点与样条拟合，检查是否存在明显非线性或阈值效应。
- 输出：EDA图表与周几结论；确定是否加入周几哑变量/交互项。

3) 需求建模升级（高优先）

- 基线增强：OLS + HC3稳健方差；加入特征
  - 周几哑变量、时间趋势（rolling mean/volatility）、滞后销量/价格（尽量避免泄漏）。
  - 相对价格：单品价格相对于品类日均价（衡量同类竞争）。
- 鲁棒回归：RLM（Huber/Tukey）缓解异常点影响。
- 内生性修正：2SLS
  - 工具变量建议：当日/滞后批发价（[wholesale_forecasts.json](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/wholesale_forecasts.json:0:0-0:0)/历史C）、滞后价格、品类价格指数等。
  - 合法性检验：第一阶段F>10，过度识别检验（如Sargan）。
- 分层/混合效应模型：品类内做随机截距/随机斜率（lnP），对样本少的单品进行“部分池化”以稳定弹性。
- 非线性候选：GAM或分段（在 lnP 的分位点设断点，强制单调递减约束）。
- ML保底：RF/GBDT 作为参考基线（强调可解释检查与单调性验证）。
- 模型选择：基于时间阻塞CV（rolling origin），以测试期指标与稳定性优选，并保留透明可解释的方案为主模型。
- 输出：更新 [demand_models.json](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/demand_models.json:0:0-0:0)、[demand_model_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/demand_model_results.csv:0:0-0:0)、[validation_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/validation_results.csv:0:0-0:0)（新增列：方法、特征集、CV方案、置信区间）。

4) 优化器升级（高优先）

- 风险与不确定性：引入分位数需求（P50/P80/P90）或对需求分布做自助法区间，采用机会约束或鲁棒优化：
  - 缺货惩罚（lost sales penalty）与损耗惩罚（废弃/降价清仓）纳入目标。
- 价格平滑：加入 |P_t − P_{t−1}| 的惩罚项，避免大幅波动；或绑定最大日变动幅度。
- 价格边界：历史经验边界或数据驱动（按品类日价分位区间）；继续保留 C < P ≤ 2C 约束。
- 供给/空间约束（可选）：若有货架/预算上限，加入类别或全店层面的容量/成本约束。
- 求解：7天联合优化（非独立逐日），采用细粒度网格 + 动态规划/启发式搜索，保证平滑项考虑；或局部搜索微调。
- 输出：更新 [daily_optimization_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/daily_optimization_results.csv:0:0-0:0) 与 [weekly_category_strategy.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/weekly_category_strategy.csv:0:0-0:0)（新增：风险等级、价格平滑度、上下界、敏感性）。

5) 回测与报告（高优先）

- 回测：滚动起点（walk-forward）对策略级利润、缺货率、损耗率进行评估；与当前 [validation_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/validation_results.csv:0:0-0:0) 对比。
- 报告：在 [report_generator.py](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/report_generator.py:0:0-0:0) 增加
  - 单品/品类弹性分布、模型对比、价格-销量曲线、策略灵敏度（价格±5~10%）。
  - 输出“建议策略 + 备用策略”（保守/激进）。
- 输出：强化版分析报告（PDF/Markdown）与可解释图表。

---

## 集成与文件落点

- 流程编排：在 [main_analysis.py](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/main_analysis.py:0:0-0:0) 增加开关与步骤管线（数据审计→特征→建模→优化→汇总→报告）。
- 优化器：在 [vegetable_optimizer.py](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/vegetable_optimizer.py:0:0-0:0) 增加风险与平滑项、分位数/机会约束求解器。
- 报告：在 [report_generator.py](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/report_generator.py:0:0-0:0) 扩展指标与图表。
- 数据/结果：复用并扩展 [train_data.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/train_data.csv:0:0-0:0)、[test_data.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/test_data.csv:0:0-0:0)、[demand_model_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/demand_model_results.csv:0:0-0:0)、[validation_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/validation_results.csv:0:0-0:0)、[daily_optimization_results.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/daily_optimization_results.csv:0:0-0:0)、[weekly_category_strategy.csv](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/weekly_category_strategy.csv:0:0-0:0)、[wholesale_forecasts.json](cci:7://file:///Users/Mac/Downloads/Math-Modeling-Practice/23C/2/wholesale_forecasts.json:0:0-0:0)。

## 已经确认的关键参数

- 价格日变动上限（±10%/天）与整体上下界策略（以品类日均价的历史分位为上界，确保定价不会过高；而下界可考虑C的倍数（例如不低于成本价），避免亏损）。
- 风险偏好/服务水平目标（满足 P80 需求，缺货率<5%）。
- 缺货与损耗惩罚的相对权重（**缺货惩罚 > 损耗惩罚** ，比如缺货率高于5%时，损失的成本可能更高，需要通过优化策略减少缺货发生的可能。损耗惩罚可设置为较低的权重，因为它通常更容易通过库存管理来控制。）。
- 允许引入分层/混合效应（计算开销稍高，但稳定性更好）。
