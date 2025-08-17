# 问题3 建模方案

本方案面向“7月1日单日”的补货与定价决策，强调可赛用、可落地与不复杂。整体分为：需求预测 → 定价策略 → 线性优化求解补货与选品（三十个左右 SKU）。

---

## 1. 目标与约束
- 目标：在尽量满足需求的前提下，使当日收益最大（销售收入 − 进货成本）。
- 约束：
  - 单品总数限制：可售单品个数控制在 27–33 个。
  - 最小陈列量：每个被选中的单品订购量 ≥ 2.5 kg。
  - 合理上限：每个单品的订购量不超过一个保守上界，以避免过量备货。

---

## 2. 数据与变量定义
数据来源：
- 周表：`23C/3/品种周统计表_20230624-30.csv`
  - 关键字段：`周总销量(千克)`、`销量标准差`、`最大日销量`、`平均正常售价`、`平均打折售价`、`平均售价`、`平均批发价`、`平均成本加成率`、`平均损耗率`、`销售天数`、`单品编码`、`单品名称`、`分类名称`。
- 时序表：`23C/3/可售品种时间序列表_20230624-30.csv`
  - 关键字段：`销售日期`、`单品编码`、`总销量(千克)`。

符号（以 SKU i 表示）：
- 参数（由数据计算得到）：
  - $c_i$：单位进货成本 = `平均批发价`。
  - $c_{\text{eff},i}$：损耗修正后的单位有效成本 = $c_i \times (1 + \text{平均损耗率})$。
  - $p_i$：正常销售价，优先取`平均正常售价`；若缺失，用 $c_i \times (1 + \text{平均成本加成率})$ 代替。
  - $p^d_i$：打折价，优先取`平均打折售价`；若缺失，取 $0.85 \times p_i$。
  - $\rho_i$：历史打折占比 = $\frac{\text{周打折销量(千克)}}{\text{周总销量(千克)}}$（缺失则置 $\rho_i=0$）。
  - $\bar{D}_i$：周均日销量 = $\frac{\text{周总销量(千克)}}{\max(1, \text{销售天数})}$。
  - $s_i(2023\text{-}06\text{-}24)$：时序表中 2023-06-24 的`总销量(千克)`（若无记录则视为缺失）。
  - $f_i$（日效应系数）：如果 $s_i(06\text{-}24)$ 有值，则 $f_i = \frac{s_i(06\text{-}24)}{\bar{D}_i}$；否则 $f_i = 1$。为稳健，截断在 $[0.85, 1.15]$。
  - $D_i$（需求预测，见下一节）：单日总需求预测量。
  - $U_i$（单品订货上界）：$U_i = \min(\text{最大日销量}, 1.2 \times D_i)$。
- 决策变量：
  - $y_i \in \{0,1\}$：是否选择该 SKU。
  - $x_i \geq 0$：订货量（kg）。
  - $s^r_i, s^d_i \geq 0$：分别为正常价与打折价的销售量（kg）。

---

## 3. 需求预测（经验贝叶斯收缩 + 周几效应）
目的：给出 2023-07-01（周六）每个 SKU 的单日需求预测 $D_i$，利用近一周信息，考虑周几效应，并对低样本 SKU 做经验贝叶斯收缩。

- 基础计算：
  - $\bar{D}_i = \text{周均日销量} = \frac{\text{周总销量(千克)}}{\max(1, \text{销售天数})}$。
  - $\bar{D}_{\text{cat},k} = \text{类别 } k \text{ 的周均日销量均值（同类均值作为先验）}$。
  - $\sigma_i = \text{销量标准差}$（若缺失则用 $0.2 \times \bar{D}_i$ 估计）。

- 经验贝叶斯收缩（稳定低样本 SKU 预测）：
  - 权重：$w_i = \frac{\text{销售天数}}{\text{销售天数} + \tau}$，$\tau=3$（可调）。
  - 收缩基线：$\tilde{D}_i = w_i \cdot \bar{D}_i + (1 - w_i) \cdot \bar{D}_{\text{cat},k}$。
  - 直觉：销售天数越少，越向类别均值收缩；销售天数充足则保持自身均值。

- 周几效应修正：
  - 参考上一周同为周六的 2023-06-24 销量 $s_i(06-24)$。
  - $f_i = \frac{s_i(06\text{-}24)}{\bar{D}_i}$（若无 06-24 数据则 $f_i = 1$）。
  - 为稳健，截断 $f_i \in [0.85, 1.15]$。

- 最终预测与波动保守带：
  - 初步预测：$D_i' = \min(\text{最大日销量}, f_i \times \tilde{D}_i)$。
  - 波动保守调整：$D_i = \max(2.5, D_i' - \beta \cdot \sigma_i)$，$\beta=0.25$（可调）。
  - 直觉：既考虑类别先验与周几效应，也为高波动品种留出保守缓冲，且确保满足最小陈列量。

---

## 4. 定价策略（损耗感知毛利驱动微调）
为便于执行与合理性：
- 正常价基准：
  - 首选历史`平均正常售价`（贴合顾客心理锚点与最近行情）。
  - 若缺失，则用成本加成：$p_{i,\text{base}} = c_i \times (1 + \text{平均成本加成率})$。

- 损耗感知毛利驱动微调（±10%）：
  - 单位毛利评分：$g_i = (p_{i,\text{base}} - c_{\text{eff},i}) \cdot (1 - \rho_i) + (p^d_i - c_{\text{eff},i}) \cdot \rho_i$。
  - 评分百分位：$\text{rank-pct}(g_i) \in [0,1]$（在所有 SKU 中的毛利评分排名百分比）。
  - 微调公式：$p_i = \text{round}(p_{i,\text{base}} \cdot (1 + \kappa \cdot (\text{rank-pct}(g_i) - 0.5)), 0.1)$。
  - 限制范围：$p_i \in [0.9 \cdot p_{i,\text{base}}, 1.1 \cdot p_{i,\text{base}}]$，$\kappa=0.1$（可调）。
  - 直觉：高毛利潜力的 SKU 适度提价（最多 +10%），低毛利的适度降价（最多 -10%）。

- 打折价：
  - 首选历史`平均打折售价`；若缺失，设 $p^d_i = 0.85 \times p_i$。

- 折扣比例（用于销量分解）：$\rho_i = \text{周打折占比}$。无需再设触发规则，直接把 $D_i$ 分解为正常价与打折价的可售上限：
  - $D^r_i = (1-\rho_i) \cdot D_i$
  - $D^d_i = \rho_i \cdot D_i$

- 价格落地：价格四舍五入到 0.1 元，确保执行简便。

---

## 5. 线性优化模型（MILP，简单可解）
目标函数（最大化当日利润 + 品类多样性奖励）：
$\max \sum_i \left[ p_i \cdot s^r_i + p^d_i \cdot s^d_i - c_{\text{eff},i} \cdot x_i \right] + \varepsilon \cdot \sum_k z_k$

约束：
- 销量与库存：
  - $s^r_i \leq D^r_i$
  - $s^d_i \leq D^d_i$
  - $s^r_i + s^d_i \leq x_i$
- 订货上下界与选品：
  - $2.5 \cdot y_i \leq x_i \leq U_i \cdot y_i$
  - $27 \leq \sum_i y_i \leq 33$
- 按 SKU 的差异化服务水平：
  - $s^r_i + s^d_i \geq \alpha_i \cdot D_i \cdot y_i$
  - $\alpha_i = \text{clip}[\alpha_0 + \mu \cdot \text{rank-pct}(g_i) - \nu \cdot (\sigma_i/\tilde{\sigma}), \alpha_{\min}, \alpha_{\max}]$
  - 推荐参数：$\alpha_0=0.9, \mu=0.05, \nu=0.05$，$\alpha_i \in [0.85, 0.98]$
  - $\tilde{\sigma}$ 为全体或同类销量标准差的中位数
  - 直觉：高毛利、低波动的 SKU 服务水平要求更高
- 品类多样性变量：
  - $z_k \in \{0,1\}$ 表示类别 $k$ 是否被选中
  - $z_k \geq y_i, \forall i \in \text{类别} k$（只要选了类别 $k$ 中任一 SKU，$z_k$ 就为 1）
  - $\varepsilon$ 为品类多样性奖励权重（建议等价于 0.2 元/类）
- 非负与二元：$s^r_i, s^d_i, x_i \geq 0$；$y_i, z_k \in \{0,1\}$

可选（若需要加严）：
- 预算约束：$\sum_i c_i \cdot x_i \leq B$（由门店现金流或仓储能力确定）。
- 全局服务水平下限：$\sum_i (s^r_i + s^d_i) \geq \alpha_{\text{global}} \cdot \sum_i (D_i \cdot y_i)$，$\alpha_{\text{global}} \approx 0.92$。

说明：该模型是标准 MILP，变量规模为 $O(SKU + 类别)$，可用 PuLP/OR-Tools/HiGHS/Gurobi 等求解；线性且收敛快。

---

## 6. 计算流程（落地步骤）
0) 滞销过滤与预选：
   - 剔除 `销售天数` ≤ 2 或 `周总销量(千克)` ≤ 5 的 SKU。
   - 计算初步 $g_i = (p_i - c_{\text{eff},i}) \cdot (1 - \rho_i) + (p^d_i - c_{\text{eff},i}) \cdot \rho_i$。
   - 按 $g_i \times D_i$ 评分预选约 40 个 SKU 作为候选集。

1) 读取两张表：`品种周统计表_20230624-30.csv` 与 `可售品种时间序列表_20230624-30.csv`。

2) 需求预测与价格计算：
   - 计算类别均值 $\bar{D}_{\text{cat},k}$ 和各 SKU 的 $\bar{D}_i$、$\sigma_i$。
   - 应用经验贝叶斯收缩：$\tilde{D}_i = w_i \cdot \bar{D}_i + (1-w_i) \cdot \bar{D}_{\text{cat},k}$，$w_i = \frac{\text{销售天数}}{\text{销售天数}+\tau}$。
   - 由时序表取 2023-06-24 的同品销量 $s_i(06-24)$，计算 $f_i$ 并截断到 $[0.85, 1.15]$。
   - 应用周几修正与波动保守带：$D_i = \max(2.5, \min(\text{最大日销量}, f_i \cdot \tilde{D}_i) - \beta \cdot \sigma_i)$。
   - 设置订货上界：$U_i = \min(\text{最大日销量}, 1.2 \cdot D_i)$。
   - 计算基准价格 $p_{i,\text{base}}$ 与打折价 $p^d_i$。
   - 计算毛利评分 $g_i$ 并排序，得到 $\text{rank-pct}(g_i)$。
   - 应用损耗感知毛利驱动微调：$p_i = \text{round}(p_{i,\text{base}} \cdot (1 + \kappa \cdot (\text{rank-pct}(g_i) - 0.5)), 0.1)$，限制在 $\pm 10\%$。

3) 计算 SKU 服务水平与销量分解：
   - 计算 $\tilde{\sigma}$（销量标准差中位数）。
   - 设置差异化服务水平：$\alpha_i = \text{clip}[\alpha_0 + \mu \cdot \text{rank-pct}(g_i) - \nu \cdot (\sigma_i/\tilde{\sigma}), \alpha_{\min}, \alpha_{\max}]$。
   - 计算 $D^r_i = (1-\rho_i) \cdot D_i$ 与 $D^d_i = \rho_i \cdot D_i$。

4) 构建与求解 MILP：
   - 构建目标函数：$\max \sum_i [p_i \cdot s^r_i + p^d_i \cdot s^d_i - c_{\text{eff},i} \cdot x_i] + \varepsilon \cdot \sum_k z_k$。
   - 添加所有约束，包括按 SKU 的服务水平约束与品类多样性变量。
   - 使用预选结果作为 MILP 的热启动初始解。
   - 求解得到 $x_i$、$y_i$、$s^r_i$、$s^d_i$、$z_k$。

5) 结果后处理：
   - 仅输出 $y_i=1$ 的 SKU；
   - $x_i$ 向上取到 0.1kg 粒度，并确保 $x_i \geq 2.5$kg；
   - 输出定价（$p_i$、$p^d_i$）与预计销量分解（$s^r_i$、$s^d_i$）。
   - 计算预计毛利 = $(p_i - c_{\text{eff},i}) \cdot s^r_i + (p^d_i - c_{\text{eff},i}) \cdot s^d_i$。

---

## 7. 输出格式（建议）
- 字段：`单品编码`、`单品名称`、`分类名称`、`订货量kg(x_i)`、`正常价(p_i)`、`打折价(p^d_i)`、`预计正常销量(s^r_i)`、`预计打折销量(s^d_i)`、`预计毛利(元)`。
- 预计毛利 = $(p_i - c_{\text{eff},i}) \cdot s^r_i + (p^d_i - c_{\text{eff},i}) \cdot s^d_i$。

---

## 8. 无求解器时的简易贪心备选（可作为对照）
- 评分 $g_i$（单位毛利/kg）：$g_i = (p_i - c_{\text{eff},i}) \cdot (1 - \rho_i) + (p^d_i - c_{\text{eff},i}) \cdot \rho_i$。
- 类别多样性加分：
  - 按 $g_i \times \min(D_i, U_i)$ 从大到小排序。
  - 若某 SKU 是其所属类别的首个被选中的品种，则额外加分 $\delta$（建议等价于 0.5 元/kg）。
  - 这种简单调整可在贪心中也实现品类多样性，无需复杂算法。
- 选品：按调整后的评分从大到小排序，选前 27–33 个 SKU。
- 订货量：$x_i = \min(D_i, U_i)$，并满足 $x_i \geq 2.5$。
- 定价：同第 4 节，包括损耗感知毛利驱动微调。

该方法无需求解器，性能略逊但非常稳健、可解释，且通过简单调整也能实现品类多样性。


---

## 9. 符号表（汇总）
- $c_i$：`平均批发价`；$c_{\text{eff},i}$：损耗修正后的有效成本；$p_i$：正常价；$p^d_i$：打折价；$\rho_i$：周打折占比。
- $\bar{D}_i$：周均日销量；$\bar{D}_{\text{cat},k}$：类别 $k$ 的均值；$\tilde{D}_i$：经验贝叶斯收缩后的基线；$f_i$：周几修正系数。
- $\sigma_i$：销量标准差；$\tilde{\sigma}$：标准差中位数；$w_i$：收缩权重；$D_i$：最终需求预测。
- $U_i$：订货上界；$x_i$：订货量；$y_i$：是否选品；$z_k$：类别 $k$ 是否被选。
- $s^r_i / s^d_i$：正常价/打折价预计销量；$D^r_i / D^d_i$：其上限；$\alpha_i$：SKU 服务水平。
- $g_i$：单位毛利评分；$\text{rank-pct}(g_i)$：评分百分位。
- 参数：$\tau=3$（收缩强度），$\beta=0.25$（波动保守系数），$\kappa=0.1$（价格微调强度），
  $\alpha_0=0.9, \mu=0.05, \nu=0.05$（服务水平参数），$\varepsilon \approx 0.2$（品类多样性奖励）。
- 约束：$2.5 \leq x_i \leq U_i$，$27 \leq \sum_i y_i \leq 33$，$s^r_i + s^d_i \geq \alpha_i \cdot D_i \cdot y_i$。
