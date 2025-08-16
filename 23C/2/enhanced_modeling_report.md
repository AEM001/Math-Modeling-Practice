# 增强需求建模报告

## 模型性能汇总

### 各品类最佳模型（按测试R²）
- **花叶类**: GradientBoosting (测试R²=0.9282)
  - 价格弹性: nan
- **辣椒类**: RandomForest (测试R²=0.8372)
  - 价格弹性: nan
- **花菜类**: RandomForest (测试R²=0.5873)
  - 价格弹性: nan
- **食用菌**: GradientBoosting (测试R²=0.8445)
  - 价格弹性: nan
- **茄类**: RandomForest (测试R²=0.8158)
  - 价格弹性: nan
- **水生根茎类**: RandomForest (测试R²=0.6759)
  - 价格弹性: nan

## 方法论总结
- **OLS**: 普通最小二乘法基线
- **Huber**: 鲁棒回归，减少异常值影响
- **2SLS**: 工具变量法，处理价格内生性
- **RandomForest**: 随机森林，捕捉非线性关系
- **GradientBoosting**: 梯度提升，强预测性能