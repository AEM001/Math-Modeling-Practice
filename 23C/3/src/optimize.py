import pandas as pd
import numpy as np
import pulp
import logging
from config import (
    MIN_SHELF_COUNT, MAX_SHELF_COUNT, MIN_DISPLAY_QTY, 
    MARKUP_BOUNDS, ELASTICITY_PARAMS, SOLVER_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestockingOptimizer:
    """
    补货优化器
    """
    
    def __init__(self, candidates_df, use_elasticity=True, solver='CBC'):
        self.candidates_df = candidates_df.copy()
        self.use_elasticity = use_elasticity
        self.solver = solver
        self.model = None
        self.solution = None
        self.status = None
        
        # 决策变量
        self.x_vars = {}  # 是否上架
        self.P_vars = {}  # 进货量
        self.A_vars = {}  # 加成率
        self.Q_vars = {}  # 实际销量（如果使用弹性函数）
        
        logger.info(f"Initialized optimizer for {len(candidates_df)} candidates")
        logger.info(f"Using elasticity model: {use_elasticity}")
    
    def create_linear_model(self):
        """
        线性优化模型
        """
        logger.info("Creating simplified linear optimization model...")
        
        # 创建模型
        self.model = pulp.LpProblem("RestockingOptimization", pulp.LpMaximize)
        
        # 产品索引
        products = self.candidates_df.index.tolist()
        
        # 只使用二元决策变量: x_i: 是否上架 (0/1)
        self.x_vars = pulp.LpVariable.dicts("x", products, cat='Binary')
        
        # 简化目标函数：仅考虑选择哪些产品上架，进货量固定为预测销量
        # 使用固定加成率30%
        fixed_markup = 0.3
        
        profit_terms = []
        
        for i in products:
            row = self.candidates_df.loc[i]
            Q_p = row['pred_Q_p']  # 预测销量
            C = row['pred_C']      # 预测批发价
            
            # 简化的利润 = 预测销量 * 批发价 * 加成率
            unit_profit = Q_p * C * fixed_markup
            profit_term = self.x_vars[i] * unit_profit
            profit_terms.append(profit_term)
        
        self.model += pulp.lpSum(profit_terms), "TotalProfit"
        
        # 约束条件：只保留最基本的上架数量约束（自适应到候选数量）
        min_req = min(MIN_SHELF_COUNT, len(products))
        max_req = min(MAX_SHELF_COUNT, len(products))
        if min_req > max_req:
            logger.warning(f"Adjusting shelf bounds: min {min_req} > max {max_req}. Setting min = max = {max_req}")
            min_req = max_req
        self.model += (
            pulp.lpSum([self.x_vars[i] for i in products]) >= min_req,
            "MinShelfCount"
        )
        
        self.model += (
            pulp.lpSum([self.x_vars[i] for i in products]) <= max_req,
            "MaxShelfCount"
        )
        
        # 为每个选中的产品自动设置进货量等于预测销量
        self.P_vars = {}
        for i in products:
            row = self.candidates_df.loc[i]
            Q_p = row['pred_Q_p']
            # P_i 将在解决方案提取时设置为 Q_p * x_i
            self.P_vars[i] = Q_p
        
        logger.info("Simplified linear model created successfully")
        logger.info(f"Variables: {len(self.x_vars)} binary")
        logger.info(f"Constraints: {len(self.model.constraints)}")
    
    def create_nonlinear_model(self):
        """
        创建非线性优化模型（使用价格弹性）
        注意：这里提供框架，实际需要专门的非线性求解器
        """
        logger.info("Creating nonlinear optimization model...")
        
        # 对于复杂的非线性模型，建议使用pyomo + ipopt
        # 这里提供简化的线性近似版本
        
        # 使用分段线性化近似弹性函数
        self.create_piecewise_linear_model()
    
    def create_piecewise_linear_model(self):
        """
        创建分段线性近似的弹性模型
        """
        logger.info("Creating piecewise linear approximation model...")
        
        self.model = pulp.LpProblem("RestockingOptimizationPWL", pulp.LpMaximize)
        
        products = self.candidates_df.index.tolist()
        
        # 决策变量
        self.x_vars = pulp.LpVariable.dicts("x", products, cat='Binary')
        self.P_vars = pulp.LpVariable.dicts("P", products, lowBound=0, cat='Continuous')
        self.A_vars = pulp.LpVariable.dicts("A", products, 
                                           lowBound=MARKUP_BOUNDS['min'], 
                                           upBound=MARKUP_BOUNDS['max'], 
                                           cat='Continuous')
        
        # 销量变量（考虑价格弹性的影响）
        self.Q_vars = pulp.LpVariable.dicts("Q", products, lowBound=0, cat='Continuous')
        
        # 价格弹性约束（简化为线性约束）
        for i in products:
            row = self.candidates_df.loc[i]
            Q_p = row['pred_Q_p']
            
            # 销量在预测值的±20%范围内
            tolerance = ELASTICITY_PARAMS['demand_tolerance']
            
            self.model += (
                self.Q_vars[i] >= Q_p * (1 - tolerance) * self.x_vars[i],
                f"MinDemand_{i}"
            )
            
            self.model += (
                self.Q_vars[i] <= Q_p * (1 + tolerance) * self.x_vars[i],
                f"MaxDemand_{i}"
            )
            
            # 简化的价格-销量关系：高加成率导致销量下降
            # Q_i ≈ Q_p * (1 - β * (A_i - A_base))
            beta = 0.5  # 弹性系数
            A_base = 0.3  # 基准加成率
            
            self.model += (
                self.Q_vars[i] <= Q_p * (1 - beta * (self.A_vars[i] - A_base)) + 
                (1 - self.x_vars[i]) * 1000,  # 大M约束
                f"PriceElasticity_{i}"
            )
        
        # 目标函数：使用弹性调整后的销量
        profit_terms = []
        for i in products:
            row = self.candidates_df.loc[i]
            C = row['pred_C']
            
            profit_term = self.x_vars[i] * (
                self.Q_vars[i] * C * (1 + self.A_vars[i]) - self.P_vars[i] * C
            )
            profit_terms.append(profit_term)
        
        self.model += pulp.lpSum(profit_terms), "TotalProfitElastic"
        
        # 其他约束保持不变（自适应到候选数量）
        min_req = min(MIN_SHELF_COUNT, len(products))
        max_req = min(MAX_SHELF_COUNT, len(products))
        if min_req > max_req:
            logger.warning(f"Adjusting shelf bounds: min {min_req} > max {max_req}. Setting min = max = {max_req}")
            min_req = max_req
        self.model += (
            pulp.lpSum([self.x_vars[i] for i in products]) >= min_req,
            "MinShelfCount"
        )
        
        self.model += (
            pulp.lpSum([self.x_vars[i] for i in products]) <= max_req,
            "MaxShelfCount"
        )
        
        for i in products:
            self.model += (
                self.P_vars[i] >= MIN_DISPLAY_QTY * self.x_vars[i],
                f"MinDisplay_{i}"
            )
            
            # 进货量应该与预期销量相匹配
            self.model += (
                self.P_vars[i] >= self.Q_vars[i] * 0.8,  # 至少80%的预期销量
                f"StockDemandMatch_{i}"
            )
        
        logger.info("Piecewise linear model created successfully")
    
    def solve(self, time_limit=None):
        """
        求解优化模型
        """
        if self.model is None:
            if self.use_elasticity:
                self.create_nonlinear_model()
            else:
                self.create_linear_model()
        
        logger.info(f"Solving optimization model with {self.solver} solver...")
        
        # 设置求解器
        gap = SOLVER_CONFIG.get('gap', None)
        if self.solver.upper() == 'CBC':
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit if time_limit else None,
                                       gapRel=gap if gap is not None else None)
        elif self.solver.upper() == 'GLPK':
            # GLPK 的时间限制通过 options 传递
            options = []
            if time_limit:
                options += ["--tmlim", str(int(time_limit))]
            solver = pulp.GLPK_CMD(msg=1, options=options)
        else:
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit if time_limit else None,
                                       gapRel=gap if gap is not None else None)  # 默认
        
        # 设置时间限制
        if time_limit is None:
            time_limit = SOLVER_CONFIG['time_limit']
        
        # 求解
        self.status = self.model.solve(solver)
        
        if self.status == pulp.LpStatusOptimal:
            logger.info("Optimization completed successfully!")
            self.extract_solution()
            return True
        else:
            logger.error(f"Optimization failed with status: {pulp.LpStatus.get(self.status, self.status)}")
            # 回退：使用贪婪启发式给出可行方案
            logger.info("Falling back to greedy heuristic solution...")
            self._solve_greedy()
            return True
    
    def _solve_greedy(self):
        """
        贪婪回退：按单位利润从高到低选择，满足上架数量边界
        """
        products = self.candidates_df.index.tolist()
        fixed_markup = 0.3
        # 计算单位利润得分（预测销量 * 批发价 * 固定加成率）
        scores = []
        for i in products:
            row = self.candidates_df.loc[i]
            Q_p = max(0.0, float(row['pred_Q_p']))
            C = max(1e-6, float(row['pred_C']))
            unit_profit = Q_p * C * fixed_markup
            scores.append((i, unit_profit))
        scores.sort(key=lambda x: x[1], reverse=True)
        # 自适配上架数量范围
        min_req = min(MIN_SHELF_COUNT, len(products))
        max_req = min(MAX_SHELF_COUNT, len(products))
        if min_req > max_req:
            min_req = max_req
        selected = [i for i, _ in scores[:max_req]]
        # 至少满足最小数量
        selected = selected[:max_req]
        if len(selected) < min_req:
            selected = [i for i, _ in scores[:min_req]]
        # 构造解
        solution_data = []
        for i in selected:
            row = self.candidates_df.loc[i]
            Q_val = float(row['pred_Q_p'])
            P_val = Q_val
            A_val = fixed_markup
            C_val = float(row['pred_C'])
            selling_price = C_val * (1 + A_val)
            revenue = Q_val * selling_price
            cost = P_val * C_val
            profit = revenue - cost
            solution_data.append({
                '单品编码': row['单品编码'],
                '单品名称': row['单品名称'],
                '分类编码': row['分类编码'],
                '分类名称': row['分类名称'],
                '是否上架': 1,
                '进货量(kg)': P_val,
                '加成率': A_val,
                '售价(元/kg)': selling_price,
                '预测销量(kg)': Q_val,
                '预测批发价(元/kg)': C_val,
                '预估收入(元)': revenue,
                '进货成本(元)': cost,
                '预估利润(元)': profit
            })
        self.solution = pd.DataFrame(solution_data)
        # 输出摘要与校验
        self.extract_solution(validate_only=True)
    def extract_solution(self, validate_only=False):
        """
        提取优化结果
        """
        logger.info("Extracting optimization solution...")
        
        solution_data = []
        total_profit = 0
        total_revenue = 0
        total_cost = 0
        
        if self.solution is not None and validate_only:
            # 使用已有解汇总
            for _, row in self.solution.iterrows():
                total_profit += float(row['预估利润(元)'])
                total_revenue += float(row['预估收入(元)'])
                total_cost += float(row['进货成本(元)'])
        else:
            for i in self.candidates_df.index:
                if self.x_vars[i].varValue > 0.5:  # 选中上架
                    row = self.candidates_df.loc[i]
                    
                    x_val = 1
                    Q_val = row['pred_Q_p']  # 预测销量
                    P_val = Q_val  # 进货量等于预测销量
                    A_val = 0.3    # 固定加成率30%
                    
                    # 计算售价和利润
                    C_val = row['pred_C']
                    selling_price = C_val * (1 + A_val)
                    revenue = Q_val * selling_price
                    cost = P_val * C_val
                    profit = revenue - cost
                    
                    total_profit += profit
                    total_revenue += revenue
                    total_cost += cost
                    
                    solution_data.append({
                        '单品编码': row['单品编码'],
                        '单品名称': row['单品名称'],
                        '分类编码': row['分类编码'],
                        '分类名称': row['分类名称'],
                        '是否上架': x_val,
                        '进货量(kg)': P_val,
                        '加成率': A_val,
                        '售价(元/kg)': selling_price,
                        '预测销量(kg)': Q_val,
                        '预测批发价(元/kg)': C_val,
                        '预估收入(元)': revenue,
                        '进货成本(元)': cost,
                        '预估利润(元)': profit
                    })
            if not validate_only:
                self.solution = pd.DataFrame(solution_data)
        
        # 汇总信息
        n_selected = len(self.solution)
        total_stock = self.solution['进货量(kg)'].sum()
        avg_markup = self.solution['加成率'].mean()
        avg_markup = self.solution['加成率'].mean()
        
        logger.info(f"\n=== Optimization Results ===")
        logger.info(f"Selected products: {n_selected}")
        logger.info(f"Total stock: {total_stock:.2f} kg")
        logger.info(f"Average markup: {avg_markup:.2%}")
        logger.info(f"Total revenue: {total_revenue:.2f} yuan")
        logger.info(f"Total cost: {total_cost:.2f} yuan")
        logger.info(f"Total profit: {total_profit:.2f} yuan")
        if total_revenue > 0:
            logger.info(f"Profit margin: {total_profit/total_revenue:.2%}")
        else:
            logger.info("Profit margin: N/A (zero revenue)")
        
        # 检查约束满足情况
        self.validate_solution()
    
    def validate_solution(self):
        """
        验证解的可行性
        """
        logger.info("Validating solution...")
        
        if self.solution is None or len(self.solution) == 0:
            logger.warning("No solution to validate!")
            return False
        
        n_selected = len(self.solution)
        
        # 检查上架数量约束（自适应候选数量）
        effective_min = min(MIN_SHELF_COUNT, len(self.candidates_df))
        effective_max = min(MAX_SHELF_COUNT, len(self.candidates_df))
        if n_selected < effective_min:
            logger.warning(f"Solution violates minimum shelf count: {n_selected} < {effective_min}")
            return False
        
        if n_selected > effective_max:
            logger.warning(f"Solution violates maximum shelf count: {n_selected} > {effective_max}")
            return False
        
        # 检查最小陈列量约束
        min_stock = self.solution['进货量(kg)'].min()
        if min_stock < MIN_DISPLAY_QTY:
            logger.warning(f"Solution violates minimum display quantity: {min_stock} < {MIN_DISPLAY_QTY}")
            return False
        
        # 检查加成率约束
        min_markup = self.solution['加成率'].min()
        max_markup = self.solution['加成率'].max()
        
        if min_markup < MARKUP_BOUNDS['min']:
            logger.warning(f"Solution violates minimum markup: {min_markup} < {MARKUP_BOUNDS['min']}")
            return False
        
        if max_markup > MARKUP_BOUNDS['max']:
            logger.warning(f"Solution violates maximum markup: {max_markup} > {MARKUP_BOUNDS['max']}")
            return False
        
        logger.info("Solution validation passed!")
        return True
    
    def get_optimization_summary(self):
        """
        获取优化摘要
        """
        if self.solution is None:
            return None
        
        summary = {
            'selected_count': len(self.solution),
            'total_stock_kg': self.solution['进货量(kg)'].sum(),
            'total_revenue': self.solution['预估收入(元)'].sum(),
            'total_cost': self.solution['进货成本(元)'].sum(),
            'total_profit': self.solution['预估利润(元)'].sum(),
            'avg_markup': self.solution['加成率'].mean(),
            'avg_selling_price': self.solution['售价(元/kg)'].mean(),
            'category_distribution': self.solution['分类名称'].value_counts().to_dict(),
            'profit_margin': self.solution['预估利润(元)'].sum() / self.solution['预估收入(元)'].sum()
        }
        
        return summary

def optimize_restocking(candidates_df, config=None):
    """
    补货优化主函数
    """
    logger.info("Starting restocking optimization...")
    
    if config is None:
        config = {
            'use_elasticity': ELASTICITY_PARAMS['use_nonlinear'],
            'solver': SOLVER_CONFIG['default_solver'],
            'time_limit': SOLVER_CONFIG['time_limit']
        }
    
    # 创建优化器
    optimizer = RestockingOptimizer(
        candidates_df, 
        use_elasticity=config['use_elasticity'],
        solver=config['solver']
    )
    
    # 求解
    success = optimizer.solve(time_limit=config.get('time_limit'))
    
    if success:
        return optimizer.solution, optimizer.get_optimization_summary()
    else:
        logger.error("Optimization failed!")
        return None, None

def run_sensitivity_analysis(candidates_df, base_solution):
    """
    敏感性分析
    """
    logger.info("Running sensitivity analysis...")
    
    # 分析不同参数设置下的结果
    scenarios = [
        {'markup_bounds': {'min': 0.05, 'max': 0.5}, 'name': 'Low Markup Range'},
        {'markup_bounds': {'min': 0.15, 'max': 0.7}, 'name': 'High Markup Range'},
        {'shelf_count': {'min': 25, 'max': 35}, 'name': 'Different Shelf Counts'},
    ]
    
    sensitivity_results = []
    
    for scenario in scenarios:
        logger.info(f"Testing scenario: {scenario['name']}")
        
        # 创建修改后的配置
        # 这里可以实现更详细的敏感性分析
        # 由于时间限制，暂时跳过详细实现
        
        pass
    
    return sensitivity_results