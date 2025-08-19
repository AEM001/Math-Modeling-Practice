import os

RANDOM_STATE = 2025
MIN_DISPLAY_QTY = 2.5  # 最小陈列量 (kg)
N_CANDIDATES_MAX = 40  # 最大候选单品数量

MIN_SHELF_COUNT = 27  # 最小上架数量
MAX_SHELF_COUNT = 33  # 最大上架数量

RF_PARAMS = {
    'n_estimators': 50,
    'max_depth': 8,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_STATE
}

MARKUP_BOUNDS = {
    'min': 0.1,  # 最小加成率
    'max': 0.6   # 最大加成率
}

ELASTICITY_PARAMS = {
    'use_nonlinear': True,
    'alpha': 1.0,  # 弹性函数参数
    'beta': -0.5,  # 价格弹性系数
    'demand_tolerance': 0.2  # 需求区间容忍度 (±20%)
}

FORECAST_PARAMS = {
    'moving_window_days': [7, 14, 28],  # 移动窗口天数
    'use_weekend_correction': True,  # 是否使用周末修正
    'weekend_factor': 1.2  # 周六销量修正系数
}

SOLVER_CONFIG = {
    'default_solver': 'CBC',  # 默认求解器
    'time_limit': 300,  # 求解时间限制(秒)
    'gap': 0.01  # MIP Gap
}

DATA_PATHS = {
    'single_product_summary': 'data/过滤后单品级汇总表.csv',
    'sellable_timeseries': 'data/可售品种时间序列表_20230624-30.csv',
    'weekly_stats': 'data/品种周统计表_20230624-30.csv'
}

OUTPUT_PATHS = {
    'results_dir': 'outputs/results',
    'figs_dir': 'outputs/figs',
    'plan_csv': 'outputs/results/plan_2023-07-01.csv'
}

TARGET_DATE = '2023-07-01'

FEATURE_COLUMNS = {
    'single_product_cols': [
        '销售日期', '单品编码', '单品名称', '分类编码', '分类名称',
        '总销量(千克)', '销售单价(元/千克)', '批发价格(元/千克)', 
        '成本加成率', '损耗率(%)'
    ],
    'key_features': [
        'day_of_week', 'is_weekend', 'month', 'week_num',
        'sales_ma_7d', 'sales_ma_14d', 'sales_ma_28d',
        'sales_std_7d', 'price_ma_7d', 'markup_ma_7d'
    ]
}

QUALITY_THRESHOLDS = {
    'min_r2_score': 0.5,  # 最小R²分数阈值
    'min_days_history': 7  # 最少历史天数
}