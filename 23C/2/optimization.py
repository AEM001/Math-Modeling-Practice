import pandas as pd
import numpy as np
import json
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load the original data for wholesale price modeling
df_original = pd.read_csv('单品级每日汇总表.csv', encoding='utf-8-sig')  # Handle BOM
df_original['销售日期'] = pd.to_datetime(df_original['销售日期'])

# Load trained demand models
with open('demand_models.json', 'r', encoding='utf-8') as f:
    demand_models = json.load(f)

print(f"Loaded {len(demand_models)} demand models")

# Filter for items with valid demand models
df_model_items = df_original[
    df_original['单品编码'].isin([int(k) for k in demand_models.keys()])
].copy()

print(f"Original data has {len(df_model_items)} records for modeled items")

# Data preparation - filter normal sales data
df_normal = df_model_items[df_model_items['打折销量(千克)'] == 0].copy()
df_normal = df_normal[df_normal['正常销量(千克)'] > 0]
df_normal = df_normal[df_normal['正常销售单价(元/千克)'] > 0]
df_normal = df_normal[df_normal['批发价格(元/千克)'] > 0]

print(f"Filtered normal sales data: {len(df_normal)} records")

# --- Stage 3.1: 批发价格预测 ---
def forecast_wholesale_price(item_df, forecast_days=7):
    """Forecast wholesale prices using exponential smoothing"""
    item_df = item_df.sort_values('销售日期')
    
    # Extract price series
    prices = item_df['批发价格(元/千克)'].values
    
    if len(prices) < 5:
        # Not enough data, use simple average
        forecast = np.mean(prices)
    else:
        try:
            # Try exponential smoothing
            model = ExponentialSmoothing(prices, trend='add', seasonal=None)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=forecast_days)
        except:
            # Fallback to simple average
            forecast = np.mean(prices) * np.ones(forecast_days)
    
    return np.array(forecast)

print("Predicting wholesale prices for next 7 days...")

# Create forecast dates
target_start_date = datetime(2023, 7, 1)
target_dates = [target_start_date + timedelta(days=i) for i in range(7)]

# Forecast wholesale prices for each item
wholesale_forecasts = {}

for item_code in demand_models.keys():
    item_data = df_normal[df_normal['单品编码'] == int(item_code)].copy()
    
    if len(item_data) >= 3:
        forecasts = forecast_wholesale_price(item_data)
    else:
        # Use overall average for items with minimal data
        forecasts = np.array([df_normal['批发价格(元/千克)'].mean() * 7])
    
    wholesale_forecasts[item_code] = forecasts

# Save forecasts
with open('wholesale_forecasts.json', 'w', encoding='utf-8') as f:
    json.dump(
        {k: list(v) if isinstance(v, np.ndarray) else [float(v)] * 7 
         for k, v in wholesale_forecasts.items()}, 
        f, ensure_ascii=False, indent=2
    )

print("Wholesale price forecasts saved to wholesale_forecasts.json")

# --- Stage 3.2: 利润优化模型 ---
def optimize_single_item_daily(alpha, beta, cost, loss_rate):
    """
    Optimize pricing for a single item on a single day
    
    Parameters:
    - alpha, beta: demand model parameters
    - cost: wholesale price for the day
    - loss_rate: loss rate as decimal
    
    Returns:
    - optimal_price, optimal_quantity, optimal_replenishment, max_profit
    """
    # Model constraints
    min_price = cost * 1.001  # Slightly above cost
    max_price = cost * 2.0    # Maximum markup constraint
    
    if min_price >= max_price:
        min_price = cost * 1.001
        max_price = cost * 2.5
    
    # Define profit function
    def calculate_profit(P):
        """Calculate profit given price P"""
        if P <= 0:
            return -np.inf
        
        ln_Q = alpha + beta * np.log(P)
        
        # Prevent extreme values
        if ln_Q > 50 or ln_Q < -50:  # Equivalent to Q > ~5e21 or Q < ~2e-22
            return -np.inf
            
        Q = np.exp(ln_Q)  # Predicted demand
        
        # Sanity check on demand
        if Q > 1e6 or Q <= 0 or not np.isfinite(Q):  # More than 1 million kg per day is unrealistic
            return -np.inf
        
        # Replenishment quantity accounting for loss
        if loss_rate >= 1:
            return -np.inf
        R = Q / (1 - loss_rate)
        
        # Profit calculation
        profit = Q * P - R * cost
        
        if not np.isfinite(profit):
            return -np.inf
            
        return profit
    
    # Numerical optimization via grid search
    price_range = np.linspace(min_price, max_price, 100)
    profits = [calculate_profit(p) for p in price_range]
    
    # Find optimal price
    max_profit_index = np.argmax(profits)
    optimal_price = price_range[max_profit_index]
    max_profit = profits[max_profit_index]
    
    # Calculate corresponding optimal values
    optimal_q = np.exp(alpha + beta * np.log(optimal_price))
    optimal_r = optimal_q / (1 - loss_rate)
    
    return {
        'optimal_price': float(optimal_price),
        'optimal_quantity': float(optimal_q),
        'optimal_replenishment': float(optimal_r),
        'max_profit': float(max_profit),
        'cost': float(cost),
        'loss_rate': float(loss_rate),
        'margin': float((optimal_price - cost) / optimal_price)
    }

# --- Stage 3.3: Solve optimization for next 7 days ---
print("Optimizing pricing and replenishment strategies...")

optimization_results = []

for item_code in demand_models.keys():
    if item_code not in wholesale_forecasts:
        continue
    
    model = demand_models[item_code]
    
    # Skip items with extreme beta values that cause numerical instability
    if model['beta'] < -4 or model['beta'] > -0.1:
        print(f"Skipping item {item_code} ({model['item_name']}) due to extreme beta: {model['beta']}")
        continue
        
    forecasts = wholesale_forecasts[item_code]
    
    if len(forecasts) < 7:
        forecasts = forecasts.tolist() if isinstance(forecasts, np.ndarray) else [float(forecasts)] * 7
    
    for day_idx in range(7):
        date = target_dates[day_idx]
        if isinstance(forecasts, list) and len(forecasts) >= day_idx + 1:
            wholesale_price = forecasts[day_idx]
        elif isinstance(forecasts, np.ndarray) and len(forecasts) >= day_idx + 1:
            wholesale_price = forecasts[day_idx]
        else:
            # Use the last available price or average
            if isinstance(forecasts, (list, np.ndarray)) and len(forecasts) > 0:
                wholesale_price = forecasts[-1] if isinstance(forecasts, list) else forecasts[-1]
            else:
                wholesale_price = forecasts if not isinstance(forecasts, (list, np.ndarray)) else np.mean(forecasts)
        
        # Optimize daily strategy
        result = optimize_single_item_daily(
            alpha=model['alpha'],
            beta=model['beta'],
            cost=wholesale_price,
            loss_rate=model['loss_rate']
        )
        
        optimization_results.append({
            '单品编码': item_code,
            '单品名称': model['item_name'],
            '分类名称': model['category'],
            '日期': date.strftime('%Y-%m-%d'),
            '日序号': day_idx + 1,
            '批发价格(元/千克)': result['cost'],
            '最优售价(元/千克)': result['optimal_price'],
            '最优销量(千克)': result['optimal_quantity'],
            '最优补货量(千克)': result['optimal_replenishment'],
            '最大利润(元)': result['max_profit'],
            '加成率': result['margin'],
            '损耗率': result['loss_rate']
        })

# Convert to DataFrame
optimization_df = pd.DataFrame(optimization_results)
optimization_df.to_csv('daily_optimization_results.csv', index=False)

print(f"Optimization completed for {len(optimization_df)} item-days")
print("Daily optimization results saved to daily_optimization_results.csv")

# Display summary statistics
print("\n=== Optimization Results Summary ===")

summary_stats = optimization_df.groupby('日序号').agg({
    '最大利润(元)': 'sum',
    '最优销量(千克)': 'sum',
    '最优补货量(千克)': 'sum',
    '加成率': 'mean'
}).round(2)

print(summary_stats)

# Save category-level analysis for next stage - fix weighted average calculation
def calculate_weighted_price(group):
    weights = group['最优销量(千克)']
    prices = group['最优售价(元/千克)']
    # Handle edge cases
    if weights.sum() == 0 or len(prices) == 0:
        return prices.mean() if len(prices) > 0 else 0
    return (weights * prices).sum() / weights.sum()

category_summary = optimization_df.groupby(['分类名称', '日期']).agg({
    '最优补货量(千克)': 'sum',
    '最优销量(千克)': 'sum',
    '最大利润(元)': 'sum',
    '批发价格(元/千克)': 'mean'
}).reset_index()

# Calculate weighted average pricing separately
weighted_prices = optimization_df.groupby(['分类名称', '日期']).apply(calculate_weighted_price).reset_index()
weighted_prices.columns = ['分类名称', '日期', '品类别加权平均售价(元/千克)']

category_summary = category_summary.merge(weighted_prices, on=['分类名称', '日期'])

category_summary = category_summary.rename(columns={
    '最优补货量(千克)': '品类补货总量(千克)',
    '最优销量(千克)': '品类销量总量(千克)',
    '最大利润(元)': '品类总利润(元)',
    '批发价格(元/千克)': '品类平均批发价(元/千克)',
    '最优售价(元/千克)': '品类别加权平均售价(元/千克)'
})

category_summary.to_csv('weekly_category_strategy.csv', index=False)
print(f"Category-level strategy saved to weekly_category_strategy.csv")

print("\n=== Optimization Complete ===")
print(f"Successfully optimized pricing and replenishment for {len(demand_models)} items over 7 days")
print(f"Total projected profit: ¥{optimization_df['最大利润(元)'].sum():.2f}")
print(f"Total replenishment needed: {optimization_df['最优补货量(千克)'].sum():.1f} kg")