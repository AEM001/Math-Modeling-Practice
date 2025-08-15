import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm
import json

# Load training data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Convert date columns to datetime
train_df['销售日期'] = pd.to_datetime(train_df['销售日期'])
test_df['销售日期'] = pd.to_datetime(test_df['销售日期'])

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Unique items: {train_df['单品编码'].nunique()}")

# Filter data for demand modeling
def filter_data_for_modeling(df):
    """Filter data suitable for log-log demand modeling"""
    # Ensure positive values for log transformation
    mask = (
        (df['正常销量(千克)'] > 0) & 
        (df['正常销售单价(元/千克)'] > 0) &
        (df['批发价格(元/千克)'] > 0)
    )
    return df[mask].copy()

train_modeling = filter_data_for_modeling(train_df)
test_modeling = filter_data_for_modeling(test_df)

print(f"Training samples after filtering: {len(train_modeling)}")
print(f"Test samples after filtering: {len(test_modeling)}")

# Implement双对数需求模型: ln(Q) = α + β * ln(P)
def fit_demand_model(group_df):
    """Fit log-log demand model for a single item"""
    if len(group_df) < 5:  # Minimum samples required
        return None
    
    # Take log transformations
    Q = group_df['正常销量(千克)'].values
    P = group_df['正常销售单价(元/千克)'].values
    
    ln_Q = np.log(Q)
    ln_P = np.log(P)
    
    # Add constant for intercept
    X = sm.add_constant(ln_P)
    y = ln_Q
    
    try:
        model = sm.OLS(y, X).fit()
        
        # Extract parameters
        alpha = model.params[0]
        beta = model.params[1]
        
        # Statistical tests
        r_squared = model.rsquared
        f_stat = model.fvalue
        f_pvalue = model.f_pvalue
        t_beta = model.tvalues[1]
        p_beta = model.pvalues[1]
        
        # Check significance
        sig_alpha = model.pvalues[0] < 0.05
        sig_beta = p_beta < 0.05
        
        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            't_beta': t_beta,
            'p_beta': p_beta,
            'significant': sig_beta,
            'valid': (beta < 0) and (r_squared > 0.1),  # Elasticity should be negative
            'samples': len(group_df),
            'model': model
        }
    except Exception as e:
        print(f"Model fitting error: {e}")
        return None

# Train individual models for each item
print("Training demand models for each单品...")

models = {}
model_summary = []

for item_code in train_modeling['单品编码'].unique():
    item_data = train_modeling[train_modeling['单品编码'] == item_code]
    
    if len(item_data) >= 5:
        model_result = fit_demand_model(item_data)
        
        if model_result and model_result['valid']:
            models[item_code] = model_result
            item_name = item_data['单品名称'].iloc[0]
            category = item_data['分类名称'].iloc[0]
            
            model_summary.append({
                '单品编码': item_code,
                '单品名称': item_name,
                '分类名称': category,
                'alpha': model_result['alpha'],
                'beta': model_result['beta'],
                'r_squared': model_result['r_squared'],
                'significant': model_result['significant'],
                'samples': model_result['samples']
            })

print(f"Successfully trained {len(models)} valid models")

# Create model summary DataFrame
model_summary_df = pd.DataFrame(model_summary)
model_summary_df.to_csv('demand_model_results.csv', index=False)
print(f"Model summary saved to demand_model_results.csv")

# Model validation on test set
def predict_demand(row, model_params):
    """Predict demand using the log-log model"""
    if pd.isna(row['正常销售单价(元/千克)']) or row['正常销售单价(元/千克)'] <= 0:
        return np.nan
    
    P = row['正常销售单价(元/千克)']
    alpha = model_params['alpha']
    beta = model_params['beta']
    
    ln_Q_pred = alpha + beta * np.log(P)
    Q_pred = np.exp(ln_Q_pred)
    
    return Q_pred

print("Evaluating model performance on test set...")

validation_results = []

for item_code in models.keys():
    if item_code in test_modeling['单品编码'].values:
        # Test data for this item
        test_item = test_modeling[test_modeling['单品编码'] == item_code]
        
        if len(test_item) > 0:
            model_params = models[item_code]
            
            # Predict demand
            y_true = test_item['正常销量(千克)'].values
            y_pred = [predict_demand(row, model_params) for _, row in test_item.iterrows()]
            
            # Remove NaN predictions
            valid_mask = ~np.isnan(y_pred)
            if np.sum(valid_mask) > 0:
                y_true_valid = y_true[valid_mask]
                y_pred_valid = np.array(y_pred)[valid_mask]
                
                if len(y_true_valid) > 0:
                    # Calculate metrics
                    mse = mean_squared_error(y_true_valid, y_pred_valid)
                    rmse = np.sqrt(mse)
                    mape = mean_absolute_percentage_error(y_true_valid, y_pred_valid)
                    
                    validation_results.append({
                        '单品编码': item_code,
                        'test_samples': len(y_true_valid),
                        'rmse': rmse,
                        'mape': mape,
                        'y_true_mean': np.mean(y_true_valid),
                        'y_pred_mean': np.mean(y_pred_valid)
                    })

# Save validation results
validation_df = pd.DataFrame(validation_results)
validation_df.to_csv('validation_results.csv', index=False)

print(f"Model validation results saved to validation_results.csv")

# Display summary statistics
if len(model_summary_df) > 0:
    print("\n=== Model Training Summary ===")
    print(f"Total items modeled: {len(model_summary_df)}")
    print(f"Significant models (β<0): {len(model_summary_df[model_summary_df['significant']])}")
    print(f"Average R²: {model_summary_df['r_squared'].mean():.3f}")
    print(f"Beta range: [{model_summary_df['beta'].min():.3f}, {model_summary_df['beta'].max():.3f}]")

if len(validation_df) > 0:
    print("\n=== Validation Performance ===")
    print(f"Items validated: {len(validation_df)}")
    print(f"Average RMSE: {validation_df['rmse'].mean():.3f}")
    print(f"Average MAPE: {validation_df['mape'].mean():.3f}")

# Save models as JSON for next stage
model_params = {}
for item_code, model_data in models.items():
    # Find item details
    item_row = train_modeling[train_modeling['单品编码'] == item_code].iloc[0]
    model_params[str(item_code)] = {
        'alpha': model_data['alpha'],
        'beta': model_data['beta'],
        'r_squared': model_data['r_squared'],
        'item_name': item_row['单品名称'],
        'category': item_row['分类名称'],
        'loss_rate': float(item_row['损耗率(%)']) / 100
    }

with open('demand_models.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=2)

print("\nDemand modeling completed!")
print(f"- Saved individual demand models: demand_models.json")
print(f"- Saved model training summary: demand_model_results.csv")
print(f"- Saved validation results: validation_results.csv")