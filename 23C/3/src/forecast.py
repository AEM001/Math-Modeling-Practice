"""
Sales and wholesale price forecasting models.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
from config import RF_PARAMS, TARGET_DATE, FORECAST_PARAMS, QUALITY_THRESHOLDS
from features import build_feature_matrix, create_prediction_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecaster:
    """
    销量预测器
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(**RF_PARAMS)
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        准备训练特征
        """
        # 选择特征列
        feature_cols = [
            # 时间特征
            'day_of_week', 'is_weekend', 'is_saturday', 'month', 'week_num',
            
            # 移动平均特征
            'sales_ma_7d', 'sales_ma_14d', 'sales_ma_28d',
            'sales_std_7d', 'sales_std_14d',
            'price_ma_7d', 'wholesale_price_ma_7d',
            'markup_ma_7d',
            
            # 滞后特征
            'sales_lag_1d', 'sales_lag_2d', 'sales_lag_7d',
            'price_lag_1d', 'wholesale_price_lag_1d',
            
            # 周几效应
            'dow_sales_mean', 'dow_sales_std',
            'saturday_factor',
            
            # 品类特征
            'category_sales_mean', 'sales_vs_category_mean',
            
            # 价格弹性特征
            'price_change', 'markup_change'
        ]
        
        # 过滤存在的特征列
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        logger.info(f"Selected {len(available_features)} features for sales forecasting")
        
        return df[available_features].fillna(0)
    
    def train(self, df, target_col='总销量(千克)'):
        """
        训练销量预测模型
        """
        logger.info("Training sales forecasting model...")
        
        # 准备特征
        X = self.prepare_features(df)
        y = df[target_col]
        
        # 过滤有效样本（销量>0）
        valid_mask = (y > 0) & (X.sum(axis=1) != 0)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        logger.info(f"Training on {len(X_valid)} valid samples")
        
        # 训练模型
        self.model.fit(X_valid, y_valid)
        self.is_trained = True
        
        # 交叉验证评估
        cv_scores = cross_val_score(self.model, X_valid, y_valid, cv=5, 
                                   scoring='r2', n_jobs=-1)
        
        # 训练集预测评估
        y_pred_train = self.model.predict(X_valid)
        train_r2 = r2_score(y_valid, y_pred_train)
        train_mae = mean_absolute_error(y_valid, y_pred_train)
        
        logger.info(f"Sales model training completed:")
        logger.info(f"  Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"  Training R²: {train_r2:.4f}")
        logger.info(f"  Training MAE: {train_mae:.4f}")
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'train_r2': train_r2,
            'train_mae': train_mae
        }
    
    def predict(self, df):
        """
        预测销量
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        # 确保预测值非负
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(self):
        """
        获取特征重要性
        """
        if not self.is_trained:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class WholesalePriceForecaster:
    """
    批发价预测器
    """
    
    def __init__(self, method='moving_average'):
        self.method = method  # 'moving_average' or 'model'
        self.model = None
        self.is_trained = False
        
    def predict_moving_average(self, df, window=7):
        """
        移动平均法预测批发价
        """
        logger.info(f"Predicting wholesale prices using {window}-day moving average")
        
        predictions = []
        products = df['单品编码'].unique()
        
        for product_code in products:
            product_df = df[df['单品编码'] == product_code].sort_values('销售日期')
            
            # 计算最近window天的移动平均
            recent_prices = product_df['批发价格(元/千克)'].tail(window)
            
            if len(recent_prices) > 0:
                # 周六修正
                if FORECAST_PARAMS['use_weekend_correction']:
                    target_dt = pd.to_datetime(TARGET_DATE)
                    if target_dt.dayofweek == 5:  # 周六
                        avg_price = recent_prices.mean() * 1.02  # 小幅上调
                    else:
                        avg_price = recent_prices.mean()
                else:
                    avg_price = recent_prices.mean()
                
                predictions.append({
                    '单品编码': product_code,
                    'pred_wholesale_price': avg_price
                })
        
        return pd.DataFrame(predictions)
    
    def predict_model_based(self, df_train, df_pred):
        """
        基于模型的批发价预测
        """
        logger.info("Training model-based wholesale price predictor")
        
        # 准备特征
        feature_cols = [
            'day_of_week', 'is_weekend', 'month',
            'wholesale_price_ma_7d', 'wholesale_price_ma_14d',
            'wholesale_price_lag_1d', 'wholesale_price_lag_7d',
            'price_change', 'wholesale_price_change'
        ]
        
        available_features = [col for col in feature_cols if col in df_train.columns]
        
        X_train = df_train[available_features].fillna(0)
        y_train = df_train['批发价格(元/千克)']
        
        # 过滤有效样本
        valid_mask = (y_train > 0) & (X_train.sum(axis=1) != 0)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # 训练模型
        self.model = RandomForestRegressor(n_estimators=50, random_state=RF_PARAMS['random_state'])
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 预测
        X_pred = df_pred[available_features].fillna(0)
        predictions = self.model.predict(X_pred)
        
        return pd.DataFrame({
            '单品编码': df_pred['单品编码'],
            'pred_wholesale_price': predictions
        })
    
    def predict(self, df_train, df_pred=None):
        """
        预测批发价
        """
        if self.method == 'moving_average':
            return self.predict_moving_average(df_train)
        elif self.method == 'model' and df_pred is not None:
            return self.predict_model_based(df_train, df_pred)
        else:
            raise ValueError("Invalid method or missing prediction data")

def forecast_sales_and_prices(df, target_date=TARGET_DATE):
    """
    预测销量和批发价的主函数
    """
    logger.info(f"Starting sales and price forecasting for {target_date}")
    
    # 1. 构建特征矩阵
    logger.info("Building feature matrix...")
    df_features = build_feature_matrix(df)
    
    # 2. 准备训练数据（排除预测日期）
    train_df = df_features[df_features['销售日期'] < target_date].copy()
    
    if len(train_df) == 0:
        raise ValueError("No training data available")
    
    # 3. 创建预测特征
    pred_df = create_prediction_features(train_df, target_date)
    
    # 4. 训练销量预测模型
    sales_forecaster = SalesForecaster()
    train_metrics = sales_forecaster.train(train_df)
    
    # 5. 预测销量
    pred_sales = sales_forecaster.predict(pred_df)
    
    # 6. 预测批发价
    wholesale_forecaster = WholesalePriceForecaster(method='moving_average')
    wholesale_pred_df = wholesale_forecaster.predict(train_df)
    
    # 7. 整合预测结果
    results = []
    
    for idx, row in pred_df.iterrows():
        product_code = row['单品编码']
        
        # 获取批发价预测
        wholesale_row = wholesale_pred_df[
            wholesale_pred_df['单品编码'] == product_code
        ]
        
        if len(wholesale_row) > 0:
            pred_wholesale_price = wholesale_row['pred_wholesale_price'].iloc[0]
        else:
            # 回退到最近的批发价
            recent_price = train_df[
                train_df['单品编码'] == product_code
            ]['批发价格(元/千克)'].tail(1)
            pred_wholesale_price = recent_price.iloc[0] if len(recent_price) > 0 else 10.0
        
        # 获取模型质量评分
        model_quality = train_metrics['cv_r2_mean']
        
        results.append({
            '单品编码': product_code,
            '单品名称': row['单品名称'],
            '分类编码': row['分类编码'],
            '分类名称': row['分类名称'],
            'pred_Q_p': pred_sales[idx],
            'pred_C': pred_wholesale_price,
            'model_quality': model_quality,
            'is_saturday': row['is_saturday']
        })
    
    results_df = pd.DataFrame(results)
    
    # 8. 质量评估和过滤
    logger.info("Evaluating forecast quality...")
    
    # 计算每个产品的历史平均销量作为基线对比
    baseline_sales = train_df.groupby('单品编码')['总销量(千克)'].mean().to_dict()
    results_df['baseline_sales'] = results_df['单品编码'].map(baseline_sales)
    
    # 计算预测vs基线的比值
    results_df['pred_vs_baseline'] = (
        results_df['pred_Q_p'] / (results_df['baseline_sales'] + 1e-8)
    )
    
    logger.info(f"Forecasting completed for {len(results_df)} products")
    logger.info(f"Average predicted sales: {results_df['pred_Q_p'].mean():.2f} kg")
    logger.info(f"Average predicted wholesale price: {results_df['pred_C'].mean():.2f} yuan/kg")
    
    return results_df, sales_forecaster

def quick_forecast_baseline(df, target_date=TARGET_DATE):
    """
    快速基线预测方法（移动平均+周六修正）
    """
    logger.info("Running quick baseline forecast...")
    
    target_dt = pd.to_datetime(target_date)
    is_saturday = target_dt.dayofweek == 5
    
    results = []
    products = df['单品编码'].unique()
    
    for product_code in products:
        product_df = df[df['单品编码'] == product_code].sort_values('销售日期')
        
        if len(product_df) < 7:
            continue
            
        # 获取最近7天平均销量
        recent_sales = product_df['总销量(千克)'].tail(7).mean()
        recent_price = product_df['批发价格(元/千克)'].tail(7).mean()
        
        # 周六修正
        if is_saturday and FORECAST_PARAMS['use_weekend_correction']:
            pred_sales = recent_sales * FORECAST_PARAMS['weekend_factor']
        else:
            pred_sales = recent_sales
        
        # 获取产品基本信息
        latest_info = product_df.iloc[-1]
        
        results.append({
            '单品编码': product_code,
            '单品名称': latest_info['单品名称'],
            '分类编码': latest_info['分类编码'],
            '分类名称': latest_info['分类名称'],
            'pred_Q_p': pred_sales,
            'pred_C': recent_price,
            'model_quality': 0.5,  # 基线模型质量
            'is_saturday': int(is_saturday)
        })
    
    baseline_df = pd.DataFrame(results)
    
    logger.info(f"Baseline forecast completed for {len(baseline_df)} products")
    
    return baseline_df