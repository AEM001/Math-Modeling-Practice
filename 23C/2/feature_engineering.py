# -*- coding: utf-8 -*-
"""
特征工程模块
基于EDA结果构建增强特征集
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, data_path='clean_data_full.csv'):
        """初始化特征工程器"""
        self.data_path = data_path
        self.data = None
        self.feature_data = None
        
    def load_data(self):
        """加载清洗后的数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)
        self.data['销售日期'] = pd.to_datetime(self.data['销售日期'])
        
        # 基础对数变量
        self.data['ln_price'] = np.log(self.data['正常销售单价(元/千克)'])
        self.data['ln_quantity'] = np.log(self.data['正常销量(千克)'])
        self.data['ln_wholesale'] = np.log(self.data['批发价格(元/千克)'])
        
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        return self
        
    def create_time_features(self):
        """创建时间特征"""
        print("创建时间特征...")
        
        # 基础时间特征
        self.data['year'] = self.data['销售日期'].dt.year
        self.data['month'] = self.data['销售日期'].dt.month
        self.data['day'] = self.data['销售日期'].dt.day
        self.data['weekday'] = self.data['销售日期'].dt.dayofweek
        self.data['is_weekend'] = (self.data['weekday'] >= 5).astype(int)
        
        # 周几哑变量（基于EDA发现的显著效应）
        weekday_dummies = pd.get_dummies(self.data['weekday'], prefix='weekday')
        self.data = pd.concat([self.data, weekday_dummies], axis=1)
        
        # 月份哑变量（季节性）
        month_dummies = pd.get_dummies(self.data['month'], prefix='month')
        self.data = pd.concat([self.data, month_dummies], axis=1)
        
        # 时间趋势
        self.data['days_since_start'] = (self.data['销售日期'] - self.data['销售日期'].min()).dt.days
        self.data['time_trend'] = self.data['days_since_start'] / self.data['days_since_start'].max()
        
        print("时间特征创建完成")
        
    def create_price_features(self):
        """创建价格相关特征"""
        print("创建价格特征...")
        
        # 相对价格特征
        # 1. 相对于品类日均价
        category_daily_price = self.data.groupby(['分类名称', '销售日期'])['正常销售单价(元/千克)'].mean().reset_index()
        category_daily_price.columns = ['分类名称', '销售日期', 'category_daily_avg_price']
        self.data = self.data.merge(category_daily_price, on=['分类名称', '销售日期'], how='left')
        self.data['relative_price_to_category'] = self.data['正常销售单价(元/千克)'] / self.data['category_daily_avg_price']
        self.data['ln_relative_price_to_category'] = np.log(self.data['relative_price_to_category'])
        
        # 2. 相对于批发价（加成倍数）
        self.data['markup_ratio'] = self.data['正常销售单价(元/千克)'] / self.data['批发价格(元/千克)']
        self.data['ln_markup_ratio'] = np.log(self.data['markup_ratio'])
        
        # 3. 品类价格指数（以第一个观测日为基准）
        for category in self.data['分类名称'].unique():
            cat_mask = self.data['分类名称'] == category
            cat_data = self.data[cat_mask].sort_values('销售日期')
            
            if len(cat_data) > 0:
                base_price = cat_data['category_daily_avg_price'].iloc[0]
                self.data.loc[cat_mask, 'category_price_index'] = (
                    self.data.loc[cat_mask, 'category_daily_avg_price'] / base_price
                )
        
        print("价格特征创建完成")
        
    def create_rolling_features(self):
        """创建滚动统计特征"""
        print("创建滚动特征...")
        
        # 按单品创建滚动特征
        rolling_features = []
        
        for item_code in self.data['单品编码'].unique():
            item_data = self.data[self.data['单品编码'] == item_code].sort_values('销售日期').copy()
            
            if len(item_data) < 5:  # 样本太少跳过
                continue
                
            # 7天滚动均值和标准差
            item_data['price_rolling_7d_mean'] = item_data['正常销售单价(元/千克)'].rolling(window=7, min_periods=1).mean()
            item_data['price_rolling_7d_std'] = item_data['正常销售单价(元/千克)'].rolling(window=7, min_periods=1).std()
            item_data['quantity_rolling_7d_mean'] = item_data['正常销量(千克)'].rolling(window=7, min_periods=1).mean()
            item_data['quantity_rolling_7d_std'] = item_data['正常销量(千克)'].rolling(window=7, min_periods=1).std()
            
            # 14天滚动均值
            item_data['price_rolling_14d_mean'] = item_data['正常销售单价(元/千克)'].rolling(window=14, min_periods=1).mean()
            item_data['quantity_rolling_14d_mean'] = item_data['正常销量(千克)'].rolling(window=14, min_periods=1).mean()
            
            # 价格波动率（变异系数）
            item_data['price_volatility_7d'] = item_data['price_rolling_7d_std'] / item_data['price_rolling_7d_mean']
            item_data['quantity_volatility_7d'] = item_data['quantity_rolling_7d_std'] / item_data['quantity_rolling_7d_mean']
            
            # 相对于滚动均值的偏离
            item_data['price_deviation_from_7d_mean'] = (
                item_data['正常销售单价(元/千克)'] - item_data['price_rolling_7d_mean']
            ) / item_data['price_rolling_7d_mean']
            
            rolling_features.append(item_data)
        
        # 合并滚动特征
        if rolling_features:
            rolling_df = pd.concat(rolling_features, ignore_index=True)
            
            # 合并到主数据
            rolling_cols = ['单品编码', '销售日期', 'price_rolling_7d_mean', 'price_rolling_7d_std',
                          'quantity_rolling_7d_mean', 'quantity_rolling_7d_std', 'price_rolling_14d_mean',
                          'quantity_rolling_14d_mean', 'price_volatility_7d', 'quantity_volatility_7d',
                          'price_deviation_from_7d_mean']
            
            self.data = self.data.merge(
                rolling_df[rolling_cols], 
                on=['单品编码', '销售日期'], 
                how='left'
            )
        
        print("滚动特征创建完成")
        
    def create_lag_features(self):
        """创建滞后特征"""
        print("创建滞后特征...")
        
        lag_features = []
        
        for item_code in self.data['单品编码'].unique():
            item_data = self.data[self.data['单品编码'] == item_code].sort_values('销售日期').copy()
            
            if len(item_data) < 3:
                continue
                
            # 1天滞后
            item_data['price_lag1'] = item_data['正常销售单价(元/千克)'].shift(1)
            item_data['quantity_lag1'] = item_data['正常销量(千克)'].shift(1)
            item_data['ln_price_lag1'] = item_data['ln_price'].shift(1)
            item_data['ln_quantity_lag1'] = item_data['ln_quantity'].shift(1)
            
            # 7天滞后（周同比）
            item_data['price_lag7'] = item_data['正常销售单价(元/千克)'].shift(7)
            item_data['quantity_lag7'] = item_data['正常销量(千克)'].shift(7)
            
            # 价格变化率
            item_data['price_change_1d'] = (
                item_data['正常销售单价(元/千克)'] - item_data['price_lag1']
            ) / item_data['price_lag1']
            
            item_data['price_change_7d'] = (
                item_data['正常销售单价(元/千克)'] - item_data['price_lag7']
            ) / item_data['price_lag7']
            
            lag_features.append(item_data)
        
        if lag_features:
            lag_df = pd.concat(lag_features, ignore_index=True)
            
            lag_cols = ['单品编码', '销售日期', 'price_lag1', 'quantity_lag1', 'ln_price_lag1',
                       'ln_quantity_lag1', 'price_lag7', 'quantity_lag7', 'price_change_1d', 'price_change_7d']
            
            self.data = self.data.merge(
                lag_df[lag_cols],
                on=['单品编码', '销售日期'],
                how='left'
            )
        
        print("滞后特征创建完成")
        
    def create_interaction_features(self):
        """创建交互特征"""
        print("创建交互特征...")
        
        # 价格与时间的交互
        self.data['ln_price_x_time_trend'] = self.data['ln_price'] * self.data['time_trend']
        self.data['ln_price_x_weekend'] = self.data['ln_price'] * self.data['is_weekend']
        
        # 价格与季节的交互
        for month in range(1, 13):
            if f'month_{month}' in self.data.columns:
                self.data[f'ln_price_x_month_{month}'] = self.data['ln_price'] * self.data[f'month_{month}']
        
        # 相对价格与周几的交互
        for weekday in range(7):
            if f'weekday_{weekday}' in self.data.columns:
                self.data[f'relative_price_x_weekday_{weekday}'] = (
                    self.data['ln_relative_price_to_category'] * self.data[f'weekday_{weekday}']
                )
        
        print("交互特征创建完成")
        
    def create_category_features(self):
        """创建品类级特征"""
        print("创建品类特征...")
        
        # 品类竞争强度（同品类单品数量）
        category_competition = self.data.groupby(['分类名称', '销售日期'])['单品编码'].nunique().reset_index()
        category_competition.columns = ['分类名称', '销售日期', 'category_competition_intensity']
        self.data = self.data.merge(category_competition, on=['分类名称', '销售日期'], how='left')
        
        # 品类总销量
        category_total_quantity = self.data.groupby(['分类名称', '销售日期'])['正常销量(千克)'].sum().reset_index()
        category_total_quantity.columns = ['分类名称', '销售日期', 'category_total_quantity']
        self.data = self.data.merge(category_total_quantity, on=['分类名称', '销售日期'], how='left')
        
        # 单品在品类中的份额
        self.data['item_share_in_category'] = self.data['正常销量(千克)'] / self.data['category_total_quantity']
        
        print("品类特征创建完成")
        
    def handle_missing_values(self):
        """处理缺失值"""
        print("处理缺失值...")
        
        # 统计缺失值
        missing_stats = self.data.isnull().sum()
        missing_features = missing_stats[missing_stats > 0]
        
        if len(missing_features) > 0:
            print("缺失值统计:")
            for feature, count in missing_features.items():
                print(f"  {feature}: {count} ({count/len(self.data)*100:.1f}%)")
            
            # 数值特征用中位数填充
            numeric_features = self.data.select_dtypes(include=[np.number]).columns
            for feature in numeric_features:
                if self.data[feature].isnull().sum() > 0:
                    median_value = self.data[feature].median()
                    self.data[feature].fillna(median_value, inplace=True)
            
            print("缺失值处理完成")
        else:
            print("无缺失值")
            
    def select_features_for_modeling(self):
        """选择建模特征"""
        print("选择建模特征...")
        
        # 基础特征
        base_features = ['ln_price', 'ln_wholesale', 'ln_markup_ratio']
        
        # 时间特征
        time_features = ['time_trend', 'is_weekend'] + [col for col in self.data.columns if col.startswith('weekday_')]
        
        # 价格特征
        price_features = ['ln_relative_price_to_category', 'category_price_index']
        
        # 滚动特征
        rolling_features = [col for col in self.data.columns if 'rolling' in col or 'volatility' in col or 'deviation' in col]
        
        # 滞后特征
        lag_features = [col for col in self.data.columns if 'lag' in col or 'change' in col]
        
        # 交互特征（选择重要的）
        interaction_features = ['ln_price_x_time_trend', 'ln_price_x_weekend']
        
        # 品类特征
        category_features = ['category_competition_intensity', 'item_share_in_category']
        
        # 促销标记
        promotion_features = ['is_promotion'] if 'is_promotion' in self.data.columns else []
        
        # 合并所有特征
        all_features = (base_features + time_features + price_features + 
                       rolling_features + lag_features + interaction_features + 
                       category_features + promotion_features)
        
        # 过滤存在的特征
        available_features = [f for f in all_features if f in self.data.columns]
        
        # 目标变量和标识符
        target_and_ids = ['ln_quantity', '单品编码', '单品名称', '分类名称', '销售日期']
        
        # 创建特征数据集
        feature_columns = target_and_ids + available_features
        self.feature_data = self.data[feature_columns].copy()
        
        print(f"选择了 {len(available_features)} 个建模特征")
        print("特征类别分布:")
        print(f"  基础特征: {len([f for f in base_features if f in available_features])}")
        print(f"  时间特征: {len([f for f in time_features if f in available_features])}")
        print(f"  价格特征: {len([f for f in price_features if f in available_features])}")
        print(f"  滚动特征: {len([f for f in rolling_features if f in available_features])}")
        print(f"  滞后特征: {len([f for f in lag_features if f in available_features])}")
        print(f"  交互特征: {len([f for f in interaction_features if f in available_features])}")
        print(f"  品类特征: {len([f for f in category_features if f in available_features])}")
        
        return available_features
        
    def save_feature_data(self, train_ratio=0.7):
        """保存特征工程后的数据"""
        print("保存特征数据...")
        
        if self.feature_data is None:
            print("错误：没有特征数据")
            return
            
        # 按时间排序
        feature_data_sorted = self.feature_data.sort_values('销售日期')
        
        # 按时间切分
        split_date = feature_data_sorted['销售日期'].quantile(train_ratio)
        
        train_features = feature_data_sorted[feature_data_sorted['销售日期'] <= split_date]
        test_features = feature_data_sorted[feature_data_sorted['销售日期'] > split_date]
        
        # 保存文件
        train_features.to_csv('train_features.csv', index=False, encoding='utf-8')
        test_features.to_csv('test_features.csv', index=False, encoding='utf-8')
        feature_data_sorted.to_csv('full_features.csv', index=False, encoding='utf-8')
        
        print(f"训练特征: {len(train_features):,} 条记录")
        print(f"测试特征: {len(test_features):,} 条记录")
        print("特征文件已保存:")
        print("  - train_features.csv")
        print("  - test_features.csv")
        print("  - full_features.csv")
        
    def generate_feature_report(self):
        """生成特征工程报告"""
        print("生成特征工程报告...")
        
        report_content = []
        report_content.append("# 特征工程报告")
        report_content.append("")
        
        if self.feature_data is not None:
            report_content.append(f"## 数据概览")
            report_content.append(f"- 总样本数: {len(self.feature_data):,}")
            report_content.append(f"- 特征数量: {len(self.feature_data.columns) - 5}")  # 减去目标变量和ID
            report_content.append(f"- 单品数量: {self.feature_data['单品编码'].nunique()}")
            report_content.append(f"- 品类数量: {self.feature_data['分类名称'].nunique()}")
            report_content.append("")
            
            # 特征统计
            numeric_features = self.feature_data.select_dtypes(include=[np.number]).columns
            numeric_features = [f for f in numeric_features if f not in ['单品编码']]
            
            report_content.append("## 特征统计")
            report_content.append(f"- 数值特征: {len(numeric_features)}")
            
            # 特征相关性（与目标变量）
            if 'ln_quantity' in numeric_features:
                correlations = self.feature_data[numeric_features].corr()['ln_quantity'].abs().sort_values(ascending=False)
                top_corr_features = correlations.head(10)
                
                report_content.append("")
                report_content.append("## 与目标变量相关性最高的特征")
                for feature, corr in top_corr_features.items():
                    if feature != 'ln_quantity':
                        report_content.append(f"- {feature}: {corr:.4f}")
        
        # 保存报告
        report_text = "\n".join(report_content)
        with open('feature_engineering_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print("特征工程报告已保存: feature_engineering_report.md")
        
    def run_full_feature_engineering(self):
        """运行完整的特征工程流程"""
        print("开始特征工程...")
        
        self.load_data()
        self.create_time_features()
        self.create_price_features()
        self.create_rolling_features()
        self.create_lag_features()
        self.create_interaction_features()
        self.create_category_features()
        self.handle_missing_values()
        
        features = self.select_features_for_modeling()
        self.save_feature_data()
        self.generate_feature_report()
        
        print("\n特征工程完成！")
        return self, features

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run_full_feature_engineering()
