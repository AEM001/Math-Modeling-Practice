# -*- coding: utf-8 -*-
"""
数据质量审计与清洗模块
识别并处理异常数据、促销数据、极端值等
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataQualityAuditor:
    """数据质量审计器"""
    
    def __init__(self, data_path):
        """初始化审计器"""
        self.data_path = data_path
        self.raw_data = None
        self.clean_data = None
        self.audit_report = {}
        
    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
        self.raw_data = pd.read_csv(self.data_path)
        self.raw_data['销售日期'] = pd.to_datetime(self.raw_data['销售日期'])
        self.raw_data['weekday'] = self.raw_data['销售日期'].dt.dayofweek
        self.raw_data['month'] = self.raw_data['销售日期'].dt.month
        print(f"原始数据加载完成，共 {len(self.raw_data)} 条记录")
        return self
        
    def audit_basic_stats(self):
        """基础统计审计"""
        print("\n=== 基础统计审计 ===")
        
        # 基本信息
        total_records = len(self.raw_data)
        date_range = f"{self.raw_data['销售日期'].min()} 至 {self.raw_data['销售日期'].max()}"
        unique_items = self.raw_data['单品编码'].nunique()
        unique_categories = self.raw_data['分类名称'].nunique()
        
        print(f"数据时间范围: {date_range}")
        print(f"单品数量: {unique_items}")
        print(f"品类数量: {unique_categories}")
        
        # 品类分布
        category_dist = self.raw_data['分类名称'].value_counts()
        print(f"\n品类分布:")
        for cat, count in category_dist.items():
            print(f"  {cat}: {count} 条记录 ({count/total_records*100:.1f}%)")
            
        self.audit_report['basic_stats'] = {
            'total_records': total_records,
            'date_range': date_range,
            'unique_items': unique_items,
            'unique_categories': unique_categories,
            'category_distribution': category_dist.to_dict()
        }
        
    def audit_anomalies(self):
        """异常值审计"""
        print("\n=== 异常值审计 ===")
        
        anomalies = {}
        
        # 1. 零价格或零销量
        zero_price = self.raw_data[self.raw_data['正常销售单价(元/千克)'] <= 0]
        zero_quantity = self.raw_data[self.raw_data['正常销量(千克)'] <= 0]
        
        print(f"零价格记录: {len(zero_price)} 条")
        print(f"零销量记录: {len(zero_quantity)} 条")
        
        # 2. 负成本加成率
        negative_markup = self.raw_data[self.raw_data['成本加成率'] < 0]
        print(f"负成本加成率记录: {len(negative_markup)} 条")
        if len(negative_markup) > 0:
            print("负成本加成率样本:")
            print(negative_markup[['单品名称', '分类名称', '正常销售单价(元/千克)', 
                                 '批发价格(元/千克)', '成本加成率']].head())
        
        # 3. 异常高价格（超过批发价3倍）
        high_price = self.raw_data[
            self.raw_data['正常销售单价(元/千克)'] > 3 * self.raw_data['批发价格(元/千克)']
        ]
        print(f"异常高价格记录: {len(high_price)} 条")
        
        # 4. 异常高销量（使用MAD方法）
        def detect_outliers_mad(series, threshold=3):
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        # 按品类检测销量异常
        quantity_outliers = pd.DataFrame()
        for category in self.raw_data['分类名称'].unique():
            cat_data = self.raw_data[self.raw_data['分类名称'] == category]
            if len(cat_data) > 10:  # 只对样本量足够的品类检测
                outliers_mask = detect_outliers_mad(cat_data['正常销量(千克)'])
                if outliers_mask.sum() > 0:
                    cat_outliers = cat_data[outliers_mask]
                    quantity_outliers = pd.concat([quantity_outliers, cat_outliers])
        
        print(f"销量异常值记录: {len(quantity_outliers)} 条")
        
        # 5. 价格异常值检测
        price_outliers = pd.DataFrame()
        for category in self.raw_data['分类名称'].unique():
            cat_data = self.raw_data[self.raw_data['分类名称'] == category]
            if len(cat_data) > 10:
                outliers_mask = detect_outliers_mad(cat_data['正常销售单价(元/千克)'])
                if outliers_mask.sum() > 0:
                    cat_outliers = cat_data[outliers_mask]
                    price_outliers = pd.concat([price_outliers, cat_outliers])
        
        print(f"价格异常值记录: {len(price_outliers)} 条")
        
        anomalies = {
            'zero_price': len(zero_price),
            'zero_quantity': len(zero_quantity),
            'negative_markup': len(negative_markup),
            'high_price': len(high_price),
            'quantity_outliers': len(quantity_outliers),
            'price_outliers': len(price_outliers)
        }
        
        self.audit_report['anomalies'] = anomalies
        
        # 保存异常数据详情
        self.anomaly_data = {
            'zero_price': zero_price,
            'zero_quantity': zero_quantity,
            'negative_markup': negative_markup,
            'high_price': high_price,
            'quantity_outliers': quantity_outliers,
            'price_outliers': price_outliers
        }
        
    def audit_promotions(self):
        """促销数据审计"""
        print("\n=== 促销数据审计 ===")
        
        # 有打折销量的记录
        promotion_records = self.raw_data[self.raw_data['打折销量(千克)'] > 0]
        print(f"促销记录: {len(promotion_records)} 条 ({len(promotion_records)/len(self.raw_data)*100:.1f}%)")
        
        if len(promotion_records) > 0:
            # 促销品类分布
            promo_by_category = promotion_records['分类名称'].value_counts()
            print("促销记录按品类分布:")
            for cat, count in promo_by_category.items():
                print(f"  {cat}: {count} 条")
                
            # 促销折扣分析
            promotion_records = promotion_records.copy()
            promotion_records['discount_rate'] = (
                promotion_records['正常销售单价(元/千克)'] - 
                promotion_records['打折销售单价(元/千克)']
            ) / promotion_records['正常销售单价(元/千克)']
            
            print(f"平均折扣率: {promotion_records['discount_rate'].mean():.1%}")
            print(f"折扣率范围: {promotion_records['discount_rate'].min():.1%} - {promotion_records['discount_rate'].max():.1%}")
        
        self.audit_report['promotions'] = {
            'promotion_records': len(promotion_records),
            'promotion_rate': len(promotion_records)/len(self.raw_data),
            'promo_by_category': promo_by_category.to_dict() if len(promotion_records) > 0 else {}
        }
        
    def identify_stockout_candidates(self):
        """识别潜在售罄情况"""
        print("\n=== 潜在售罄识别 ===")
        
        stockout_candidates = []
        
        # 按单品分析
        for item_code in self.raw_data['单品编码'].unique():
            item_data = self.raw_data[self.raw_data['单品编码'] == item_code].sort_values('销售日期')
            
            if len(item_data) < 5:  # 样本太少跳过
                continue
                
            # 计算销量的历史分位数
            q95 = item_data['正常销量(千克)'].quantile(0.95)
            q50 = item_data['正常销量(千克)'].quantile(0.50)
            
            # 识别可能的售罄日：销量接近历史高位，且次日销量显著回升
            for i in range(len(item_data) - 1):
                current_day = item_data.iloc[i]
                next_day = item_data.iloc[i + 1]
                
                # 条件：当日销量 > P95，次日销量 < P50，价格不异常高
                if (current_day['正常销量(千克)'] > q95 and 
                    next_day['正常销量(千克)'] < q50 and
                    current_day['正常销售单价(元/千克)'] <= 2 * current_day['批发价格(元/千克)']):
                    
                    stockout_candidates.append({
                        'date': current_day['销售日期'],
                        'item_code': item_code,
                        'item_name': current_day['单品名称'],
                        'category': current_day['分类名称'],
                        'quantity': current_day['正常销量(千克)'],
                        'price': current_day['正常销售单价(元/千克)'],
                        'q95': q95,
                        'next_day_quantity': next_day['正常销量(千克)']
                    })
        
        stockout_df = pd.DataFrame(stockout_candidates)
        print(f"潜在售罄记录: {len(stockout_df)} 条")
        
        if len(stockout_df) > 0:
            print("按品类分布:")
            stockout_by_cat = stockout_df['category'].value_counts()
            for cat, count in stockout_by_cat.items():
                print(f"  {cat}: {count} 条")
        
        self.audit_report['stockouts'] = {
            'stockout_candidates': len(stockout_df),
            'stockout_by_category': stockout_by_cat.to_dict() if len(stockout_df) > 0 else {}
        }
        
        self.stockout_candidates = stockout_df
        
    def create_cleaning_rules(self):
        """创建数据清洗规则"""
        print("\n=== 数据清洗规则 ===")
        
        cleaning_rules = {
            'exclude_zero_price': True,  # 排除零价格
            'exclude_zero_quantity': True,  # 排除零销量
            'exclude_negative_markup': True,  # 排除负成本加成率
            'exclude_extreme_price': True,  # 排除极端价格（>3倍批发价）
            'handle_promotions': 'flag',  # 促销处理：'exclude'排除, 'flag'标记, 'keep'保留
            'exclude_quantity_outliers': True,  # 排除销量异常值
            'exclude_price_outliers': True,  # 排除价格异常值
            'exclude_stockout_candidates': False,  # 是否排除潜在售罄（保守处理）
        }
        
        print("清洗规则:")
        for rule, value in cleaning_rules.items():
            print(f"  {rule}: {value}")
            
        self.cleaning_rules = cleaning_rules
        
    def apply_cleaning_rules(self):
        """应用清洗规则"""
        print("\n=== 应用清洗规则 ===")
        
        clean_data = self.raw_data.copy()
        excluded_count = 0
        
        # 记录清洗过程
        cleaning_log = []
        
        if self.cleaning_rules['exclude_zero_price']:
            before = len(clean_data)
            clean_data = clean_data[clean_data['正常销售单价(元/千克)'] > 0]
            excluded = before - len(clean_data)
            excluded_count += excluded
            cleaning_log.append(f"排除零价格: {excluded} 条")
            
        if self.cleaning_rules['exclude_zero_quantity']:
            before = len(clean_data)
            clean_data = clean_data[clean_data['正常销量(千克)'] > 0]
            excluded = before - len(clean_data)
            excluded_count += excluded
            cleaning_log.append(f"排除零销量: {excluded} 条")
            
        if self.cleaning_rules['exclude_negative_markup']:
            before = len(clean_data)
            clean_data = clean_data[clean_data['成本加成率'] >= 0]
            excluded = before - len(clean_data)
            excluded_count += excluded
            cleaning_log.append(f"排除负成本加成率: {excluded} 条")
            
        if self.cleaning_rules['exclude_extreme_price']:
            before = len(clean_data)
            clean_data = clean_data[
                clean_data['正常销售单价(元/千克)'] <= 3 * clean_data['批发价格(元/千克)']
            ]
            excluded = before - len(clean_data)
            excluded_count += excluded
            cleaning_log.append(f"排除极端价格: {excluded} 条")
        
        # 处理促销数据
        if self.cleaning_rules['handle_promotions'] == 'exclude':
            before = len(clean_data)
            clean_data = clean_data[clean_data['打折销量(千克)'] == 0]
            excluded = before - len(clean_data)
            excluded_count += excluded
            cleaning_log.append(f"排除促销记录: {excluded} 条")
        elif self.cleaning_rules['handle_promotions'] == 'flag':
            clean_data['is_promotion'] = (clean_data['打折销量(千克)'] > 0).astype(int)
            cleaning_log.append("促销记录已标记")
        
        # 排除异常值（需要重新计算，因为数据已经变化）
        if self.cleaning_rules['exclude_quantity_outliers'] or self.cleaning_rules['exclude_price_outliers']:
            def detect_outliers_mad(series, threshold=3):
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad == 0:
                    return pd.Series([False] * len(series), index=series.index)
                modified_z_scores = 0.6745 * (series - median) / mad
                return np.abs(modified_z_scores) > threshold
            
            outlier_mask = pd.Series([False] * len(clean_data), index=clean_data.index)
            
            for category in clean_data['分类名称'].unique():
                cat_mask = clean_data['分类名称'] == category
                cat_data = clean_data[cat_mask]
                
                if len(cat_data) > 10:
                    if self.cleaning_rules['exclude_quantity_outliers']:
                        qty_outliers = detect_outliers_mad(cat_data['正常销量(千克)'])
                        outlier_mask[cat_data.index] |= qty_outliers
                        
                    if self.cleaning_rules['exclude_price_outliers']:
                        price_outliers = detect_outliers_mad(cat_data['正常销售单价(元/千克)'])
                        outlier_mask[cat_data.index] |= price_outliers
            
            before = len(clean_data)
            clean_data = clean_data[~outlier_mask]
            excluded = before - len(clean_data)
            excluded_count += excluded
            cleaning_log.append(f"排除异常值: {excluded} 条")
        
        print(f"清洗完成，共排除 {excluded_count} 条记录")
        print(f"清洗后数据: {len(clean_data)} 条记录")
        print("清洗日志:")
        for log in cleaning_log:
            print(f"  {log}")
            
        self.clean_data = clean_data
        self.audit_report['cleaning'] = {
            'excluded_count': excluded_count,
            'final_count': len(clean_data),
            'cleaning_log': cleaning_log
        }
        
    def generate_audit_report(self):
        """生成审计报告"""
        print("\n=== 生成审计报告 ===")
        
        report_content = []
        report_content.append("# 数据质量审计报告")
        report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # 基础统计
        report_content.append("## 基础统计")
        basic = self.audit_report['basic_stats']
        report_content.append(f"- 总记录数: {basic['total_records']:,}")
        report_content.append(f"- 数据时间范围: {basic['date_range']}")
        report_content.append(f"- 单品数量: {basic['unique_items']}")
        report_content.append(f"- 品类数量: {basic['unique_categories']}")
        report_content.append("")
        
        # 异常值统计
        report_content.append("## 异常值统计")
        anomalies = self.audit_report['anomalies']
        for key, value in anomalies.items():
            report_content.append(f"- {key}: {value:,} 条")
        report_content.append("")
        
        # 促销数据
        report_content.append("## 促销数据")
        promos = self.audit_report['promotions']
        report_content.append(f"- 促销记录: {promos['promotion_records']:,} 条 ({promos['promotion_rate']:.1%})")
        report_content.append("")
        
        # 清洗结果
        report_content.append("## 清洗结果")
        cleaning = self.audit_report['cleaning']
        report_content.append(f"- 排除记录: {cleaning['excluded_count']:,} 条")
        report_content.append(f"- 最终记录: {cleaning['final_count']:,} 条")
        report_content.append(f"- 保留率: {cleaning['final_count']/basic['total_records']:.1%}")
        
        # 保存报告
        report_text = "\n".join(report_content)
        with open('data_quality_audit_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print("审计报告已保存到: data_quality_audit_report.md")
        
    def save_clean_data(self, train_ratio=0.7):
        """保存清洗后的数据"""
        print(f"\n=== 保存清洗后数据 ===")
        
        if self.clean_data is None:
            print("错误：没有清洗后的数据")
            return
            
        # 按时间排序
        clean_data_sorted = self.clean_data.sort_values('销售日期')
        
        # 按时间切分训练集和测试集
        split_date = clean_data_sorted['销售日期'].quantile(train_ratio)
        
        train_data = clean_data_sorted[clean_data_sorted['销售日期'] <= split_date]
        test_data = clean_data_sorted[clean_data_sorted['销售日期'] > split_date]
        
        # 保存文件
        train_data.to_csv('train_data_cleaned.csv', index=False, encoding='utf-8')
        test_data.to_csv('test_data_cleaned.csv', index=False, encoding='utf-8')
        clean_data_sorted.to_csv('clean_data_full.csv', index=False, encoding='utf-8')
        
        print(f"训练集: {len(train_data):,} 条记录 (截止 {split_date.date()})")
        print(f"测试集: {len(test_data):,} 条记录")
        print("文件已保存:")
        print("  - train_data_cleaned.csv")
        print("  - test_data_cleaned.csv") 
        print("  - clean_data_full.csv")
        
    def run_full_audit(self):
        """运行完整审计流程"""
        print("开始数据质量审计...")
        
        self.load_data()
        self.audit_basic_stats()
        self.audit_anomalies()
        self.audit_promotions()
        self.identify_stockout_candidates()
        self.create_cleaning_rules()
        self.apply_cleaning_rules()
        self.generate_audit_report()
        self.save_clean_data()
        
        print("\n数据质量审计完成！")
        return self

if __name__ == "__main__":
    # 运行审计
    auditor = DataQualityAuditor('单品级每日汇总表.csv')
    auditor.run_full_audit()
