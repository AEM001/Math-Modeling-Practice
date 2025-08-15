#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本：按时间顺序创建单品级和品类级每日汇总表
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载所有数据文件"""
    print("正在加载数据文件...")
    
    # 加载附件1：商品信息
    df_products = pd.read_excel('附件1.xlsx')
    print(f"附件1 商品信息: {df_products.shape[0]} 个单品")
    
    # 加载附件2：销售流水
    df_sales = pd.read_excel('附件2.xlsx')
    df_sales['销售日期'] = pd.to_datetime(df_sales['销售日期'])
    print(f"附件2 销售流水: {df_sales.shape[0]} 条记录")
    print(f"销售日期范围: {df_sales['销售日期'].min()} 到 {df_sales['销售日期'].max()}")
    
    # 加载附件3：批发价格
    df_wholesale = pd.read_excel('附件3.xlsx')
    df_wholesale['日期'] = pd.to_datetime(df_wholesale['日期'])
    print(f"附件3 批发价格: {df_wholesale.shape[0]} 条记录")
    
    # 加载附件4：损耗率 (Sheet1)
    df_loss_rate = pd.read_excel('附件4.xlsx', sheet_name='Sheet1')
    print(f"附件4 损耗率: {df_loss_rate.shape[0]} 个单品")
    
    return df_products, df_sales, df_wholesale, df_loss_rate

def create_daily_item_summary(df_products, df_sales, df_wholesale, df_loss_rate):
    """创建单品级每日汇总表"""
    print("创建单品级每日汇总表...")
    
    # 按日期、单品编码、是否打折分组统计销售数据
    daily_sales = df_sales.groupby(['销售日期', '单品编码', '是否打折销售']).agg({
        '销量(千克)': 'sum',
        '销售单价(元/千克)': 'mean'
    }).reset_index()
    
    # 为了保持完整性，我们需要将打折和非打折的数据合并到一行
    # 先分别处理打折和非打折的数据
    discount_sales = daily_sales[daily_sales['是否打折销售'] == '是'].copy()
    discount_sales = discount_sales.rename(columns={
        '销量(千克)': '打折销量(千克)',
        '销售单价(元/千克)': '打折销售单价(元/千克)'
    })
    discount_sales = discount_sales.drop('是否打折销售', axis=1)
    
    normal_sales = daily_sales[daily_sales['是否打折销售'] == '否'].copy()
    normal_sales = normal_sales.rename(columns={
        '销量(千克)': '正常销量(千克)',
        '销售单价(元/千克)': '正常销售单价(元/千克)'
    })
    normal_sales = normal_sales.drop('是否打折销售', axis=1)
    
    # 合并打折和非打折数据
    item_daily = pd.merge(normal_sales, discount_sales, 
                         on=['销售日期', '单品编码'], how='outer')
    
    # 填充缺失值为0
    item_daily['正常销量(千克)'] = item_daily['正常销量(千克)'].fillna(0)
    item_daily['打折销量(千克)'] = item_daily['打折销量(千克)'].fillna(0)
    item_daily['正常销售单价(元/千克)'] = item_daily['正常销售单价(元/千克)'].fillna(0)
    item_daily['打折销售单价(元/千克)'] = item_daily['打折销售单价(元/千克)'].fillna(0)
    
    # 计算总销量和平均售价
    item_daily['总销量(千克)'] = item_daily['正常销量(千克)'] + item_daily['打折销量(千克)']
    
    # 计算加权平均售价
    item_daily['平均销售单价(元/千克)'] = np.where(
        item_daily['总销量(千克)'] > 0,
        (item_daily['正常销量(千克)'] * item_daily['正常销售单价(元/千克)'] + 
         item_daily['打折销量(千克)'] * item_daily['打折销售单价(元/千克)']) / item_daily['总销量(千克)'],
        0
    )
    
    # 关联批发价格
    item_daily = pd.merge(item_daily, df_wholesale, 
                         left_on=['销售日期', '单品编码'], 
                         right_on=['日期', '单品编码'], 
                         how='left')
    item_daily = item_daily.drop('日期', axis=1)
    
    # 关联商品信息
    item_daily = pd.merge(item_daily, df_products, on='单品编码', how='left')
    
    # 关联损耗率
    item_daily = pd.merge(item_daily, df_loss_rate[['单品编码', '损耗率(%)']], 
                         on='单品编码', how='left')
    
    # 计算成本加成率
    item_daily['成本加成率'] = np.where(
        (item_daily['批发价格(元/千克)'] > 0) & (item_daily['平均销售单价(元/千克)'] > 0),
        (item_daily['平均销售单价(元/千克)'] - item_daily['批发价格(元/千克)']) / item_daily['批发价格(元/千克)'],
        np.nan
    )
    
    # 处理缺失的批发价格（用相邻日期均值填补）
    item_daily = item_daily.sort_values(['单品编码', '销售日期'])
    item_daily['批发价格(元/千克)'] = item_daily.groupby('单品编码')['批发价格(元/千克)'].transform(
        lambda x: x.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    )
    
    # 重新计算成本加成率
    item_daily['成本加成率'] = np.where(
        (item_daily['批发价格(元/千克)'] > 0) & (item_daily['平均销售单价(元/千克)'] > 0),
        (item_daily['平均销售单价(元/千克)'] - item_daily['批发价格(元/千克)']) / item_daily['批发价格(元/千克)'],
        np.nan
    )
    
    # 按时间顺序排序
    item_daily = item_daily.sort_values(['销售日期', '单品编码'])
    
    # 选择最终列
    final_columns = [
        '销售日期', '单品编码', '单品名称', '分类编码', '分类名称',
        '正常销量(千克)', '打折销量(千克)', '总销量(千克)',
        '正常销售单价(元/千克)', '打折销售单价(元/千克)', '平均销售单价(元/千克)',
        '批发价格(元/千克)', '成本加成率', '损耗率(%)'
    ]
    
    item_daily = item_daily[final_columns]
    
    print(f"单品级每日汇总表完成: {item_daily.shape[0]} 行记录")
    return item_daily

def create_daily_category_summary(item_daily):
    """创建品类级每日汇总表"""
    print("创建品类级每日汇总表...")
    
    # 按日期和品类分组汇总
    category_daily = item_daily.groupby(['销售日期', '分类编码', '分类名称']).agg({
        '正常销量(千克)': 'sum',
        '打折销量(千克)': 'sum',
        '总销量(千克)': 'sum',
        '正常销售单价(元/千克)': lambda x: np.average(x[x > 0]) if len(x[x > 0]) > 0 else 0,
        '打折销售单价(元/千克)': lambda x: np.average(x[x > 0]) if len(x[x > 0]) > 0 else 0,
        '平均销售单价(元/千克)': lambda x: np.average(x[x > 0]) if len(x[x > 0]) > 0 else 0,
        '批发价格(元/千克)': lambda x: np.average(x[x > 0]) if len(x[x > 0]) > 0 else 0,
        '成本加成率': 'mean',
        '损耗率(%)': 'mean'
    }).reset_index()
    
    # 重新计算品类级的加权平均售价
    category_daily['平均销售单价(元/千克)'] = np.where(
        category_daily['总销量(千克)'] > 0,
        (category_daily['正常销量(千克)'] * category_daily['正常销售单价(元/千克)'] + 
         category_daily['打折销量(千克)'] * category_daily['打折销售单价(元/千克)']) / category_daily['总销量(千克)'],
        0
    )
    
    # 重新计算品类级成本加成率
    category_daily['成本加成率'] = np.where(
        (category_daily['批发价格(元/千克)'] > 0) & (category_daily['平均销售单价(元/千克)'] > 0),
        (category_daily['平均销售单价(元/千克)'] - category_daily['批发价格(元/千克)']) / category_daily['批发价格(元/千克)'],
        np.nan
    )
    
    # 按时间顺序排序
    category_daily = category_daily.sort_values(['销售日期', '分类编码'])
    
    print(f"品类级每日汇总表完成: {category_daily.shape[0]} 行记录")
    return category_daily

def save_to_csv(item_daily, category_daily):
    """保存为CSV格式"""
    print("保存CSV文件...")
    
    # 保存单品级每日汇总
    item_daily.to_csv('单品级每日汇总表.csv', index=False, encoding='utf-8-sig')
    print(f"单品级每日汇总表已保存: {item_daily.shape[0]} 行")
    
    # 保存品类级每日汇总
    category_daily.to_csv('品类级每日汇总表.csv', index=False, encoding='utf-8-sig')
    print(f"品类级每日汇总表已保存: {category_daily.shape[0]} 行")
    
    # 显示数据概览
    print("\n=== 单品级每日汇总表概览 ===")
    print(f"日期范围: {item_daily['销售日期'].min()} 到 {item_daily['销售日期'].max()}")
    print(f"单品数量: {item_daily['单品编码'].nunique()}")
    print(f"品类数量: {item_daily['分类编码'].nunique()}")
    print("前5行数据:")
    print(item_daily.head())
    
    print("\n=== 品类级每日汇总表概览 ===")
    print(f"日期范围: {category_daily['销售日期'].min()} 到 {category_daily['销售日期'].max()}")
    print(f"品类数量: {category_daily['分类编码'].nunique()}")
    print("前5行数据:")
    print(category_daily.head())

def main():
    """主函数"""
    try:
        # 1. 加载数据
        df_products, df_sales, df_wholesale, df_loss_rate = load_data()
        
        # 2. 创建单品级每日汇总表
        item_daily = create_daily_item_summary(df_products, df_sales, df_wholesale, df_loss_rate)
        
        # 3. 创建品类级每日汇总表
        category_daily = create_daily_category_summary(item_daily)
        
        # 4. 保存为CSV
        save_to_csv(item_daily, category_daily)
        
        print("\n数据处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
