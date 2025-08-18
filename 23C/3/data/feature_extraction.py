import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extract_vegetable_features():
    """
    从单品级和品类级汇总表中提取2023年6月24-30日的可售品种特征
    生成三个CSV文件：
    1. 可售品种时间序列表
    2. 品种周统计表
    3. 过滤后的原始数据表
    """
    
    # 读取数据文件
    single_product_path = "/Users/Mac/Downloads/Math-Modeling-Practice/23C/3/单品级每日汇总表.csv"
    category_path = "/Users/Mac/Downloads/Math-Modeling-Practice/23C/3/品类级每日汇总表.csv"
    
    print("正在读取数据文件...")
    df_single = pd.read_csv(single_product_path)
    df_category = pd.read_csv(category_path)
    
    # 转换日期格式
    df_single['销售日期'] = pd.to_datetime(df_single['销售日期'])
    df_category['销售日期'] = pd.to_datetime(df_category['销售日期'])
    
    # 定义目标时间范围：2023年6月24-30日
    start_date = datetime(2023, 6, 24)
    end_date = datetime(2023, 6, 30)
    
    print(f"提取时间范围：{start_date.date()} 到 {end_date.date()}")
    
    # 筛选目标时间范围内的数据
    mask_single = (df_single['销售日期'] >= start_date) & (df_single['销售日期'] <= end_date)
    mask_category = (df_category['销售日期'] >= start_date) & (df_category['销售日期'] <= end_date)
    
    df_single_filtered = df_single[mask_single].copy()
    df_category_filtered = df_category[mask_category].copy()
    
    print(f"单品级数据筛选后记录数：{len(df_single_filtered)}")
    print(f"品类级数据筛选后记录数：{len(df_category_filtered)}")
    
    # 1. 生成可售品种时间序列表（按时间顺序排列）
    print("\n生成表1：可售品种时间序列表...")
    
    # 获取所有可售品种
    available_varieties = df_single_filtered.groupby(['销售日期', '单品编码', '单品名称', '分类编码', '分类名称']).agg({
        '正常销量(千克)': 'sum',
        '打折销量(千克)': 'sum',
        '总销量(千克)': 'sum',
        '正常销售单价(元/千克)': 'mean',
        '打折销售单价(元/千克)': 'mean',
        '平均销售单价(元/千克)': 'mean',
        '批发价格(元/千克)': 'mean',
        '成本加成率': 'mean',
        '损耗率(%)': 'mean'
    }).reset_index()
    
    # 按日期和单品名称排序
    available_varieties = available_varieties.sort_values(['销售日期', '单品名称'])
    
    # 将日期转换为字符串格式
    available_varieties['销售日期'] = available_varieties['销售日期'].dt.strftime('%Y-%m-%d')
    
    # 重新排列列顺序
    time_series_columns = ['销售日期', '单品编码', '单品名称', '分类编码', '分类名称', 
                          '正常销量(千克)', '打折销量(千克)', '总销量(千克)', 
                          '正常销售单价(元/千克)', '打折销售单价(元/千克)', '平均销售单价(元/千克)', 
                          '批发价格(元/千克)', '成本加成率', '损耗率(%)']
    available_varieties = available_varieties[time_series_columns]
    
    print(f"可售品种总数：{len(available_varieties['单品名称'].unique())}")
    print(f"时间序列记录总数：{len(available_varieties)}")
    
    # 2. 生成品种周统计表（精简版）
    print("\n生成表2：品种周统计表...")
    
    # 先计算全部统计指标
    weekly_stats_full = df_single_filtered.groupby(['单品编码', '单品名称', '分类编码', '分类名称']).agg({
        '正常销量(千克)': 'sum',
        '打折销量(千克)': 'sum',
        '总销量(千克)': ['sum', 'std', 'max'],
        '正常销售单价(元/千克)': 'mean',
        '打折销售单价(元/千克)': 'mean',
        '平均销售单价(元/千克)': ['mean', 'std'],
        '批发价格(元/千克)': 'mean',
        '成本加成率': 'mean',
        '损耗率(%)': 'mean',
        '销售日期': 'count'  # 销售天数
    }).round(4)
    
    # 展平多级列名
    weekly_stats_full.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in weekly_stats_full.columns]
    weekly_stats = weekly_stats_full.reset_index()
    
    # 重命名列（精简版）
    column_rename = {
        '销售日期_count': '销售天数',
        '正常销量(千克)_sum': '周正常销量(千克)',
        '打折销量(千克)_sum': '周打折销量(千克)',
        '总销量(千克)_sum': '周总销量(千克)',
        '总销量(千克)_std': '销量标准差',
        '总销量(千克)_max': '最大日销量',
        '正常销售单价(元/千克)_mean': '平均正常售价',
        '打折销售单价(元/千克)_mean': '平均打折售价',
        '平均销售单价(元/千克)_mean': '平均售价',
        '平均销售单价(元/千克)_std': '售价标准差',
        '批发价格(元/千克)_mean': '平均批发价',
        '成本加成率_mean': '平均成本加成率',
        '损耗率(%)_mean': '平均损耗率'
    }
    
    weekly_stats = weekly_stats.rename(columns=column_rename)
    
    # 按周总销量降序排列
    weekly_stats = weekly_stats.sort_values('周总销量(千克)', ascending=False)
    
    print(f"周统计品种数：{len(weekly_stats)}")
    
    # 3. 生成过滤后的原始数据表（保留选定品种的所有数据）
    print("\n生成表3：过滤后的原始数据表...")
    
    # 获取选定品种的编码列表
    selected_varieties = set(weekly_stats['单品编码'].unique())
    
    # 从原始数据中筛选这些品种的所有记录（不限时间范围）
    filtered_original_single = df_single[df_single['单品编码'].isin(selected_varieties)].copy()
    filtered_original_category = df_category[df_category['分类编码'].isin(
        df_single[df_single['单品编码'].isin(selected_varieties)]['分类编码'].unique()
    )].copy()
    
    # 按日期排序
    filtered_original_single = filtered_original_single.sort_values(['销售日期', '单品名称'])
    filtered_original_category = filtered_original_category.sort_values(['销售日期', '分类名称'])
    
    print(f"过滤后单品级数据记录数：{len(filtered_original_single)}")
    print(f"过滤后品类级数据记录数：{len(filtered_original_category)}")
    
    # 保存CSV文件
    print("\n保存CSV文件...")
    
    # 保存表1：可售品种时间序列表
    time_series_file = "/Users/Mac/Downloads/Math-Modeling-Practice/23C/3/可售品种时间序列表_20230624-30.csv"
    available_varieties.to_csv(time_series_file, index=False, encoding='utf-8-sig')
    print(f"已保存：{time_series_file}")
    
    # 保存表2：品种周统计表
    weekly_stats_file = "/Users/Mac/Downloads/Math-Modeling-Practice/23C/3/品种周统计表_20230624-30.csv"
    weekly_stats.to_csv(weekly_stats_file, index=False, encoding='utf-8-sig')
    print(f"已保存：{weekly_stats_file}")
    
    # 保存表3：过滤后的原始数据表
    filtered_single_file = "/Users/Mac/Downloads/Math-Modeling-Practice/23C/3/过滤后单品级汇总表.csv"
    filtered_original_single.to_csv(filtered_single_file, index=False, encoding='utf-8-sig')
    print(f"已保存：{filtered_single_file}")
    
    filtered_category_file = "/Users/Mac/Downloads/Math-Modeling-Practice/23C/3/过滤后品类级汇总表.csv"
    filtered_original_category.to_csv(filtered_category_file, index=False, encoding='utf-8-sig')
    print(f"已保存：{filtered_category_file}")
    
    # 输出统计信息
    print("\n=== 特征提取完成 ===")
    print(f"目标时间范围：2023-06-24 到 2023-06-30")
    print(f"可售品种数量：{len(weekly_stats)}")
    print(f"时间序列记录数：{len(available_varieties)}")
    print(f"品种分类数：{len(weekly_stats['分类名称'].unique())}")
    
    print("\n主要品类分布：")
    category_counts = weekly_stats['分类名称'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}个品种")
    
    print("\n销量前10的品种：")
    top_varieties = weekly_stats.head(10)[['单品名称', '分类名称', '周总销量(千克)', '平均售价']]
    for idx, row in top_varieties.iterrows():
        print(f"  {row['单品名称']} ({row['分类名称']}): {row['周总销量(千克)']:.2f}千克, 均价{row['平均售价']:.2f}元/千克")
    
    return {
        'time_series': available_varieties,
        'weekly_stats': weekly_stats,
        'filtered_single': filtered_original_single,
        'filtered_category': filtered_original_category
    }

if __name__ == "__main__":
    # 执行特征提取
    results = extract_vegetable_features()
    print("\n特征提取任务完成！")
