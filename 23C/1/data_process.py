import pandas as pd
import time
import re

def clean_product_name(name):
   
    # 清理单品名称，去除数字编号（来源标识）
    
   
    if pd.isna(name):
        return name
    # 去除末尾的 (数字) 格式的编号
    cleaned_name = re.sub(r'\(\d+\)$', '', str(name)).strip()
    return cleaned_name

def process_sales_data_by_item(info_xlsx_path, sales_xlsx_path):
    """
    处理销售数据，按照单品进行合并和聚合
    
    Args:
        info_xlsx_path (str): 附件1（商品信息）的Excel文件路径
        sales_xlsx_path (str): 附件2（销售流水）的Excel文件路径
    
    Returns:
        DataFrame: 包含销售日期、单品名称、单品销量（天）、所属品类的数据表
    """
    print("开始处理销售数据...")
    start_time = time.time()

    # --- 1. 加载数据 ---
    print("正在读取附件1: 商品信息...")
    df_info = pd.read_excel(info_xlsx_path)
    
    print("正在读取附件2: 销售流水...")
    # 只提取需要的列：销售日期、单品编码、销量
    required_cols = ['销售日期', '单品编码', '销量(千克)']
    df_sales = pd.read_excel(sales_xlsx_path, usecols=required_cols)
    print(f"附件2 加载完成，共 {len(df_sales)} 条销售记录。")

    # --- 2. 数据清洗与类型转换 ---
    print("正在进行数据清洗和类型转换...")
    
    # 转换日期格式
    df_sales['销售日期'] = pd.to_datetime(df_sales['销售日期'], errors='coerce')
    
    # 转换销量为数值类型
    df_sales['销量(千克)'] = pd.to_numeric(df_sales['销量(千克)'], errors='coerce')
    
    # 确保单品编码类型一致
    df_info['单品编码'] = df_info['单品编码'].astype(str)
    df_sales['单品编码'] = df_sales['单品编码'].astype(str)
    
    # 删除无效数据行
    initial_rows = len(df_sales)
    df_sales.dropna(subset=['销售日期', '销量(千克)'], inplace=True)
    if initial_rows > len(df_sales):
        print(f"已清理 {initial_rows - len(df_sales)} 条无效数据行。")

    # --- 3. 合并销售数据与商品信息 ---
    print("正在合并销售数据与商品信息...")
    df_merged = pd.merge(df_sales, df_info, on='单品编码', how='left')
    
    # 检查未匹配的记录
    unmatched_count = df_merged['分类名称'].isnull().sum()
    if unmatched_count > 0:
        print(f"警告：有 {unmatched_count} 条销售记录未能找到对应的商品信息。")
    
    # --- 4. 清理单品名称，去除来源编号 ---
    print("正在清理单品名称，去除来源编号...")
    df_merged['清理后单品名称'] = df_merged['单品名称'].apply(clean_product_name)
    
    # 显示清理示例
    sample_cleaning = df_merged[['单品名称', '清理后单品名称']].drop_duplicates().head(10)
    print("单品名称清理示例：")
    for _, row in sample_cleaning.iterrows():
        if row['单品名称'] != row['清理后单品名称']:
            print(f"  '{row['单品名称']}' -> '{row['清理后单品名称']}'")
    
    # 统计清理效果
    original_count = df_merged['单品名称'].nunique()
    cleaned_count = df_merged['清理后单品名称'].nunique()
    print(f"清理前单品数量: {original_count}, 清理后单品数量: {cleaned_count}")
    print(f"合并了 {original_count - cleaned_count} 个不同来源的重复单品")

    # --- 5. 按天聚合单品销量 ---
    print("正在按天聚合单品销量...")
    # 按销售日期、清理后单品名称、分类名称进行分组，对销量求和
    df_daily_item_sales = df_merged.groupby(
        ['销售日期', '清理后单品名称', '分类名称'], observed=True
    )['销量(千克)'].sum().reset_index()
    
    # 重命名列以符合要求
    df_daily_item_sales = df_daily_item_sales.rename(columns={
        '清理后单品名称': '单品名称',
        '销量(千克)': '单品销量(天)'
    })
    
    # 按日期和单品名称排序
    df_daily_item_sales = df_daily_item_sales.sort_values(['销售日期', '单品名称'])

    # --- 任务完成 ---
    end_time = time.time()
    print(f"\n数据处理完成！总耗时: {end_time - start_time:.2f} 秒。")
    
    return df_daily_item_sales

# --- 主程序入口 ---
if __name__ == '__main__':
    # 文件路径
    INFO_FILE_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C/附件1.xlsx'
    SALES_FILE_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C/附件2.xlsx'

    try:
        # 执行数据处理
        daily_item_sales = process_sales_data_by_item(INFO_FILE_PATH, SALES_FILE_PATH)

        # --- 查看处理结果 ---
        print("\n--- 每日单品销量汇总 (前10条) ---")
        print(daily_item_sales.head(10))
        print(f"\n生成了 {len(daily_item_sales)} 条每日单品销售记录。")
        
        # 显示数据统计信息
        print(f"\n=== 数据统计信息 ===")
        print(f"时间范围: {daily_item_sales['销售日期'].min()} 至 {daily_item_sales['销售日期'].max()}")
        print(f"单品种类数: {daily_item_sales['单品名称'].nunique()}")
        print(f"品类数量: {daily_item_sales['分类名称'].nunique()}") # Changed from '所属品类' to '分类名称'
        print(f"总销量: {daily_item_sales['单品销量(天)'].sum():.2f} 千克")
        
        # --- 保存为CSV文件 ---
        output_filename = 'daily_item_sales_summary.csv'
        print(f"\n正在保存结果到CSV文件: {output_filename}")
        daily_item_sales.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"结果已保存至 '{output_filename}'")
        
        # 显示文件大小
        import os
        file_size = os.path.getsize(output_filename) / 1024  # KB
        print(f"文件大小: {file_size:.2f} KB")

    except FileNotFoundError:
        print("错误：找不到文件。请检查以下路径是否正确：")
        print(f"  附件1（商品信息）: {INFO_FILE_PATH}")
        print(f"  附件2（销售流水）: {SALES_FILE_PATH}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()