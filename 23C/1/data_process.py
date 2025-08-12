import pandas as pd
import time
import re

def clean_product_name(name):
    """
    清理单品名称，去除数字编号（来源标识）
    例如：'黄心菜(1)' -> '黄心菜'
         '小青菜(2)' -> '小青菜'
         '鲜粽叶(袋)(1)' -> '鲜粽叶(袋)'
    """
    if pd.isna(name):
        return name
    # 去除末尾的 (数字) 格式的编号
    cleaned_name = re.sub(r'\(\d+\)$', '', str(name)).strip()
    return cleaned_name

def process_vegetable_sales_data(info_xlsx_path, sales_xlsx_path):
    """
    高效处理蔬菜销售数据，整合附件1（商品信息，独立文件）和附件2（销售流水，独立文件），并进行聚合。

    Args:
        info_xlsx_path (str): 附件1（商品信息）的Excel文件路径。
        sales_xlsx_path (str): 附件2（销售流水）的Excel文件路径。

    Returns:
        tuple: 包含两个DataFrame: 
               1. df_daily_item_sales (每日单品销量)
               2. df_daily_category_sales (每日品类销量)
    """
    print("开始处理数据...")
    start_time = time.time()

    # --- 1. 高效加载数据 ---
    # 读取附件1: 商品信息
    # 先不指定dtype，让pandas自动推断，避免类型不匹配问题
    print("正在读取附件1: 商品信息（独立文件）...")
    df_info = pd.read_excel(info_xlsx_path)

    # 读取附件2: 销售流水，并立即筛选所需列以节省内存
    print("正在读取附件2: 销售流水（独立文件，可能需要一些时间）...")
    required_cols = ['销售日期', '单品编码', '销量(千克)']
    df_sales = pd.read_excel(
        sales_xlsx_path,
        usecols=required_cols # 只加载我们需要的列，这是第一步优化
    )
    print(f"附件2 加载完成，共 {len(df_sales)} 条销售记录。")

    # --- 2. 数据清洗与类型转换 ---
    print("正在进行数据清洗和类型转换...")
    
    # 转换日期格式， errors='coerce' 会将无法转换的值变为NaT(Not a Time)
    df_sales['销售日期'] = pd.to_datetime(df_sales['销售日期'], errors='coerce')

    # 转换销量为数值类型，errors='coerce' 会将无效值变为NaN(Not a Number)
    df_sales['销量(千克)'] = pd.to_numeric(df_sales['销量(千克)'], errors='coerce')
    
    # 确保两个DataFrame中的单品编码类型一致
    print(f"附件1单品编码类型: {df_info['单品编码'].dtype}")
    print(f"附件2单品编码类型: {df_sales['单品编码'].dtype}")
    
    # 将两个DataFrame的单品编码都转换为相同的数据类型
    df_info['单品编码'] = df_info['单品编码'].astype(str)
    df_sales['单品编码'] = df_sales['单品编码'].astype(str)
    
    # 删除转换过程中产生的无效数据行（如没有日期或销量的记录）
    initial_rows = len(df_sales)
    df_sales.dropna(subset=['销售日期', '销量(千克)'], inplace=True)
    if initial_rows > len(df_sales):
        print(f"已清理 {initial_rows - len(df_sales)} 条无效数据行。")


    # --- 3. 合并数据 ---
    print("正在合并销售数据与商品信息...")
    # 使用 left join，保留所有销售记录
    df_merged = pd.merge(df_sales, df_info, on='单品编码', how='left')
    
    # 检查是否有未匹配上的商品信息
    unmatched_count = df_merged['分类名称'].isnull().sum()
    if unmatched_count > 0:
        print(f"警告：有 {unmatched_count} 条销售记录未能找到对应的商品信息。")
    
    # --- 3.5 清理单品名称，去除来源编号 ---
    print("正在清理单品名称，去除来源编号...")
    df_merged['清理后单品名称'] = df_merged['单品名称'].apply(clean_product_name)
    
    # 显示一些清理示例
    sample_cleaning = df_merged[['单品名称', '清理后单品名称']].drop_duplicates().head(10)
    print("单品名称清理示例：")
    for _, row in sample_cleaning.iterrows():
        if row['单品名称'] != row['清理后单品名称']:
            print(f"  '{row['单品名称']}' -> '{row['清理后单品名称']}'")
    
    # 统计清理前后的单品数量变化
    original_count = df_merged['单品名称'].nunique()
    cleaned_count = df_merged['清理后单品名称'].nunique()
    print(f"清理前单品数量: {original_count}, 清理后单品数量: {cleaned_count}")
    print(f"合并了 {original_count - cleaned_count} 个不同来源的重复单品")


    # --- 4. 数据聚合 ---
    print("正在按 '原始单品编码' 聚合日销量...")
    # 聚合1: 计算每个原始单品编码每天的总销量（保留原始数据）
    df_daily_item_sales_original = df_merged.groupby(
        ['销售日期', '单品编码', '单品名称', '分类名称'], observed=True
    )['销量(千克)'].sum().reset_index()

    print("正在按 '清理后单品名称' 聚合日销量...")
    # 聚合2: 按清理后的单品名称聚合（合并不同来源的同种蔬菜）
    df_daily_item_sales_cleaned = df_merged.groupby(
        ['销售日期', '清理后单品名称', '分类名称'], observed=True
    )['销量(千克)'].sum().reset_index()

    print("正在按 '品类' 聚合日销量...")
    # 聚合3: 基于清理后的数据，计算每个品类每天的总销量
    df_daily_category_sales = df_daily_item_sales_cleaned.groupby(
        ['销售日期', '分类名称'], observed=True
    )['销量(千克)'].sum().reset_index()


    # --- 任务完成 ---
    end_time = time.time()
    print(f"\n数据处理完成！总耗时: {end_time - start_time:.2f} 秒。")
    
    return df_daily_item_sales_original, df_daily_item_sales_cleaned, df_daily_category_sales

# --- 主程序入口 ---
if __name__ == '__main__':
    # 使用两个独立文件路径（用户提供的绝对路径）
    INFO_FILE_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C/附件1.xlsx'
    SALES_FILE_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C/附件2.xlsx'

    try:
        # 执行处理函数（两个独立文件）
        daily_item_sales_original, daily_item_sales_cleaned, daily_category_sales = process_vegetable_sales_data(INFO_FILE_PATH, SALES_FILE_PATH)

        # --- 查看处理结果 ---
        print("\n--- 每日单品销量（原始，按编码） (前5条) ---")
        print(daily_item_sales_original.head())
        print(f"\n生成了 {len(daily_item_sales_original)} 条原始每日单品销售记录。")

        print("\n--- 每日单品销量（清理后，合并来源） (前5条) ---")
        print(daily_item_sales_cleaned.head())
        print(f"\n生成了 {len(daily_item_sales_cleaned)} 条清理后每日单品销售记录。")

        print("\n--- 每日品类销量 (前5条) ---")
        print(daily_category_sales.head())
        print(f"\n生成了 {len(daily_category_sales)} 条每日品类销售记录。")
        
        # --- 保存处理结果到文件 ---
        print("\n正在保存结果到新的Excel文件...")
        with pd.ExcelWriter('processed_sales_data.xlsx') as writer:
            daily_item_sales_original.to_excel(writer, sheet_name='每日单品销量_原始', index=False)
            daily_item_sales_cleaned.to_excel(writer, sheet_name='每日单品销量_合并来源', index=False)
            daily_category_sales.to_excel(writer, sheet_name='每日品类销量', index=False)
        print("结果已保存至 'processed_sales_data.xlsx'")
        
        # --- 显示清理效果统计 ---
        print(f"\n=== 数据清理效果统计 ===")
        original_unique_items = daily_item_sales_original['单品名称'].nunique()
        cleaned_unique_items = daily_item_sales_cleaned['清理后单品名称'].nunique()
        print(f"原始单品种类数: {original_unique_items}")
        print(f"清理后单品种类数: {cleaned_unique_items}")
        print(f"合并的重复单品数: {original_unique_items - cleaned_unique_items}")

    except FileNotFoundError:
        print("错误：找不到文件。请检查以下路径是否正确：")
        print(f"  附件1（商品信息）: {INFO_FILE_PATH}")
        print(f"  附件2（销售流水）: {SALES_FILE_PATH}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")