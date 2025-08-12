import pandas as pd
import numpy as np
import time
import os

BASE_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C'
OUTPUT_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C/2'

# ---------------------- 数据加载与清洗 ----------------------

def load_all_data(base_path: str):
    print('=== 开始加载所有附件数据 ===')
    
    # 附件1: 商品信息
    print('正在加载附件1: 商品信息...')
    df_info = pd.read_excel(os.path.join(base_path, '附件1.xlsx'))
    print(f'附件1加载完成: {len(df_info)} 个单品')
    
    # 附件2: 销售流水 
    print('正在加载附件2: 销售流水（可能需要较长时间）...')
    df_sales = pd.read_excel(os.path.join(base_path, '附件2.xlsx'))
    print(f'附件2加载完成: {len(df_sales)} 条销售记录')
    
    # 附件3: 批发价格
    print('正在加载附件3: 批发价格...')
    df_wholesale = pd.read_excel(os.path.join(base_path, '附件3.xlsx'))
    print(f'附件3加载完成: {len(df_wholesale)} 条批发价格记录')
    
    # 附件4: 损耗率
    print('正在加载附件4: 损耗率...')
    df_loss_item = pd.read_excel(os.path.join(base_path, '附件4.xlsx'), sheet_name='Sheet1')
    xls = pd.ExcelFile(os.path.join(base_path, '附件4.xlsx'))
    loss_sheet_names = [name for name in xls.sheet_names if '平均损耗率' in name]
    if loss_sheet_names:
        df_loss_category = pd.read_excel(os.path.join(base_path, '附件4.xlsx'), sheet_name=loss_sheet_names[0])
        print(f'附件4-品类损耗率加载完成: {len(df_loss_category)} 条记录')
    else:
        df_loss_category = pd.DataFrame()
        print('未找到品类损耗率数据')
    
    return df_info, df_sales, df_wholesale, df_loss_item, df_loss_category

def clean_sales_data(df_sales: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 清洗销售流水数据 ===')
    initial_count = len(df_sales)
    df = df_sales.copy()
    
    # 类型与有效性
    df['销售日期'] = pd.to_datetime(df['销售日期'], errors='coerce')
    df['销量(千克)'] = pd.to_numeric(df['销量(千克)'], errors='coerce')
    df['销售单价(元/千克)'] = pd.to_numeric(df['销售单价(元/千克)'], errors='coerce')
    
    df = df.dropna(subset=['销售日期'])
    df = df.dropna(subset=['销量(千克)', '销售单价(元/千克)'])
    df = df[df['销售单价(元/千克)'] > 0]
    df = df[df['销量(千克)'] > 0]
    
    print(f'数据清洗完成，保留 {len(df)}/{initial_count} 条有效记录')
    return df

def merge_product_info(df_sales: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 合并商品信息 ===')
    df = df_sales.copy()
    info = df_info.copy()
    df['单品编码'] = df['单品编码'].astype(str)
    info['单品编码'] = info['单品编码'].astype(str)
    df = pd.merge(df, info, on='单品编码', how='left')
    missing = df['分类名称'].isnull().sum()
    if missing:
        print(f'警告: {missing} 条销售记录未找到对应商品信息，将被剔除。')
        df = df.dropna(subset=['分类名称'])
    return df

def merge_wholesale_prices(df_merged: pd.DataFrame, df_wholesale: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 合并批发价格数据 ===')
    df_wh = df_wholesale.copy()
    df_wh['日期'] = pd.to_datetime(df_wh['日期'], errors='coerce')
    df_wh['单品编码'] = df_wh['单品编码'].astype(str)
    df_wh['批发价格(元/千克)'] = pd.to_numeric(df_wh['批发价格(元/千克)'], errors='coerce')
    df_wh = df_wh.dropna(subset=['日期', '批发价格(元/千克)'])
    df_wh = df_wh[df_wh['批发价格(元/千克)'] > 0]

    df = pd.merge(
        df_merged, 
        df_wh.rename(columns={'日期': '销售日期'})[['销售日期', '单品编码', '批发价格(元/千克)']], 
        on=['销售日期', '单品编码'], 
        how='left'
    )

    # 前向填充：同一SKU近7天
    df = df.sort_values(['单品编码', '销售日期'])
    df['批发价格(元/千克)'] = df.groupby('单品编码')['批发价格(元/千克)'].ffill(limit=7)

    matched = df['批发价格(元/千克)'].notna().mean() * 100
    print(f'批发价格覆盖率（含前向填充）: {matched:.1f}%')
    return df

def merge_loss_rates(df_merged: pd.DataFrame, df_loss_item: pd.DataFrame, df_loss_category: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 合并损耗率数据 ===')
    df = df_merged.copy()

    # 单品级
    if not df_loss_item.empty:
        dli = df_loss_item.copy()
        dli['单品编码'] = dli['单品编码'].astype(str)
        dli['损耗率(%)'] = pd.to_numeric(dli['损耗率(%)'], errors='coerce')
        df = pd.merge(df, dli[['单品编码', '损耗率(%)']], on='单品编码', how='left')
    else:
        df['损耗率(%)'] = np.nan

    # 品类级（用于填充缺失）
    if not df_loss_category.empty:
        loss_rate_col = [col for col in df_loss_category.columns if '损耗率' in col and '%' in col]
        if loss_rate_col:
            loss_rate_col = loss_rate_col[0]
            dlc = df_loss_category.copy()
            dlc[loss_rate_col] = pd.to_numeric(dlc[loss_rate_col], errors='coerce')
            df = pd.merge(
                df,
                dlc[['小分类编码', loss_rate_col]].rename(columns={loss_rate_col: '品类损耗率(%)'}),
                left_on='分类编码', right_on='小分类编码', how='left'
            )
            df['损耗率(%)'] = df['损耗率(%)'].fillna(df['品类损耗率(%)'])
            df = df.drop(['小分类编码', '品类损耗率(%)'], axis=1, errors='ignore')

    # 兜底填充
    df['损耗率(%)'] = df['损耗率(%)'].fillna(df.groupby('分类名称')['损耗率(%)'].transform('mean'))
    df['损耗率(%)'] = df['损耗率(%)'].fillna(df['损耗率(%)'].mean())
    df['损耗率'] = df['损耗率(%)'] / 100

    covered = df['损耗率'].notna().mean() * 100
    print(f'损耗率覆盖率: {covered:.1f}%')
    return df

# ---------------------- 指标计算与聚合 ----------------------

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 计算关键指标 ===')
    df = df.copy()
    df['加成率'] = np.where(
        df['批发价格(元/千克)'] > 0,
        (df['销售单价(元/千克)'] - df['批发价格(元/千克)']) / df['批发价格(元/千克)'],
        np.nan
    )
    df['销售额(元)'] = df['销量(千克)'] * df['销售单价(元/千克)']
    df['成本额(元)'] = df['销量(千克)'] * df['批发价格(元/千克)']
    df['毛利(元)'] = df['销售额(元)'] - df['成本额(元)']
    return df

def aggregate_by_category_date(df: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 按品类-日期聚合数据 ===')
    agg = {
        '销量(千克)': 'sum',
        '销售额(元)': 'sum',
        '成本额(元)': 'sum',
        '毛利(元)': 'sum',
        '销售单价(元/千克)': 'mean',
        '批发价格(元/千克)': 'mean',
        '加成率': 'mean',
        '损耗率': 'mean',
        '是否打折销售': lambda x: (x == '是').mean()
    }
    cat = df.groupby(['销售日期', '分类编码', '分类名称']).agg(agg).reset_index()
    cat = cat.rename(columns={
        '销量(千克)': '销售总量_kg',
        '销售单价(元/千克)': '平均销售单价',
        '批发价格(元/千克)': '平均批发价格',
        '加成率': '平均加成率',
        '损耗率': '平均损耗率',
        '是否打折销售': '打折销售比例'
    })
    # 时间特征
    cat['销售日期'] = pd.to_datetime(cat['销售日期'])
    cat['年'] = cat['销售日期'].dt.year
    cat['月'] = cat['销售日期'].dt.month
    cat['日'] = cat['销售日期'].dt.day
    cat['星期'] = cat['销售日期'].dt.dayofweek
    cat['是否周末'] = cat['星期'].isin([5, 6])
    # 补货量
    cat['补货量_kg'] = np.where(cat['平均损耗率'] < 1,
                          cat['销售总量_kg'] / (1 - cat['平均损耗率']),
                          cat['销售总量_kg'] * 1.1)
    print(f'聚合完成: {len(cat)} 条记录')
    return cat

# ---------------------- 优化版Excel输出 ----------------------

def write_optimized_outputs(df_category_daily: pd.DataFrame, df_detailed_sample: pd.DataFrame, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    # 保存核心CSV
    core_file = os.path.join(output_path, 'category_daily_sales.csv')
    df_category_daily.to_csv(core_file, index=False, encoding='utf-8-sig')

    # 创建优化版Excel
    xlsx_path = os.path.join(output_path, 'problem2_data_optimized.xlsx')

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # 数据概览
        summary_rows = []
        summary_rows.append(['数据集概览', '', '', ''])
        summary_rows.append(['指标', '数值', '说明', ''])
        summary_rows.append(['记录数', len(df_category_daily), '品类-日级别', ''])
        summary_rows.append(['时间范围', f"{df_category_daily['销售日期'].min().date()} 到 {df_category_daily['销售日期'].max().date()}", '销售日期', ''])
        summary_rows.append(['品类数量', df_category_daily['分类名称'].nunique(), '唯一品类数', ''])
        summary_rows.append(['', '', '', ''])
        summary_rows.append(['品类统计', '', '', ''])
        summary_rows.append(['品类名称', '总销量(kg)', '平均日销量(kg)', '平均加成率'])
        cstats = df_category_daily.groupby('分类名称').agg({'销售总量_kg': ['sum', 'mean'], '平均加成率': 'mean'}).round(3)
        for cat in cstats.index:
            total_sales = cstats.loc[cat, ('销售总量_kg', 'sum')]
            avg_sales = cstats.loc[cat, ('销售总量_kg', 'mean')]
            avg_markup = cstats.loc[cat, ('平均加成率', 'mean')]
            summary_rows.append([cat, total_sales, avg_sales, avg_markup])
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='数据概览', index=False, header=False)

        # 品类-日聚合（完整）
        df_category_daily.to_excel(writer, sheet_name='品类日销量_聚合', index=False)

        # 销售明细样本（如存在）
        if df_detailed_sample is not None and not df_detailed_sample.empty:
            df_detailed_sample.to_excel(writer, sheet_name='销售明细_样本', index=False)

        # 最近30天
        recent_end = df_category_daily['销售日期'].max()
        recent_start = recent_end - pd.Timedelta(days=29)
        df_recent = df_category_daily[df_category_daily['销售日期'] >= recent_start].copy()
        df_recent.to_excel(writer, sheet_name='最近30天数据', index=False)

    print(f"已生成: {xlsx_path}")
    print(f"已生成: {core_file}")

# ---------------------- 主流程 ----------------------

def prepare_and_write_optimized_only():
    print('=' * 60)
    print('开始生成问题2的优化版输出（只保留CSV与优化版Excel）')
    print('=' * 60)
    t0 = time.time()

    # 1) 加载
    df_info, df_sales, df_wh, df_loss_item, df_loss_category = load_all_data(BASE_PATH)

    # 2) 清洗与合并
    df_sales_clean = clean_sales_data(df_sales)
    df_with_info = merge_product_info(df_sales_clean, df_info)
    df_with_wholesale = merge_wholesale_prices(df_with_info, df_wh)
    df_with_loss = merge_loss_rates(df_with_wholesale, df_loss_item, df_loss_category)

    # 3) 计算指标
    df_metrics = calculate_metrics(df_with_loss)

    # 4) 聚合（用于核心CSV与优化版Excel）
    df_category_daily = aggregate_by_category_date(df_metrics)

    # 5) 生成一个采样版的销售明细（用于Excel查看，避免超大体积）
    df_sample = None
    try:
        if len(df_metrics) > 50000:
            sampled_list = []
            for cat in df_metrics['分类名称'].unique():
                part = df_metrics[df_metrics['分类名称'] == cat]
                if len(part) > 8000:
                    sampled_list.append(part.sample(n=8000, random_state=42))
                else:
                    sampled_list.append(part)
            df_sample = pd.concat(sampled_list, ignore_index=True)
            df_sample = df_sample.sort_values(['销售日期', '分类名称'])
        else:
            df_sample = df_metrics.copy()
    except Exception as e:
        print(f'采样明细时出错（将跳过样本明细）: {e}')
        df_sample = None

    # 6) 写出优化版结果
    write_optimized_outputs(df_category_daily, df_sample, OUTPUT_PATH)

    print(f"总耗时: {time.time() - t0:.2f} 秒")

if __name__ == '__main__':
    prepare_and_write_optimized_only()
