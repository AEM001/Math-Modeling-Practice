import pandas as pd
import numpy as np
import os
import time
import re
from datetime import datetime, timedelta

BASE_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C'
OUTPUT_PATH = '/Users/Mac/Downloads/Math-Modeling-Practice/23C/3'

# ---------- 通用工具 ----------

def clean_product_name(name: str) -> str:
    """去除来源编号：如 '小青菜(1)' -> '小青菜'"""
    if pd.isna(name):
        return name
    return re.sub(r'\(\d+\)$', '', str(name)).strip()


def load_all_data(base_path: str):
    print('=== 加载附件数据 ===')
    df_info = pd.read_excel(os.path.join(base_path, '附件1.xlsx'))
    df_sales = pd.read_excel(os.path.join(base_path, '附件2.xlsx'))
    df_wholesale = pd.read_excel(os.path.join(base_path, '附件3.xlsx'))
    # 附件4：单品损耗率在 Sheet1，小分类平均损耗率在另一个sheet
    df_loss_item = pd.read_excel(os.path.join(base_path, '附件4.xlsx'), sheet_name='Sheet1')
    xls = pd.ExcelFile(os.path.join(base_path, '附件4.xlsx'))
    loss_sheet = [s for s in xls.sheet_names if '平均损耗率' in s]
    df_loss_cat = pd.read_excel(os.path.join(base_path, '附件4.xlsx'), sheet_name=loss_sheet[0]) if loss_sheet else pd.DataFrame()
    return df_info, df_sales, df_wholesale, df_loss_item, df_loss_cat


def clean_sales(df_sales: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 清洗销售流水 ===')
    df = df_sales.copy()
    df['销售日期'] = pd.to_datetime(df['销售日期'], errors='coerce')
    df['销量(千克)'] = pd.to_numeric(df['销量(千克)'], errors='coerce')
    df['销售单价(元/千克)'] = pd.to_numeric(df['销售单价(元/千克)'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['销售日期'])
    df = df.dropna(subset=['销量(千克)', '销售单价(元/千克)'])
    df = df[df['销售单价(元/千克)'] > 0]
    df = df[df['销量(千克)'] > 0]
    print(f'有效记录: {len(df)}/{before}')
    return df


def merge_info(df_sales: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 合并商品信息 ===')
    df_sales = df_sales.copy()
    df_info = df_info.copy()
    df_sales['单品编码'] = df_sales['单品编码'].astype(str)
    df_info['单品编码'] = df_info['单品编码'].astype(str)
    out = pd.merge(df_sales, df_info, on='单品编码', how='left')
    missing = out['分类名称'].isna().sum()
    if missing:
        print(f'警告：有 {missing} 条记录缺少商品信息，将被剔除。')
        out = out.dropna(subset=['分类名称'])
    # 清理来源编号
    out['清理后单品名称'] = out['单品名称'].apply(clean_product_name)
    return out


def merge_wholesale(df: pd.DataFrame, df_wholesale: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 合并批发价格 ===')
    df = df.copy()
    df_wh = df_wholesale.copy()
    df_wh['日期'] = pd.to_datetime(df_wh['日期'], errors='coerce')
    df_wh['批发价格(元/千克)'] = pd.to_numeric(df_wh['批发价格(元/千克)'], errors='coerce')
    df_wh = df_wh.dropna(subset=['日期', '批发价格(元/千克)'])
    df_wh = df_wh[df_wh['批发价格(元/千克)'] > 0]
    df_wh['单品编码'] = df_wh['单品编码'].astype(str)
    df = pd.merge(
        df,
        df_wh.rename(columns={'日期': '销售日期'})[['销售日期', '单品编码', '批发价格(元/千克)']],
        on=['销售日期', '单品编码'], how='left'
    )
    # 前向填充同一单品近期价格（7天窗口）
    df = df.sort_values(['单品编码', '销售日期'])
    df['批发价格(元/千克)'] = df.groupby('单品编码')['批发价格(元/千克)'].ffill(limit=7)
    return df


def merge_loss(df: pd.DataFrame, df_loss_item: pd.DataFrame, df_loss_cat: pd.DataFrame) -> pd.DataFrame:
    print('\n=== 合并损耗率 ===')
    df = df.copy()
    if not df_loss_item.empty:
        dli = df_loss_item.copy()
        dli['单品编码'] = dli['单品编码'].astype(str)
        dli['损耗率(%)'] = pd.to_numeric(dli['损耗率(%)'], errors='coerce')
        df = pd.merge(df, dli[['单品编码', '损耗率(%)']], on='单品编码', how='left')
    else:
        df['损耗率(%)'] = np.nan
    if not df_loss_cat.empty:
        # 识别损耗率列名
        loss_col = [c for c in df_loss_cat.columns if '损耗率' in c and '%' in c]
        if loss_col:
            loss_col = loss_col[0]
            dlc = df_loss_cat.copy()
            dlc[loss_col] = pd.to_numeric(dlc[loss_col], errors='coerce')
            df = pd.merge(df, dlc[['小分类编码', loss_col]].rename(columns={loss_col: '品类损耗率(%)'}),
                          left_on='分类编码', right_on='小分类编码', how='left')
            df['损耗率(%)'] = df['损耗率(%)'].fillna(df['品类损耗率(%)'])
            df = df.drop(columns=['小分类编码', '品类损耗率(%)'], errors='ignore')
    # 填充剩余缺失
    df['损耗率(%)'] = df['损耗率(%)'].fillna(df.groupby('分类名称')['损耗率(%)'].transform('mean'))
    df['损耗率(%)'] = df['损耗率(%)'].fillna(df['损耗率(%)'].mean())
    df['损耗率'] = df['损耗率(%)'] / 100
    return df


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['加成率'] = np.where(df['批发价格(元/千克)'] > 0,
                        (df['销售单价(元/千克)'] - df['批发价格(元/千克)']) / df['批发价格(元/千克)'],
                        np.nan)
    df['销售额(元)'] = df['销量(千克)'] * df['销售单价(元/千克)']
    df['成本额(元)'] = df['销量(千克)'] * df['批发价格(元/千克)']
    df['毛利(元)'] = df['销售额(元)'] - df['成本额(元)']
    df['是否打折销售'] = df['是否打折销售'].map({'是': 1, '否': 0}).fillna(0)
    return df

# ---------- 问题3特定准备 ----------

def build_week_window_dataset(df_all: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """提取指定周间隔内的销售记录（含0销量填充逻辑的支持）"""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (df_all['销售日期'] >= start_dt) & (df_all['销售日期'] <= end_dt)
    df_week = df_all.loc[mask].copy()
    # 可售单品集合：该周出现过销售的SKU
    sku_set = sorted(df_week['单品编码'].astype(str).unique().tolist())
    print(f'可售单品（SKU）数量: {len(sku_set)}')
    return df_week, sku_set


def compute_item_week_stats(df_week: pd.DataFrame) -> pd.DataFrame:
    """计算SKU级周内统计特征，并构造0701价格成本基线"""
    # 日维度透视（缺失天作为0销量）
    date_index = pd.date_range(df_week['销售日期'].min(), df_week['销售日期'].max(), freq='D')
    key_cols = ['销售日期', '单品编码', '单品名称', '清理后单品名称', '分类编码', '分类名称']
    df_daily = (df_week
                .groupby(key_cols, observed=True)['销量(千克)']
                .sum()
                .reset_index())
    # 为每个SKU补全日期
    # 构造所有组合
    sku_info = df_daily[['单品编码', '单品名称', '清理后单品名称', '分类编码', '分类名称']].drop_duplicates()
    all_idx = sku_info.assign(tmp=1).merge(pd.DataFrame({'销售日期': date_index, 'tmp': 1}), on='tmp')
    df_daily_full = all_idx.merge(df_daily, on=['销售日期', '单品编码', '单品名称', '清理后单品名称', '分类编码', '分类名称'], how='left')
    df_daily_full['销量(千克)'] = df_daily_full['销量(千克)'].fillna(0.0)

    # 周统计
    grp = df_daily_full.groupby(['单品编码', '单品名称', '清理后单品名称', '分类编码', '分类名称'], observed=True)
    stats = grp['销量(千克)'].agg(
        周总销量_kg='sum', 周日均销量_kg='mean', 周销量标准差='std'
    ).reset_index()
    stats['周销量变异系数'] = np.where(stats['周日均销量_kg']>0, stats['周销量标准差']/stats['周日均销量_kg'], np.nan)

    # 线性趋势（按日期序号）
    df_daily_full = df_daily_full.sort_values(['单品编码', '销售日期'])
    df_daily_full['day_idx'] = (df_daily_full['销售日期'] - df_daily_full['销售日期'].min()).dt.days
    def slope(g):
        x = g['day_idx'].values
        y = g['销量(千克)'].values
        if len(x) >= 2:
            b1 = np.polyfit(x, y, 1)[0]
        else:
            b1 = 0.0
        return b1
    trend = df_daily_full.groupby(['单品编码'], observed=True).apply(slope).reset_index(name='周销量趋势_斜率')
    stats = stats.merge(trend, on='单品编码', how='left')

    # 价格与成本（周内均值）
    price_cost = (df_week.groupby(['单品编码'], observed=True)
                  .agg(周均销售单价=('销售单价(元/千克)', 'mean'),
                       周均批发价=('批发价格(元/千克)', 'mean'),
                       周均加成率=('加成率', 'mean'),
                       打折销售比例=('是否打折销售', 'mean'),
                       单品损耗率=('损耗率', 'mean'))
                  .reset_index())
    stats = stats.merge(price_cost, on='单品编码', how='left')

    # 最近一天销量（6/30）
    last_day = df_daily_full['销售日期'].max()
    last_sales = (df_daily_full[df_daily_full['销售日期']==last_day]
                  [['单品编码', '销量(千克)']]
                  .rename(columns={'销量(千克)':'最近一天销量_kg'}))
    stats = stats.merge(last_sales, on='单品编码', how='left')

    # 类内份额（基于清理后单品名称以合并来源）
    cat_week = (df_daily_full.groupby(['分类名称'], observed=True)['销量(千克)'].sum().reset_index()
                .rename(columns={'销量(千克)':'品类周总销量_kg'}))
    item_week = (df_daily_full.groupby(['清理后单品名称','分类名称'], observed=True)['销量(千克)'].sum().reset_index()
                 .rename(columns={'销量(千克)':'单品周总销量_kg_合并来源'}))
    share = item_week.merge(cat_week, on='分类名称', how='left')
    share['单品类内份额'] = np.where(share['品类周总销量_kg']>0,
                                   share['单品周总销量_kg_合并来源']/share['品类周总销量_kg'], 0)
    stats = stats.merge(share[['清理后单品名称','分类名称','单品类内份额']],
                        on=['清理后单品名称','分类名称'], how='left')

    # 最小陈列量
    stats['最小陈列量_kg'] = 2.5

    return stats


def build_category_targets(df_all: pd.DataFrame) -> pd.DataFrame:
    """构建2023-07-01的品类目标需求（特征），不做最终预测，仅提供参考特征"""
    # 历史按星期的季节性因子（品类）
    df = df_all.copy()
    df['星期'] = df['销售日期'].dt.dayofweek
    cat_daily = df.groupby(['销售日期','分类编码','分类名称'], observed=True)['销量(千克)'].sum().reset_index()
    cat_daily['星期'] = cat_daily['销售日期'].dt.dayofweek
    cat_mean = cat_daily.groupby(['分类名称'], observed=True)['销量(千克)'].mean().reset_index(name='历史日均销量_kg')
    cat_dow = cat_daily.groupby(['分类名称','星期'], observed=True)['销量(千克)'].mean().reset_index(name='按星期均销量_kg')
    cat = cat_dow.merge(cat_mean, on='分类名称', how='left')
    cat['星期因子'] = np.where(cat['历史日均销量_kg']>0, cat['按星期均销量_kg']/cat['历史日均销量_kg'], 1.0)

    # 最近1周（0624-0630）的品类均量
    recent_start = pd.to_datetime('2023-06-24')
    recent_end = pd.to_datetime('2023-06-30')
    mask = (df['销售日期']>=recent_start) & (df['销售日期']<=recent_end)
    recent = (df.loc[mask]
              .groupby(['分类编码','分类名称'], observed=True)['销量(千克)']
              .mean().reset_index(name='近7天日均销量_kg'))

    # 2023-07-01是周六，星期=5
    sat_factor = cat[cat['星期']==5][['分类名称','星期因子']].rename(columns={'星期因子':'周六因子'})
    target = recent.merge(sat_factor, on='分类名称', how='left')
    target['建议目标销量_kg_0701'] = target['近7天日均销量_kg'] * target['周六因子']

    return target


def prepare_problem3_data():
    print('='*60)
    print('开始准备问题3数据（不求解，只准备特征与输入）')
    print('='*60)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1) 加载并整合
    df_info, df_sales, df_wh, df_loss_item, df_loss_cat = load_all_data(BASE_PATH)
    df_sales = clean_sales(df_sales)
    df = merge_info(df_sales, df_info)
    df = merge_wholesale(df, df_wh)
    df = merge_loss(df, df_loss_item, df_loss_cat)
    df = add_metrics(df)

    # 2) 提取0624-0630窗口
    df_week, sku_set = build_week_window_dataset(df, '2023-06-24', '2023-06-30')

    # 3) 计算SKU级周统计
    item_stats = compute_item_week_stats(df_week)

    # 4) 构建0701的品类目标（仅作为模型输入特征，不是最终预测）
    category_targets = build_category_targets(df)

    # 5) 构建0701的SKU价格与成本基线
    # 获取0701当日的批发价，如缺失则向前7天填充
    date_0701 = pd.to_datetime('2023-07-01')
    df_wh = df_wh.copy()
    df_wh['日期'] = pd.to_datetime(df_wh['日期'], errors='coerce')
    df_wh['单品编码'] = df_wh['单品编码'].astype(str)
    # 取该SKU在 <=0701 的最近价格
    df_wh_sorted = df_wh.sort_values(['单品编码','日期'])
    df_wh_sorted = df_wh_sorted[df_wh_sorted['日期'] <= date_0701]
    df_wh_latest = df_wh_sorted.groupby('单品编码', as_index=False).last()[['单品编码','日期','批发价格(元/千克)']]
    df_wh_latest = df_wh_latest.rename(columns={'日期':'批发价日期_最近','批发价格(元/千克)':'批发价_0701或最近'})

    # 合并到候选SKU
    item_pricing = item_stats.merge(df_wh_latest, on='单品编码', how='left')

    # 补充0701的建议定价基线：采用周内中位加成率 * 最新批发价
    item_pricing['建议加成率_基线'] = item_pricing['周均加成率']
    item_pricing['建议售价_基线(元/千克)'] = np.where(item_pricing['批发价_0701或最近']>0,
                                               item_pricing['批发价_0701或最近'] * (1 + item_pricing['建议加成率_基线']),
                                               np.nan)

    # 仅保留0624-0630出现过的SKU作为可售单品集合
    item_pricing = item_pricing.sort_values(['分类名称','清理后单品名称','单品编码'])

    # 6) 输出文件
    out_xlsx = os.path.join(OUTPUT_PATH, 'problem3_item_prepared.xlsx')
    out_items_csv = os.path.join(OUTPUT_PATH, 'problem3_item_candidates.csv')
    out_cats_csv = os.path.join(OUTPUT_PATH, 'problem3_category_targets_0701.csv')

    # 构建合并来源的版本（按清理后单品名称聚合）
    merged_cols = ['清理后单品名称','分类编码','分类名称']
    item_stats_merged = (item_stats
                          .groupby(merged_cols, observed=True)
                          .agg(周总销量_kg=('周总销量_kg','sum'),
                               周日均销量_kg=('周日均销量_kg','mean'),
                               周销量标准差=('周销量标准差','mean'),
                               周销量变异系数=('周销量变异系数','mean'),
                               周销量趋势_斜率=('周销量趋势_斜率','mean'),
                               周均销售单价=('周均销售单价','mean'),
                               周均批发价=('周均批发价','mean'),
                               周均加成率=('周均加成率','mean'),
                               打折销售比例=('打折销售比例','mean'),
                               单品损耗率=('单品损耗率','mean'),
                               单品类内份额=('单品类内份额','mean'))
                          .reset_index())

    with pd.ExcelWriter(out_xlsx) as writer:
        item_pricing.to_excel(writer, sheet_name='SKU周统计_0624_0630', index=False)
        item_stats_merged.to_excel(writer, sheet_name='合并来源周统计', index=False)
        category_targets.to_excel(writer, sheet_name='类别目标_0701', index=False)

    # 导出CSV便于建模使用
    item_pricing.to_csv(out_items_csv, index=False, encoding='utf-8-sig')
    category_targets.to_csv(out_cats_csv, index=False, encoding='utf-8-sig')

    # 简要打印
    print('\n=== 输出完成 ===')
    print(f'Excel: {out_xlsx}')
    print(f'候选单品CSV: {out_items_csv}')
    print(f'类别目标CSV: {out_cats_csv}')
    print(f"可售SKU数量（0624-0630出现）: {item_pricing['单品编码'].nunique()}")
    print('样例：')
    print(item_pricing.head())


if __name__ == '__main__':
    prepare_problem3_data()