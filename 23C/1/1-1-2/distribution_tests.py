import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '1-1-2'
ITEM_HIST_DIR = os.path.join(OUTPUT_DIR, 'items_hist')
CAT_HIST_DIR = os.path.join(OUTPUT_DIR, 'categories_hist')


def ensure_dirs():
    # 仅确保输出根目录存在；图片目录暂不创建（按需求关闭绘图输出）
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    print('正在加载数据...')
    df = pd.read_csv(csv_path)
    df['销售日期'] = pd.to_datetime(df['销售日期'])
    print(f"数据加载完成，共 {len(df)} 条记录，单品 {df['单品名称'].nunique()} 种，品类 {df['分类名称'].nunique()} 个")
    return df


def compute_daily_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    print('正在按天汇总品类销量...')
    daily_category = (
        df.groupby(['销售日期', '分类名称'], observed=True)['单品销量(天)']
        .sum()
        .reset_index()
        .rename(columns={'单品销量(天)': '品类销量(天)'})
    )
    print(f'生成每日品类销量记录 {len(daily_category)} 条')
    return daily_category


def ks_test_series(values: np.ndarray) -> dict:
    """对一组样本进行三种分布（正态/泊松/伽马）的K-S检验。
    返回：各分布的D、p、是否通过(alpha=0.05)，以及最佳分布。"""
    result = {
        'normal_D': np.nan, 'normal_p': np.nan, 'normal_pass': False,
        'poisson_D': np.nan, 'poisson_p': np.nan, 'poisson_pass': False,
        'gamma_D': np.nan, 'gamma_p': np.nan, 'gamma_pass': False,
        'best_fit': None, 'best_p': np.nan
    }

    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[~np.isnan(clean_values)]
    n = clean_values.size
    if n < 5:
        # 样本太少，不做检验
        return result

    alpha = 0.05

    # 正态分布检验
    mu, sigma = np.mean(clean_values), np.std(clean_values, ddof=1)
    if sigma > 0:
        try:
            Dn, pn = stats.kstest(clean_values, 'norm', args=(mu, sigma))
            result['normal_D'] = float(Dn)
            result['normal_p'] = float(pn)
            result['normal_pass'] = bool(pn >= alpha)
        except Exception:
            pass

    # 泊松分布检验（将数据四舍五入为非负整数）
    int_vals = np.rint(clean_values).astype(int)
    int_vals[int_vals < 0] = 0
    lam = np.mean(int_vals)
    try:
        # 对离散分布，传入CDF函数
        Dp, pp = stats.kstest(int_vals, lambda x: stats.poisson(mu=lam).cdf(x))
        result['poisson_D'] = float(Dp)
        result['poisson_p'] = float(pp)
        result['poisson_pass'] = bool(pp >= alpha)
    except Exception:
        pass

    # 伽马分布检验
    # 伽马定义域为x>0，若存在<=0数据，fit会通过loc处理
    try:
        ag, locg, scaleg = stats.gamma.fit(clean_values, floc=None)
        Dg, pg = stats.kstest(clean_values, 'gamma', args=(ag, locg, scaleg))
        result['gamma_D'] = float(Dg)
        result['gamma_p'] = float(pg)
        result['gamma_pass'] = bool(pg >= alpha)
    except Exception:
        pass

    # 选择最大p值作为最佳分布
    p_map = {
        'normal': result['normal_p'],
        'poisson': result['poisson_p'],
        'gamma': result['gamma_p']
    }
    best = max(p_map.items(), key=lambda kv: (kv[1] if (kv[1] == kv[1]) else -1))  # 处理NaN
    result['best_fit'] = best[0]
    result['best_p'] = float(best[1]) if best[1] == best[1] else np.nan
    return result


def plot_hist_with_fits(values: np.ndarray, title: str, out_path: str):
    
    return


def run_tests_for_group(df: pd.DataFrame, group_col: str, value_col: str, hist_dir: str, title_prefix: str) -> pd.DataFrame:
    records = []
    groups = df[group_col].unique().tolist()
    total = len(groups)
    for idx, g in enumerate(groups, 1):
        sub = df[df[group_col] == g][value_col].values
        res = ks_test_series(sub)
        # 仅输出最佳分布对应的D与p以及是否通过
        best = res['best_fit']
        d_key = f"{best}_D" if best else None
        p_key = f"{best}_p" if best else None
        pass_key = f"{best}_pass" if best else None
        rec = {
            group_col: g,
            '样本数': int(len(sub)),
            '最佳分布': best,
            '最佳D': res.get(d_key, np.nan) if d_key else np.nan,
            '最佳p': res.get(p_key, np.nan) if p_key else np.nan,
            '是否通过(α=0.05)': res.get(pass_key, False) if pass_key else False
        }
        records.append(rec)

        # 绘图已暂停

        if idx % 20 == 0 or idx == total:
            print(f'{title_prefix}: {idx}/{total} 已处理')

    result_df = pd.DataFrame.from_records(records)
    # 排序：按最佳p降序
    if '最佳p' in result_df.columns:
        result_df = result_df.sort_values('最佳p', ascending=False)
    return result_df


def main():
    ensure_dirs()

    csv_path = 'daily_item_sales_summary.csv'
    if not os.path.exists(csv_path):
        print(f'错误：未找到数据文件 {csv_path}')
        return

    df = load_data(csv_path)

    # 单品检验
    print('\n=== 单品分布检验 ===')
    item_results = run_tests_for_group(
        df=df,
        group_col='单品名称',
        value_col='单品销量(天)',
        hist_dir=ITEM_HIST_DIR,
        title_prefix='单品分布'
    )
    item_csv = os.path.join(OUTPUT_DIR, '单品_分布检验结果.csv')
    item_results.to_csv(item_csv, index=False, encoding='utf-8-sig')
    print(f'单品分布检验结果已保存：{item_csv}')

    # 品类按天汇总后检验
    daily_cat = compute_daily_category_sales(df)
    print('\n=== 品类分布检验 ===')
    cat_results = run_tests_for_group(
        df=daily_cat.rename(columns={'品类销量(天)': '值'}),
        group_col='分类名称',
        value_col='值',
        hist_dir=CAT_HIST_DIR,
        title_prefix='品类分布'
    )
    cat_csv = os.path.join(OUTPUT_DIR, '品类_分布检验结果.csv')
    cat_results.to_csv(cat_csv, index=False, encoding='utf-8-sig')
    print(f'品类分布检验结果已保存：{cat_csv}')

    # 生成简单报告
    report_path = os.path.join(OUTPUT_DIR, '分布检验报告.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 1.1.2 分布检验结果\n\n')
        f.write('在显著性水平 0.05 下，对每日销量进行了正态/泊松/伽马的K-S检验；若多种分布均通过，选择P值最大者作为最佳拟合。\n\n')
        f.write('## 单品结果概览（前20条按P值排序）\n\n')
        f.write(item_results.head(20).to_markdown(index=False))
        f.write('\n\n## 品类结果概览（全部）\n\n')
        f.write(cat_results.to_markdown(index=False))
    print(f'报告已生成：{report_path}')


if __name__ == '__main__':
    main()
