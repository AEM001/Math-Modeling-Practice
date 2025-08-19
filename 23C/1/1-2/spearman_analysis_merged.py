import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SpearmanAnalysis:
    """斯皮尔曼相关性分析类"""
    
    def __init__(self, data_path='../daily_item_sales_summary.csv'):
        self.data_path = data_path
        self.raw_data = None
        self.pivot_data = None
        self.category_pivot = None
        self.product_clusters = None
        self.results = {}
        
    def load_and_preprocess(self):
        """加载和预处理数据"""
        print("正在加载和预处理数据...")
        
        # 加载数据
        self.raw_data = pd.read_csv(self.data_path, encoding='utf-8-sig')
        self.raw_data['销售日期'] = pd.to_datetime(self.raw_data['销售日期'])
        
        # 数据清洗：删除无效数据
        valid_mask = (
            self.raw_data['单品销量(天)'].notna() & 
            (self.raw_data['单品销量(天)'] >= 0)
        )
        clean_data = self.raw_data[valid_mask].copy()
        
        # 创建透视表
        self.pivot_data = clean_data.pivot_table(
            index='销售日期', columns='单品名称', 
            values='单品销量(天)', aggfunc='sum'
        )
        
        self.category_pivot = clean_data.groupby(['销售日期', '分类名称'])['单品销量(天)'].sum().unstack(fill_value=0)
        
        print(f"数据时间范围：{self.category_pivot.index.min()} 到 {self.category_pivot.index.max()}")
        print(f"有效天数：{len(self.category_pivot)}")
        print(f"分类：{list(self.category_pivot.columns)}")
        print(f"单品数量：{self.pivot_data.shape[1]}")
        
    def cluster_products(self, max_k=20, plot_elbow=True):
        """使用K-Means对单品进行聚类"""
        print("\n正在对单品进行聚类分析...")
        
        # 1. 数据准备
        product_sales = self.pivot_data.fillna(0).T  # 转置并填充NaN
        
        # 筛选掉总销量为0的商品
        product_sales = product_sales[product_sales.sum(axis=1) > 0]
        
        if product_sales.empty:
            print("没有可供聚类的有效单品数据。")
            return

        # 2. 特征缩放
        scaler = StandardScaler()
        sales_scaled = scaler.fit_transform(product_sales)
        
        # 3. 使用肘部法则确定最佳K值
        print("正在使用肘部法则确定最佳K值...")
        inertia = []
        k_range = range(2, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(sales_scaled)
            inertia.append(kmeans.inertia_)
            
        # 寻找肘部点
        # 计算二阶导数（变化率的变化率）来找拐点
        try:
            deltas = np.diff(inertia, 2)
            best_k = k_range[np.argmax(deltas) + 1] # +1因为diff会减少一个点
        except (ValueError, IndexError):
            # 如果点数太少无法计算二阶导数，就选择一个默认值
            best_k = 5
            print("无法自动确定最佳K值，将使用默认值 K=5")

        print(f"肘部法则建议的最佳K值为: {best_k}")

        if plot_elbow:
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertia, 'bo-')
            plt.xlabel('聚类数量 (k)')
            plt.ylabel('簇内平方和 (Inertia)')
            plt.title('K-Means聚类肘部法则示意图')
            plt.axvline(x=best_k, color='r', linestyle='--', label=f'最佳 K = {best_k}')
            plt.legend()
            plt.grid(True)
            elbow_path = 'kmeans_elbow_plot.png'
            plt.savefig(elbow_path, dpi=300)
            print(f"肘部法则图已保存为: {elbow_path}")

        # 4. 执行K-Means聚类
        print(f"正在使用 K={best_k} 进行最终聚类...")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(sales_scaled)
        
        # 5. 保存并展示结果
        self.product_clusters = pd.DataFrame({'单品名称': product_sales.index, '簇标签': clusters})
        
        print("\n聚类结果摘要:")
        for i in range(best_k):
            cluster_items = self.product_clusters[self.product_clusters['簇标签'] == i]['单品名称'].tolist()
            print(f"--- 簇 {i} (共 {len(cluster_items)} 个单品) ---")
            # 为了简洁，只显示前10个
            print(", ".join(cluster_items[:10]) + (',...' if len(cluster_items) > 10 else ''))
            
        # 可视化每个簇的平均销售模式
        cluster_centers_scaled = kmeans.cluster_centers_
        
        plt.figure(figsize=(14, 8))
        for i in range(best_k):
            # 反标准化以获得可解释的模式
            center_original_scale = scaler.inverse_transform(cluster_centers_scaled[i].reshape(1, -1))
            plt.plot(product_sales.columns, center_original_scale.flatten(), label=f'簇 {i} 平均销售模式')
        plt.title('各簇的平均每日销售量模式')
        plt.xlabel('日期')
        plt.ylabel('平均销售量 (标准化反转后)')
        plt.legend()
        plt.grid(True)
        cluster_pattern_path = 'cluster_sales_patterns.png'
        plt.savefig(cluster_pattern_path, dpi=300)
        print(f"\n各簇销售模式图已保存为: {cluster_pattern_path}")
        
    def calculate_spearman(self, x, y):
        """
        使用SciPy计算斯皮尔曼相关系数
        """
        # 删除无效数据
        valid_mask = ~(pd.isna(x) | pd.isna(y))
        x_clean = x[valid_mask].values
        y_clean = y[valid_mask].values
        
        n = len(x_clean)
        if n < 3:
            return np.nan, np.nan, n
        
        # 直接使用SciPy进行计算
        r_s, p_value = stats.spearmanr(x_clean, y_clean)
        
        return r_s, p_value, n
    
    def analyze_correlations(self, data_type='categories', min_samples=30, items_to_analyze=None):

        if data_type == 'categories':
            data = self.category_pivot
            items = list(data.columns)
            print(f"\n正在分析分类相关性...")
        else: # products
            data = self.pivot_data
            if items_to_analyze is not None:
                items = items_to_analyze
                # print(f"\n正在分析指定的 {len(items)} 个单品...") # 在循环中打印，避免重复
            else:
                # 筛选有足够数据的单品
                items = [col for col in data.columns if data[col].notna().sum() >= min_samples]
                print(f"\n正在分析单品相关性（最小样本数：{min_samples}）...")
                print(f"符合条件的单品数量：{len(items)}")
        
        results = []
        
        if len(items) < 2:
            # print("不足两个项目，无法进行相关性分析。") # 静默处理
            return pd.DataFrame()

        total_pairs = len(list(combinations(items, 2)))
        
        for i, (item1, item2) in enumerate(combinations(items, 2)):
            # if i > 0 and i % 1000 == 0:
            #     print(f"已处理 {i}/{total_pairs} 对...") # 在簇内分析时信息过多
            
            x, y = data[item1], data[item2]
            r_s, p_value, n = self.calculate_spearman(x, y)
            
            # 判断显著性和关系类型
            significant = p_value < 0.05 if not pd.isna(p_value) else False
            if significant:
                relationship = '正相关' if r_s > 0 else '负相关'
            else:
                relationship = '无显著关联'
            
            results.append({
                f'{data_type[:-1]}1': item1,
                f'{data_type[:-1]}2': item2,
                '样本数': n,
                '相关系数': r_s,
                'p值': p_value,
                '显著性': 'Yes' if significant else 'No',
                '关系类型': relationship
            })
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('相关系数', key=abs, ascending=False)
        
        # 不在此处设置 self.results, 在主流程中进行
        significant_count = len(results_df[results_df['显著性'] == 'Yes'])
        # print(f"发现 {significant_count} 对显著相关的{data_type}") # 在簇循环中打印
        
        return results_df
    
    def demonstrate_calculation(self, cat1='花叶类', cat2='辣椒类'):
        """演示具体计算过程"""
        print(f"\n=== 演示计算：{cat1} vs {cat2} ===")
        
        x = self.category_pivot[cat1]
        y = self.category_pivot[cat2]
        
        r_s, p_value, n = self.calculate_spearman(x, y)
        
        print(f"有效样本数 n = {n}")
        print(f"斯皮尔曼相关系数 r_s = {r_s:.6f}")
        print(f"p值 = {p_value:.2e}")
        print(f"显著性水平 α = 0.05")
        
        if p_value < 0.05:
            if r_s > 0:
                print("结论：存在显著正相关，消费者有同时购买意愿或上架销售存在关联")
            else:
                print("结论：存在显著负相关，消费者存在替代性选择意愿或销售时节相反")
        else:
            print("结论：不存在显著关联")
        
        return r_s, p_value, n
    
    def plot_correlation_heatmap(self, results_df, save_path='correlation_heatmap.png', title_prefix='分类'):
        """绘制相关性热力图"""
        if results_df.empty:
            print(f"{title_prefix}无相关性结果可供绘图。")
            return

        # 从结果中动态获取项目列表
        key1 = f'{results_df.columns[0]}'
        key2 = f'{results_df.columns[1]}'
        items1 = results_df[key1].unique()
        items2 = results_df[key2].unique()
        items = sorted(list(set(items1) | set(items2)))
        
        n_items = len(items)
        if n_items < 2:
            return # 无法绘制热力图

        corr_matrix = pd.DataFrame(np.eye(n_items), index=items, columns=items)
        
        # 填充相关系数矩阵
        for _, row in results_df.iterrows():
            item1, item2 = row[key1], row[key2]
            r_s = row['相关系数']
            if item1 in corr_matrix.index and item2 in corr_matrix.columns:
                corr_matrix.loc[item1, item2] = r_s
                corr_matrix.loc[item2, item1] = r_s
        
        # 绘制热力图
        plt.figure(figsize=(max(10, n_items*0.6), max(8, n_items*0.6)))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.2f',
                   square=True,
                   annot_kws={"size": 8})
        plt.title(f'{title_prefix}间斯皮尔曼相关系数热力图', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存为：{save_path}")
        
    def generate_summary_report(self, output_file='spearman_analysis_summary.txt'):
        """生成简化分析报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("斯皮尔曼相关性分析总结报告\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("1. 数据概况\n")
            f.write(f"   时间范围：{self.category_pivot.index.min().date()} 到 {self.category_pivot.index.max().date()}\n")
            f.write(f"   有效天数：{len(self.category_pivot)}\n")
            f.write(f"   分类数量：{len(self.category_pivot.columns)}\n")
            f.write(f"   单品数量：{self.pivot_data.shape[1]}\n\n")

            if self.product_clusters is not None:
                f.write("2. 单品聚类分析总结\n")
                num_clusters = self.product_clusters['簇标签'].max() + 1
                f.write(f"   通过K-Means算法将 {len(self.product_clusters)} 个单品聚为 {num_clusters} 个簇。\n")
                f.write("   每个簇代表一种独特的销售时间模式，为后续的关联分析提供了基础。\n\n")

            if 'categories' in self.results:
                cat_results = self.results['categories']
                significant_cats = cat_results[cat_results['显著性'] == 'Yes']
                f.write("3. 分类相关性分析\n")
                f.write(f"   总分析对数：{len(cat_results)}\n")
                f.write(f"   显著相关对数：{len(significant_cats)}\n")
                pos_corr = len(significant_cats[significant_cats['关系类型'] == '正相关'])
                neg_corr = len(significant_cats[significant_cats['关系类型'] == '负相关'])
                f.write(f"   正相关：{pos_corr}对，负相关：{neg_corr}对\n\n")
                f.write("   前5个最强相关：\n")
                for _, row in significant_cats.head().iterrows():
                    f.write(f"   - {row['categorie1']} vs {row['categorie2']}: r_s={row['相关系数']:.4f} ({row['关系类型']})\n")
                f.write("\n")
            
            if 'products' in self.results:
                prod_results = self.results['products']
                significant_prods = prod_results[prod_results['显著性'] == 'Yes']
                f.write("4. 基于聚类的单品相关性分析\n")
                f.write(f"   对各簇内单品进行分析，总共发现 {len(significant_prods)} 对显著关系。\n")
                if not significant_prods.empty:
                    f.write("   最强的10个显著关系如下：\n")
                    for _, row in significant_prods.head(10).iterrows():
                        f.write(f"   - (簇 {row['簇']}) {row['product1']} vs {row['product2']}: r_s={row['相关系数']:.4f} ({row['关系类型']})\n")
                f.write("\n")
            
            f.write("5. 主要结论\n")
            f.write("   - 采用‘先聚类，后关联’的策略，有效解决了单品数量庞大导致分析困难的问题。\n")
            f.write("   - 品类层面：多数品类呈现显著正相关，反映了顾客购买的协同效应。\n")
            f.write("   - 单品层面：通过聚类识别出不同销售模式的商品群，并在簇内发现了更细致的互补或替代关系，为精细化运营提供了数据支持。\n")
        
        print(f"总结报告已保存为：{output_file}")
    
    def save_results(self, prefix='spearman_results'):
        """保存分析结果"""
        for data_type, df in self.results.items():
            filename = f"{prefix}_{data_type}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"{data_type}结果已保存为：{filename}")

def main():
    """主函数 - 完整分析流程"""
    analyzer = SpearmanAnalysis(data_path='../daily_item_sales_summary.csv')
    
    # 数据加载和预处理
    analyzer.load_and_preprocess()
    
    # --- 分类分析 ---
    print("\n--- 步骤1: 分类相关性分析 ---")
    analyzer.demonstrate_calculation('花叶类', '辣椒类')
    category_results = analyzer.analyze_correlations('categories')
    analyzer.results['categories'] = category_results # 保存结果
    print("\n分类相关性分析结果：")
    print(category_results[category_results['显著性'] == 'Yes'].to_string(index=False, float_format='%.4f'))
    analyzer.plot_correlation_heatmap(category_results, save_path='category_correlation_heatmap.png', title_prefix='分类')

    # --- 单品聚类与分析 ---
    print("\n--- 步骤2: 单品聚类分析 ---")
    analyzer.cluster_products(max_k=15) # 设定一个合适的max_k

    if analyzer.product_clusters is not None:
        print("\n--- 步骤3: 基于聚类的单品相关性分析 ---")
        # 对每个簇进行内部相关性分析
        all_product_results = []
        num_clusters = analyzer.product_clusters['簇标签'].max() + 1
        for i in range(num_clusters):
            cluster_items = analyzer.product_clusters[analyzer.product_clusters['簇标签'] == i]['单品名称'].tolist()
            print(f"\n正在分析 簇 {i} (包含 {len(cluster_items)} 个单品)...", end='')
            
            # 对簇内所有商品进行分析
            items_to_run = cluster_items

            if len(items_to_run) < 2:
                print(" -> 单品少于2个，跳过分析。")
                continue

            cluster_results_df = analyzer.analyze_correlations(
                data_type='products', 
                items_to_analyze=items_to_run
            )
            
            if not cluster_results_df.empty:
                significant_count = len(cluster_results_df[cluster_results_df['显著性'] == 'Yes'])
                print(f" -> 完成，发现 {significant_count} 对显著关系。")
                cluster_results_df['簇'] = i
                all_product_results.append(cluster_results_df)
                
                # 为有显著关系的簇绘制热力图
                if significant_count > 0:
                    analyzer.plot_correlation_heatmap(
                        cluster_results_df[cluster_results_df['显著性'] == 'Yes'],
                        save_path=f'product_cluster_{i}_heatmap.png',
                        title_prefix=f'簇 {i} 内单品'
                    )
            else:
                print(" -> 完成，未发现关系。")

        if all_product_results:
            # 合并所有簇的分析结果
            final_product_results = pd.concat(all_product_results, ignore_index=True)
            final_product_results = final_product_results.sort_values('相关系数', key=abs, ascending=False)
            analyzer.results['products'] = final_product_results
            print("\n--- 所有簇内部分析完成 ---")
            print("最显著相关的单品对 (Top 10):")
            print(final_product_results[final_product_results['显著性']=='Yes'].head(10).to_string(index=False))
        else:
            print("\n未发现任何显著的簇内单品关系。")

    # --- 生成最终报告和结果 ---
    print("\n--- 步骤4: 生成报告与保存结果 ---")
    analyzer.generate_summary_report()
    analyzer.save_results()
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()