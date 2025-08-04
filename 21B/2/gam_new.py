import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from pygam import LinearGAM, s, f, l
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def set_chinese_font():
    """设置中文字体"""
    chinese_fonts = ['PingFang HK', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 成功设置中文字体: {font_name}")
            return
        except:
            continue
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠ 使用默认字体")

class OptimizedGAMAnalyzer:
    """优化的GAM分析器（含交互特征）"""
    
    def __init__(self, data_path, catalyst_info_path):
        """初始化分析器"""
        self.data_path = data_path
        self.catalyst_info_path = catalyst_info_path
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train_conversion = None
        self.y_test_conversion = None
        self.y_train_selectivity = None
        self.y_test_selectivity = None
        # 特征名称将在步骤1中动态生成
        self.feature_names = None
        self.scalers = {}
        self.models = {}
        
    def step1_enhanced_preprocessing(self):
        """步骤1: 数据预处理与交互特征工程"""
        print("=" * 60)
        print("步骤1: 数据预处理与交互特征工程")
        print("=" * 60)
        
        print("1.1 加载和清洗数据...")
        raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        catalyst_info = pd.read_csv(self.catalyst_info_path, encoding='utf-8')
        
        raw_data = raw_data[raw_data['催化剂组合编号'] != 'A11'].copy()
        catalyst_info = catalyst_info[catalyst_info['催化剂组合编号'] != 'A11'].copy()
        self.processed_data = pd.merge(raw_data, catalyst_info, on='催化剂组合编号', how='left')
        
        print("1.2 特征工程...")
        # 提取基础数值特征
        self.processed_data['Co负载量_数值'] = self.processed_data['Co负载量'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['乙醇浓度_数值'] = self.processed_data['乙醇浓度'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['装料质量比'] = self._calculate_loading_ratio()
        
        # 定义基础特征列名
        base_feature_columns = ['温度', 'Co负载量_数值', '装料质量比', '乙醇浓度_数值']
        base_feature_names = ['温度', 'Co负载量', '装料质量比', '乙醇浓度']

        # --- 新增：创建交互特征 ---
        print("1.3 创建交互特征...")
        self.processed_data['温度_x_乙醇浓度'] = self.processed_data['温度'] * self.processed_data['乙醇浓度_数值']
        self.processed_data['温度_x_Co负载量'] = self.processed_data['温度'] * self.processed_data['Co负载量_数值']
        self.processed_data['Co负载量_x_装料比'] = self.processed_data['Co负载量_数值'] * self.processed_data['装料质量比']
        
        interaction_feature_columns = ['温度_x_乙醇浓度', '温度_x_Co负载量', 'Co负载量_x_装料比']
        interaction_feature_names = ['温度*乙醇浓度', '温度*Co负载量', 'Co负载量*装料比']
        print(f"  - 新增交互特征: {', '.join(interaction_feature_names)}")
        
        # --- 更新总的特征列表 ---
        self.feature_names = base_feature_names + interaction_feature_names
        feature_columns = base_feature_columns + interaction_feature_columns

        print("1.4 异常值检测...")
        target_columns = ['乙醇转化率(%)', 'C4烯烃选择性(%)']
        clean_data = self.processed_data.dropna(subset=feature_columns + target_columns).copy()
        
        for col in target_columns:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (clean_data[col] < lower_bound) | (clean_data[col] > upper_bound)
            if outliers.sum() > 0:
                print(f"  发现{col}异常值: {outliers.sum()}个")
                clean_data.loc[outliers, col] = np.clip(clean_data.loc[outliers, col], lower_bound, upper_bound)
        
        # 准备特征矩阵
        self.X = clean_data[feature_columns].values
        self.y_conversion = clean_data['乙醇转化率(%)'].values
        self.y_selectivity = clean_data['C4烯烃选择性(%)'].values
        
        print(f"✓ 数据预处理完成")
        print(f"  - 有效样本数: {len(clean_data)}")
        print(f"  - 特征变量 (共{len(self.feature_names)}个): {', '.join(self.feature_names)}")
        print(f"  - 异常值处理: 已使用IQR方法")
        
        return {'sample_size': len(clean_data), 'features': self.feature_names}
    
    def step2_feature_selection(self):
        """步骤2: 特征选择和数据分割"""
        print("\n" + "=" * 60)
        print("步骤2: 特征选择和数据分割")
        print("=" * 60)
        
        print("2.1 特征重要性评估...")
        selector = SelectKBest(score_func=f_regression, k='all')
        
        selector.fit(self.X, self.y_conversion)
        conv_scores = selector.scores_
        
        selector.fit(self.X, self.y_selectivity)
        sel_scores = selector.scores_
        
        print("  特征F统计量:")
        for i, feature in enumerate(self.feature_names):
            print(f"    {feature}: 转化率F={conv_scores[i]:.2f}, 选择性F={sel_scores[i]:.2f}")
        
        print("\n2.2 数据分割和缩放...")
        X_train, X_test, y_train_conv, y_test_conv, y_train_sel, y_test_sel = train_test_split(
            self.X, self.y_conversion, self.y_selectivity, 
            test_size=0.2, random_state=42, stratify=None
        )
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.y_train_conversion = y_train_conv
        self.y_test_conversion = y_test_conv
        self.y_train_selectivity = y_train_sel
        self.y_test_selectivity = y_test_sel
        self.scalers['robust'] = scaler
        
        print(f"✓ 特征选择和分割完成")
        print(f"  - 训练集: {len(X_train)}样本")
        print(f"  - 测试集: {len(X_test)}样本")
        print(f"  - 数据缩放: RobustScaler")
        
        return {'train_size': len(X_train), 'test_size': len(X_test)}
    
    def step3_regularized_gam(self):
        """步骤3: GAM模型构建（动态结构）"""
        print("\n" + "=" * 60)
        print("步骤3: GAM模型构建（动态结构）")
        print("=" * 60)
        
        gam_configs = {
            '简单': {'n_splines': 3, 'spline_order': 2, 'lam': 10},
            '中等': {'n_splines': 5, 'spline_order': 3, 'lam': 1},
            '复杂': {'n_splines': 8, 'spline_order': 3, 'lam': 0.1}
        }
        
        results = {}
        n_features = self.X_train.shape[1]
        
        for target_name, y_train, y_test in [('转化率', self.y_train_conversion, self.y_test_conversion),
                                            ('选择性', self.y_train_selectivity, self.y_test_selectivity)]:
            
            print(f"\n3.{1 if target_name == '转化率' else 2} {target_name}模型优化...")
            
            best_score = -np.inf
            best_model = None
            best_config = None
            
            for config_name, params in gam_configs.items():
                try:
                    # --- 修改：动态构建GAM模型结构 ---
                    terms = s(0, n_splines=params['n_splines'], spline_order=params['spline_order'])
                    for i in range(1, n_features):
                        terms += s(i, n_splines=params['n_splines'], spline_order=params['spline_order'])
                    
                    gam = LinearGAM(terms)
                    
                    gam.fit(self.X_train, y_train)
                    
                    try:
                        lam_range = np.logspace(-2, 2, 10)
                        gam.gridsearch(self.X_train, y_train, lam=lam_range)
                    except:
                        pass
                    
                    train_score = gam.score(self.X_train, y_train)
                    test_score = gam.score(self.X_test, y_test)
                    
                    cv_scores = self._cross_validate_gam(gam, target_name)
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                    try:
                        aic = gam.statistics_['AIC']
                    except:
                        aic = np.nan
                    
                    print(f"  {config_name}模型: 训练R²={train_score:.3f}, 测试R²={test_score:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
                    
                    composite_score = test_score - cv_std
                    if composite_score > best_score:
                        best_score = composite_score
                        best_model = gam
                        best_config = config_name
                        
                except Exception as e:
                    print(f"  {config_name}模型: 构建失败 ({str(e)[:50]}...)")
            
            print(f"  最佳{target_name}模型: {best_config}")
            
            model_key = 'conversion' if target_name == '转化率' else 'selectivity'
            self.models[f'{model_key}_best'] = best_model
            
            results[target_name] = {
                'best_config': best_config,
                'best_score': best_score,
                'model': best_model
            }
        
        return results
    
    def step4_comprehensive_evaluation(self):
        """步骤4: 模型评估"""
        print("\n" + "=" * 60)
        print("步骤4: 模型评估")
        print("=" * 60)
        
        evaluation_results = {}
        
        for target_name, y_train, y_test in [('转化率', self.y_train_conversion, self.y_test_conversion),
                                            ('选择性', self.y_train_selectivity, self.y_test_selectivity)]:
            
            print(f"\n4.{1 if target_name == '转化率' else 2} {target_name}模型详细评估")
            print("-" * 50)
            
            model_key = 'conversion' if target_name == '转化率' else 'selectivity'
            model = self.models[f'{model_key}_best']
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            cv_scores = self._cross_validate_gam(model, target_name)
            cv_r2_mean = np.mean(cv_scores)
            cv_r2_std = np.std(cv_scores)
            
            feature_importance = self._calculate_improved_feature_importance(model, target_name)
            
            residuals = y_test - y_pred_test
            
            try:
                aic = model.statistics_['AIC']
            except:
                aic = np.nan
            
            print(f"  基础指标:")
            print(f"    训练集 R² = {r2_train:.4f}")
            print(f"    测试集 R² = {r2_test:.4f}")
            print(f"    RMSE = {rmse_test:.4f}")
            print(f"    交叉验证 R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
            print(f"    AIC = {aic:.2f}" if not np.isnan(aic) else "    AIC = 无法计算")
            
            print(f"  改进的特征重要性:")
            for i, feature in enumerate(self.feature_names):
                importance = feature_importance[i]
                print(f"    {feature}: {importance:.3f}")
            
            print(f"  残差分析:")
            print(f"    残差均值 = {np.mean(residuals):.4f}")
            print(f"    残差标准差 = {np.std(residuals):.4f}")
            
            stability = "优秀" if cv_r2_std < 0.1 else "良好" if cv_r2_std < 0.2 else "一般"
            overfitting = "low" if r2_train - r2_test < 0.1 else "medium" if r2_train - r2_test < 0.2 else "high"
            
            print(f"  模型质量:")
            print(f"    稳定性 = {stability} (CV标准差={cv_r2_std:.3f})")
            print(f"    过拟合风险 = {overfitting}")
            
            evaluation_results[target_name] = {
                'r2_train': r2_train, 'r2_test': r2_test, 'rmse': rmse_test,
                'cv_mean': cv_r2_mean, 'cv_std': cv_r2_std,
                'feature_importance': feature_importance,
                'stability': stability, 'overfitting': overfitting
            }
        
        return evaluation_results
    
    def step5_visualization(self, step4_results):
        """步骤5: 模型可视化（自适应布局）"""
        print("\n" + "=" * 60)
        print("步骤5: 模型可视化")
        print("=" * 60)
        
        n_features = len(self.feature_names)
        # --- 修复：确保有足够的列数来容纳所有特征 ---
        n_cols = max(4, n_features)  # 确保至少有4列，或者等于特征数
        fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 4, 12))
        
        for row, (target_name, model_key) in enumerate([('转化率', 'conversion_best'), ('选择性', 'selectivity_best')]):
            model = self.models[model_key]
            y_train = self.y_train_conversion if target_name == '转化率' else self.y_train_selectivity
            
            # 绘制偏依赖图
            for col in range(n_features):
                ax = axes[row, col]
                feature_name = self.feature_names[col]
                
                try:
                    XX = model.generate_X_grid(term=col, n=100)
                    pdep, confi = model.partial_dependence(term=col, X=XX, width=0.95)
                    
                    ax.plot(XX[:, col], pdep, 'b-')
                    ax.plot(XX[:, col], confi, c='r', ls='--')
                    ax.set_xlabel(feature_name)
                    ax.set_ylabel(f'{target_name}偏效应')
                    ax.set_title(f'{target_name} - {feature_name}')
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    ax.text(0.5, 0.5, f'可视化失败', ha='center', va='center')
                    ax.set_title(f'{target_name} - {feature_name}')

        # 特征重要性对比图
        conv_importance = step4_results['转化率']['feature_importance']
        sel_importance = step4_results['选择性']['feature_importance']
        ax = axes[2, 0]
        x = np.arange(n_features)
        width = 0.35
        ax.bar(x - width/2, conv_importance, width, label='转化率', alpha=0.8)
        ax.bar(x + width/2, sel_importance, width, label='选择性', alpha=0.8)
        ax.set_xlabel('特征'); ax.set_ylabel('重要性'); ax.set_title('特征重要性对比')
        ax.set_xticks(x); ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # 模型性能对比
        ax = axes[2, 1]
        metrics = ['R²_test', 'CV_mean', 'Stability']
        conv_values = [step4_results['转化率']['r2_test'], step4_results['转化率']['cv_mean'], 1 - step4_results['转化率']['cv_std']]
        sel_values = [step4_results['选择性']['r2_test'], step4_results['选择性']['cv_mean'], 1 - step4_results['选择性']['cv_std']]
        x = np.arange(len(metrics))
        ax.bar(x - width/2, conv_values, width, label='转化率', alpha=0.8)
        ax.bar(x + width/2, sel_values, width, label='选择性', alpha=0.8)
        ax.set_xlabel('评估指标'); ax.set_ylabel('数值'); ax.set_title('模型性能对比')
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # 隐藏所有未使用的子图
        for i in range(n_features, n_cols):
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)
        for i in range(2, n_cols):
            axes[2, i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('gam_interaction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 可视化完成，保存为 gam_interaction_analysis.png")
    
    def _calculate_loading_ratio(self):
        """计算装料质量比"""
        ratios = []
        for _, row in self.processed_data.iterrows():
            ratio_str = row['Co/SiO2与HAP装料比']
            if pd.isna(ratio_str) or ratio_str == '50mg:无':
                ratios.append(np.nan)
            else:
                try:
                    parts = ratio_str.split(':')
                    if len(parts) == 2:
                        val1 = float(parts[0].replace('mg', ''))
                        val2 = float(parts[1].replace('mg', ''))
                        ratios.append(val1 / val2 if val2 != 0 else np.nan)
                    else: ratios.append(np.nan)
                except: ratios.append(np.nan)
        return ratios
    
    def _cross_validate_gam(self, model, target_name, cv=5):
        """GAM交叉验证"""
        y_all = np.concatenate([self.y_train_conversion, self.y_test_conversion]) if target_name == '转化率' else np.concatenate([self.y_train_selectivity, self.y_test_selectivity])
        X_all = np.vstack([self.X_train, self.X_test])
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_all):
            X_cv_train, X_cv_val = X_all[train_idx], X_all[val_idx]
            y_cv_train, y_cv_val = y_all[train_idx], y_all[val_idx]
            
            try:
                cv_gam = LinearGAM(model.terms).fit(X_cv_train, y_cv_train)
                cv_score = cv_gam.score(X_cv_val, y_cv_val)
                cv_scores.append(cv_score)
            except:
                cv_scores.append(0)
                
        return cv_scores
    
    def _calculate_improved_feature_importance(self, model, target_name):
        """改进的特征重要性计算"""
        importance_scores = []
        y_true = self.y_test_conversion if target_name == '转化率' else self.y_test_selectivity
        
        for i in range(len(self.feature_names)):
            try:
                baseline_score = r2_score(y_true, model.predict(self.X_test))
                X_permuted = self.X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_score = r2_score(y_true, model.predict(X_permuted))
                importance = max(0, baseline_score - permuted_score)
                importance_scores.append(importance)
            except:
                importance_scores.append(0.0)
        
        total_importance = sum(importance_scores)
        if total_importance > 0:
            return [score / total_importance for score in importance_scores]
        else:
            return [1.0 / len(self.feature_names)] * len(self.feature_names)

def main():
    """主函数"""
    print("乙醇偶合制备C4烯烃 - 含交互特征的GAM分析")
    print("=" * 80)
    
    set_chinese_font()
    
    analyzer = OptimizedGAMAnalyzer(
        data_path='../附件1.csv',
        catalyst_info_path='每组指标.csv'
    )
    
    step1_results = analyzer.step1_enhanced_preprocessing()
    step2_results = analyzer.step2_feature_selection()
    step3_results = analyzer.step3_regularized_gam()
    step4_results = analyzer.step4_comprehensive_evaluation()
    analyzer.step5_visualization(step4_results)
    
    print("\n" + "=" * 80)
    print("含交互特征的GAM分析完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
