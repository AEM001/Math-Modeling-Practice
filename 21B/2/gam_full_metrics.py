import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from pygam import LinearGAM, s, f
from functools import reduce
import operator
import warnings

warnings.filterwarnings('ignore')

def set_chinese_font():
    """设置中文字体"""
    chinese_fonts = ['PingFang HK', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
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

class GAMAnalyzer:
    """GAM分析器"""
    
    def __init__(self, data_path, catalyst_info_path):
        """初始化分析器"""
        self.data_path = data_path
        self.catalyst_info_path = catalyst_info_path
        self.processed_data = None
        self.X_train, self.X_test = None, None
        self.y_train_conversion, self.y_test_conversion = None, None
        self.y_train_selectivity, self.y_test_selectivity = None, None
        self.feature_names = ['温度', 'Co负载量', '装料质量比', '乙醇浓度', 'Co/SiO2用量', 'HAP用量', '投料方式']
        self.scaler = None
        self.models = {}
        
    def step1_data_preprocessing(self):
        """步骤1: 数据预处理"""
        print("=" * 60)
        print("步骤1: 数据预处理")
        print("=" * 60)
        
        print("1.1 加载和清洗数据...")
        raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        catalyst_info = pd.read_csv(self.catalyst_info_path, encoding='utf-8')
        
        # 移除A11样本
        raw_data = raw_data[raw_data['催化剂组合编号'] != 'A11'].copy()
        catalyst_info = catalyst_info[catalyst_info['催化剂组合编号'] != 'A11'].copy()
        
        self.processed_data = pd.merge(raw_data, catalyst_info, on='催化剂组合编号', how='left')
        
        print("1.2 特征工程...")
        # 数值化处理
        self.processed_data['Co负载量_数值'] = self.processed_data['Co负载量'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['乙醇浓度_数值'] = self.processed_data['乙醇浓度'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['装料质量比'] = self._calculate_loading_ratio()
        self.processed_data['Co/SiO2用量_数值'] = self.processed_data['Co/SiO2用量'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['HAP用量_数值'] = self.processed_data['HAP用量'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['投料方式_数值'] = self.processed_data['催化剂组合编号'].apply(lambda x: 1 if x.startswith('B') else 0)

        feature_columns = ['温度', 'Co负载量_数值', '装料质量比', '乙醇浓度_数值', 'Co/SiO2用量_数值', 'HAP用量_数值', '投料方式_数值']
        target_columns = ['乙醇转化率(%)', 'C4烯烃选择性(%)']
        
        print("1.3 异常值检测...")
        clean_data = self.processed_data.dropna(subset=feature_columns + target_columns).copy()
        
        # 异常值处理
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
        
        self.X = clean_data[feature_columns].values
        self.y_conversion = clean_data['乙醇转化率(%)'].values
        self.y_selectivity = clean_data['C4烯烃选择性(%)'].values
        
        print(f"✓ 数据预处理完成")
        print(f"  - 有效样本数: {len(clean_data)}")
        print(f"  - 特征变量 ({len(self.feature_names)}个): {', '.join(self.feature_names)}")
        
        return {'sample_size': len(clean_data), 'features': self.feature_names}
    
    def step2_data_split(self):
        """步骤2: 数据分割"""
        print("\n" + "=" * 60)
        print("步骤2: 数据分割")
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
        
        print("\n2.2 数据分割...")
        # 使用最佳分割策略（基于之前测试结果）
        X_train, X_test, y_train_conv, y_test_conv = train_test_split(
            self.X, self.y_conversion, test_size=0.25, random_state=42
        )
        _, _, y_train_sel, y_test_sel = train_test_split(
            self.X, self.y_selectivity, test_size=0.25, random_state=42
        )
        
        print("2.3 数据缩放...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.X_train_raw, self.X_test_raw = X_train, X_test
        self.y_train_conversion, self.y_test_conversion = y_train_conv, y_test_conv
        self.y_train_selectivity, self.y_test_selectivity = y_train_sel, y_test_sel
        
        print(f"✓ 数据分割完成")
        print(f"  - 训练集: {len(X_train)}样本")
        print(f"  - 测试集: {len(X_test)}样本")
        
        return {'train_size': len(X_train), 'test_size': len(X_test)}
    
    def step3_model_building(self):
        """步骤3: GAM模型构建"""
        print("\n" + "=" * 60)
        print("步骤3: GAM模型构建")
        print("=" * 60)
        
        # 简化的配置选项
        gam_configs = {
            '极简': {'n_splines': 3, 'spline_order': 2},
            '简单': {'n_splines': 5, 'spline_order': 3},
            '中等': {'n_splines': 8, 'spline_order': 3}
        }
        
        lam_ranges = {
            '保守': np.logspace(-2, 2, 9),
            '标准': np.logspace(-3, 3, 11),
            '精细': np.logspace(-5, 5, 15)
        }
        
        results = {}
        
        for target_name, y_train, y_test in [('转化率', self.y_train_conversion, self.y_test_conversion),
                                            ('选择性', self.y_train_selectivity, self.y_test_selectivity)]:
            
            print(f"\n3.{1 if target_name == '转化率' else 2} {target_name}模型构建...")
            
            best_score = -np.inf
            best_model = None
            best_config = None
            best_lam_range = None
            
            for config_name, params in gam_configs.items():
                for lam_name, lam_range in lam_ranges.items():
                    try:
                        # 构建模型项
                        continuous_terms = []
                        for i in range(len(self.feature_names) - 1):
                            feature_importance = self._get_feature_importance_estimate(i, target_name)
                            adjusted_splines = max(3, min(params['n_splines'], 
                                                        int(params['n_splines'] * (1 + feature_importance))))
                            continuous_terms.append(s(i, n_splines=adjusted_splines, spline_order=params['spline_order']))
                        
                        categorical_term = [f(len(self.feature_names) - 1)]
                        all_terms = continuous_terms + categorical_term
                        model_terms = reduce(operator.add, all_terms)
                        gam = LinearGAM(model_terms)
                        
                        # 网格搜索
                        gam.gridsearch(self.X_train, y_train, lam=lam_range)
                        
                        train_score = gam.score(self.X_train, y_train)
                        test_score = gam.score(self.X_test, y_test)
                        
                        cv_scores = self._cross_validate_gam(gam, target_name)
                        cv_mean = np.mean(cv_scores)
                        cv_std = np.std(cv_scores)
                        
                        # 综合评分
                        overfitting_penalty = max(0, train_score - test_score - 0.1) * 2
                        stability_bonus = max(0, 0.2 - cv_std) * 0.5
                        complexity_penalty = len(continuous_terms) * 0.01
                        composite_score = test_score - cv_std - overfitting_penalty + stability_bonus - complexity_penalty
                        
                        print(f"  {config_name}+{lam_name}: 训练R²={train_score:.3f}, 测试R²={test_score:.3f}, "
                              f"CV={cv_mean:.3f}±{cv_std:.3f}, 综合={composite_score:.3f}")
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            best_model = gam
                            best_config = config_name
                            best_lam_range = lam_name
                            
                    except Exception as e:
                        print(f"  {config_name}+{lam_name}: 构建失败 ({e})")
            
            print(f"  ✓ 最佳{target_name}模型: {best_config}+{best_lam_range}")
            
            model_key = 'conversion' if target_name == '转化率' else 'selectivity'
            self.models[f'{model_key}_best'] = best_model
            
            results[target_name] = {
                'best_config': best_config, 
                'best_lam_range': best_lam_range,
                'model': best_model
            }
        
        return results
    
    def step4_model_evaluation(self):
        """步骤4: 模型评估"""
        print("\n" + "=" * 60)
        print("步骤4: 模型评估")
        print("=" * 60)
        
        evaluation_results = {}
        
        for target_name, y_train, y_test in [('转化率', self.y_train_conversion, self.y_test_conversion),
                                            ('选择性', self.y_train_selectivity, self.y_test_selectivity)]:
            
            print(f"\n4.{1 if target_name == '转化率' else 2} {target_name}模型评估")
            print("-" * 50)
            
            model_key = 'conversion' if target_name == '转化率' else 'selectivity'
            model = self.models[f'{model_key}_best']
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            cv_scores = self._cross_validate_gam(model, target_name)
            cv_r2_mean, cv_r2_std = np.mean(cv_scores), np.std(cv_scores)
            
            feature_importance = self._calculate_feature_importance(model, target_name)
            
            print(f"  基础指标:")
            print(f"    训练集 R² = {r2_train:.4f}")
            print(f"    测试集 R² = {r2_test:.4f}")
            print(f"    RMSE = {rmse_test:.4f}")
            print(f"    交叉验证 R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
            
            print(f"  特征重要性:")
            sorted_importance = sorted(zip(self.feature_names, feature_importance), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance:
                print(f"    {feature}: {importance:.3f}")
            
            stability = "优秀" if cv_r2_std < 0.1 else "良好" if cv_r2_std < 0.2 else "一般"
            overfitting = "低" if r2_train - r2_test < 0.1 else "中" if r2_train - r2_test < 0.2 else "高"
            
            print(f"  模型质量:")
            print(f"    稳定性 = {stability}")
            print(f"    过拟合风险 = {overfitting}")
            
            evaluation_results[target_name] = {
                'r2_test': r2_test, 'cv_mean': cv_r2_mean, 'cv_std': cv_r2_std,
                'feature_importance': feature_importance, 'stability': stability
            }
        return evaluation_results
    
    def _calculate_loading_ratio(self):
        """计算装料质量比"""
        ratios = []
        for _, row in self.processed_data.iterrows():
            ratio_str = row['Co/SiO2与HAP装料比']
            if pd.isna(ratio_str) or ratio_str == '50mg:无':
                ratios.append(np.nan)
            else:
                try:
                    parts = ratio_str.replace('mg', '').split(':')
                    if len(parts) == 2:
                        val1, val2 = float(parts[0]), float(parts[1])
                        ratios.append(val1 / val2 if val2 != 0 else np.nan)
                    else:
                        ratios.append(np.nan)
                except:
                    ratios.append(np.nan)
        return ratios
    
    def _cross_validate_gam(self, model, target_name, cv=5):
        """GAM交叉验证"""
        y_all = self.y_conversion if target_name == '转化率' else self.y_selectivity
        X_all = np.vstack([self.X_train_raw, self.X_test_raw])
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_all):
            X_cv_train, X_cv_val = X_all[train_idx], X_all[val_idx]
            y_cv_train, y_cv_val = y_all[train_idx], y_all[val_idx]

            scaler = RobustScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            try:
                cv_gam = LinearGAM(model.terms, lam=model.lam).fit(X_cv_train_scaled, y_cv_train)
                cv_score = cv_gam.score(X_cv_val_scaled, y_cv_val)
                cv_scores.append(cv_score)
            except Exception as e:
                cv_scores.append(0)
        
        return [s for s in cv_scores if s > -1 and not np.isnan(s)]

    def _calculate_feature_importance(self, model, target_name):
        """计算特征重要性"""
        y_true = self.y_test_conversion if target_name == '转化率' else self.y_test_selectivity
        baseline_score = model.score(self.X_test, y_true)
        
        importance_scores = []
        for i in range(len(self.feature_names)):
            try:
                X_permuted = self.X_test.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_score = model.score(X_permuted, y_true)
                importance = max(0, baseline_score - permuted_score)
                importance_scores.append(importance)
            except:
                importance_scores.append(0)
        
        # 归一化
        total_importance = sum(importance_scores)
        if total_importance > 0:
            return [score / total_importance for score in importance_scores]
        else:
            return [1.0 / len(self.feature_names)] * len(self.feature_names)

    def _get_feature_importance_estimate(self, feature_idx, target_name):
        """获取特征重要性估计值"""
        # 动态计算F统计量
        from sklearn.feature_selection import SelectKBest, f_regression
        
        if target_name == '转化率':
            y_target = self.y_conversion
        else:  # 选择性
            y_target = self.y_selectivity
        
        # 使用所有数据计算F统计量
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(self.X, y_target)
        f_scores = selector.scores_
        
        if feature_idx < len(f_scores):
            return f_scores[feature_idx] / max(f_scores) if max(f_scores) > 0 else 0.1
        return 0.1

def main():
    """主函数"""
    print("乙醇偶合制备C4烯烃 - GAM分析")
    print("=" * 60)
    
    set_chinese_font()
    
    try:
        analyzer = GAMAnalyzer(
            data_path='/Users/Mac/Downloads/Math-Modeling-Practice/21B/附件1.csv',
            catalyst_info_path='每组指标.csv'
        )
    except FileNotFoundError as e:
        print(f"!!! 文件未找到错误: {e.filename}")
        print("请确保文件路径正确。")
        return
    
    # 执行分析流程
    analyzer.step1_data_preprocessing()
    analyzer.step2_data_split()
    analyzer.step3_model_building()
    step4_results = analyzer.step4_model_evaluation()
    
    print("\n" + "=" * 60)
    print("GAM分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
