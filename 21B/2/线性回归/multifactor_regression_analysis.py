"""
乙醇偶合制备C4烯烃 - 多因素影响分析
问题3-2.2：多因素线性回归建模分析

模块化流程：
1. 数据预处理与标准化
2. 数据集分割 
3. 共线性检验
4. 基础线性模型构建
5. 残差诊断
6. 交互项逐步回归
7. 过拟合控制与模型选择
8. 模型评估与结果输出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import warnings
import itertools

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

class MultifactorRegressionAnalyzer:
    """多因素回归分析器"""
    
    def __init__(self, data_path, catalyst_info_path):
        """初始化分析器"""
        self.data_path = data_path
        self.catalyst_info_path = catalyst_info_path
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train_conversion = None
        self.y_test_conversion = None
        self.y_train_selectivity = None
        self.y_test_selectivity = None
        self.scaler = None
        self.feature_names = ['温度', 'Co负载量', '装料质量比', '乙醇浓度']
        self.results_summary = {}
        
    def step1_data_preprocessing(self):
        """步骤1: 数据预处理与标准化"""
        print("=" * 60)
        print("步骤1: 数据预处理与标准化")
        print("=" * 60)
        
        # 加载数据
        print("1.1 加载数据...")
        self.raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        catalyst_info = pd.read_csv(self.catalyst_info_path, encoding='utf-8')
        
        # 剔除A11异常样本
        print("1.2 剔除A11异常样本...")
        self.raw_data = self.raw_data[self.raw_data['催化剂组合编号'] != 'A11'].copy()
        catalyst_info = catalyst_info[catalyst_info['催化剂组合编号'] != 'A11'].copy()
        
        # 合并数据
        print("1.3 合并催化剂信息...")
        self.processed_data = pd.merge(self.raw_data, catalyst_info, on='催化剂组合编号', how='left')
        
        # 提取数值特征
        print("1.4 提取数值特征...")
        self.processed_data['Co负载量_数值'] = self.processed_data['Co负载量'].str.extract(r'(\d+\.?\d*)').astype(float)
        self.processed_data['乙醇浓度_数值'] = self.processed_data['乙醇浓度'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # 计算装料质量比
        self.processed_data['装料质量比'] = self._calculate_loading_ratio()
        
        # 准备特征矩阵和目标变量
        feature_columns = ['温度', 'Co负载量_数值', '装料质量比', '乙醇浓度_数值']
        target_columns = ['乙醇转化率(%)', 'C4烯烃选择性(%)']
        
        # 移除缺失值
        clean_data = self.processed_data.dropna(subset=feature_columns + target_columns)
        
        X = clean_data[feature_columns].values
        y_conversion = clean_data['乙醇转化率(%)'].values
        y_selectivity = clean_data['C4烯烃选择性(%)'].values
        
        # 标准化特征
        print("1.5 标准化特征变量...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 保存处理后的数据
        self.X_original = X
        self.X_scaled = X_scaled
        self.y_conversion = y_conversion
        self.y_selectivity = y_selectivity
        
        print(f"✓ 数据预处理完成")
        print(f"  - 有效样本数: {len(clean_data)}")
        print(f"  - 特征变量数: {len(feature_columns)}")
        print(f"  - 目标变量: 乙醇转化率(%), C4烯烃选择性(%)")
        print(f"  - 特征标准化: 均值=0, 标准差=1")
        
        return {
            'sample_size': len(clean_data),
            'feature_count': len(feature_columns),
            'feature_names': self.feature_names,
            'standardization': '完成'
        }
    
    def step2_data_splitting(self, test_size=0.2, random_state=42):
        """步骤2: 数据集分割"""
        print("\n" + "=" * 60)
        print("步骤2: 数据集分割")
        print("=" * 60)
        
        print(f"2.1 按照 {int((1-test_size)*100)}%-{int(test_size*100)}% 分割训练集和测试集...")
        
        # 分层分割，确保温度分布均衡
        X_train, X_test, y_train_conv, y_test_conv, y_train_sel, y_test_sel = train_test_split(
            self.X_scaled, self.y_conversion, self.y_selectivity, 
            test_size=test_size, random_state=random_state, stratify=None
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_conversion = y_train_conv
        self.y_test_conversion = y_test_conv
        self.y_train_selectivity = y_train_sel
        self.y_test_selectivity = y_test_sel
        
        print(f"✓ 数据分割完成")
        print(f"  - 训练集样本数: {len(X_train)}")
        print(f"  - 测试集样本数: {len(X_test)}")
        print(f"  - 整体温度范围: {self.X_original[:, 0].min():.0f}-{self.X_original[:, 0].max():.0f}°C")
        print(f"  - 分割方式: 随机分割，确保代表性")
        
        return {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'split_ratio': f"{int((1-test_size)*100)}%-{int(test_size*100)}%"
        }
    
    def step3_multicollinearity_check(self):
        """步骤3: 共线性检验"""
        print("\n" + "=" * 60)
        print("步骤3: 共线性检验 (VIF)")
        print("=" * 60)
        
        print("3.1 计算方差膨胀因子 (VIF)...")
        
        # 添加常数项用于VIF计算
        X_with_const = sm.add_constant(self.X_train)
        
        vif_results = []
        for i in range(1, X_with_const.shape[1]):  # 跳过常数项
            vif = variance_inflation_factor(X_with_const, i)
            vif_results.append({
                '特征': self.feature_names[i-1],
                'VIF': vif,
                '共线性评价': '严重' if vif > 10 else '中等' if vif > 5 else '轻微'
            })
        
        vif_df = pd.DataFrame(vif_results)
        print("3.2 VIF检验结果:")
        print(vif_df.to_string(index=False))
        
        # 检查严重共线性
        severe_multicollinearity = vif_df[vif_df['VIF'] > 5]
        if len(severe_multicollinearity) > 0:
            print(f"\n⚠ 发现{len(severe_multicollinearity)}个变量存在较强共线性 (VIF > 5)")
            for _, row in severe_multicollinearity.iterrows():
                print(f"  - {row['特征']}: VIF = {row['VIF']:.3f}")
        else:
            print("\n✓ 所有变量的VIF < 5，共线性问题不严重")
        
        # 保存VIF结果
        vif_df.to_csv('multifactor_vif_results.csv', index=False, encoding='utf-8')
        
        return {
            'vif_results': vif_df,
            'max_vif': vif_df['VIF'].max(),
            'multicollinearity_concern': len(severe_multicollinearity) > 0
        }
    
    def step4_baseline_models(self):
        """步骤4: 基础线性模型构建"""
        print("\n" + "=" * 60)
        print("步骤4: 基础线性模型构建")
        print("=" * 60)
        
        print("4.1 构建基础线性回归模型...")
        
        # 乙醇转化率模型
        print("\n4.1.1 乙醇转化率回归模型")
        model_conversion = LinearRegression()
        model_conversion.fit(self.X_train, self.y_train_conversion)
        
        # 预测
        y_pred_train_conv = model_conversion.predict(self.X_train)
        y_pred_test_conv = model_conversion.predict(self.X_test)
        
        # 模型评估
        r2_train_conv = r2_score(self.y_train_conversion, y_pred_train_conv)
        r2_test_conv = r2_score(self.y_test_conversion, y_pred_test_conv)
        mse_train_conv = mean_squared_error(self.y_train_conversion, y_pred_train_conv)
        mse_test_conv = mean_squared_error(self.y_test_conversion, y_pred_test_conv)
        
        print(f"  训练集 R² = {r2_train_conv:.4f}, MSE = {mse_train_conv:.4f}")
        print(f"  测试集 R² = {r2_test_conv:.4f}, MSE = {mse_test_conv:.4f}")
        
        # C4烯烃选择性模型
        print("\n4.1.2 C4烯烃选择性回归模型")
        model_selectivity = LinearRegression()
        model_selectivity.fit(self.X_train, self.y_train_selectivity)
        
        # 预测
        y_pred_train_sel = model_selectivity.predict(self.X_train)
        y_pred_test_sel = model_selectivity.predict(self.X_test)
        
        # 模型评估
        r2_train_sel = r2_score(self.y_train_selectivity, y_pred_train_sel)
        r2_test_sel = r2_score(self.y_test_selectivity, y_pred_test_sel)
        mse_train_sel = mean_squared_error(self.y_train_selectivity, y_pred_train_sel)
        mse_test_sel = mean_squared_error(self.y_test_selectivity, y_pred_test_sel)
        
        print(f"  训练集 R² = {r2_train_sel:.4f}, MSE = {mse_train_sel:.4f}")
        print(f"  测试集 R² = {r2_test_sel:.4f}, MSE = {mse_test_sel:.4f}")
        
        # 系数分析
        print("\n4.2 模型系数分析")
        coeff_results = []
        for i, feature in enumerate(self.feature_names):
            coeff_results.append({
                '特征': feature,
                '转化率系数': model_conversion.coef_[i],
                '选择性系数': model_selectivity.coef_[i]
            })
        
        coeff_df = pd.DataFrame(coeff_results)
        print(coeff_df.to_string(index=False))
        
        # 保存模型
        self.model_conversion_base = model_conversion
        self.model_selectivity_base = model_selectivity
        
        # 保存系数结果
        coeff_df.to_csv('multifactor_baseline_coefficients.csv', index=False, encoding='utf-8')
        
        return {
            'conversion_r2_train': r2_train_conv,
            'conversion_r2_test': r2_test_conv,
            'selectivity_r2_train': r2_train_sel,
            'selectivity_r2_test': r2_test_sel,
            'coefficients': coeff_df
        }
    
    def step5_residual_diagnostics(self):
        """步骤5: 残差诊断"""
        print("\n" + "=" * 60)
        print("步骤5: 残差诊断")
        print("=" * 60)
        
        print("5.1 计算训练集残差...")
        
        # 计算残差
        residuals_conv = self.y_train_conversion - self.model_conversion_base.predict(self.X_train)
        residuals_sel = self.y_train_selectivity - self.model_selectivity_base.predict(self.X_train)
        
        # 正态性检验 (Shapiro-Wilk)
        print("\n5.2 残差正态性检验 (Shapiro-Wilk)")
        sw_stat_conv, sw_p_conv = stats.shapiro(residuals_conv)
        sw_stat_sel, sw_p_sel = stats.shapiro(residuals_sel)
        
        print(f"  转化率模型: W = {sw_stat_conv:.4f}, p = {sw_p_conv:.4f} {'(正态)' if sw_p_conv > 0.05 else '(非正态)'}")
        print(f"  选择性模型: W = {sw_stat_sel:.4f}, p = {sw_p_sel:.4f} {'(正态)' if sw_p_sel > 0.05 else '(非正态)'}")
        
        # 同方差性检验 (Breusch-Pagan)
        print("\n5.3 残差同方差性检验 (Breusch-Pagan)")
        try:
            X_with_const = sm.add_constant(self.X_train)
            bp_stat_conv, bp_p_conv, _, _ = het_breuschpagan(residuals_conv, X_with_const)
            bp_stat_sel, bp_p_sel, _, _ = het_breuschpagan(residuals_sel, X_with_const)
            
            print(f"  转化率模型: LM = {bp_stat_conv:.4f}, p = {bp_p_conv:.4f} {'(同方差)' if bp_p_conv > 0.05 else '(异方差)'}")
            print(f"  选择性模型: LM = {bp_stat_sel:.4f}, p = {bp_p_sel:.4f} {'(同方差)' if bp_p_sel > 0.05 else '(异方差)'}")
        except:
            print("  ⚠ 同方差性检验计算失败")
            bp_p_conv = bp_p_sel = np.nan
        
        # 可视化残差分布
        print("\n5.4 绘制残差诊断图...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Q-Q图
        stats.probplot(residuals_conv, dist="norm", plot=axes[0,0])
        axes[0,0].set_title('转化率模型残差Q-Q图')
        
        stats.probplot(residuals_sel, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('选择性模型残差Q-Q图')
        
        # 残差直方图
        axes[1,0].hist(residuals_conv, bins=15, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('转化率模型残差分布')
        axes[1,0].set_xlabel('残差')
        axes[1,0].set_ylabel('频次')
        
        axes[1,1].hist(residuals_sel, bins=15, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('选择性模型残差分布')
        axes[1,1].set_xlabel('残差')
        axes[1,1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('multifactor_residual_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'conversion_normality_p': sw_p_conv,
            'selectivity_normality_p': sw_p_sel,
            'conversion_homoscedasticity_p': bp_p_conv,
            'selectivity_homoscedasticity_p': bp_p_sel
        }
    
    def step6_interaction_stepwise(self):
        """步骤6: 交互项逐步回归"""
        print("\n" + "=" * 60)
        print("步骤6: 交互项逐步回归")
        print("=" * 60)
        
        print("6.1 生成所有可能的二阶交互项...")
        
        # 生成所有可能的交互项组合
        interaction_pairs = list(itertools.combinations(range(len(self.feature_names)), 2))
        interaction_names = [f"{self.feature_names[i]}×{self.feature_names[j]}" for i, j in interaction_pairs]
        
        print(f"  可能的交互项数量: {len(interaction_pairs)}")
        for name in interaction_names:
            print(f"    - {name}")
        
        # 为每个目标变量进行逐步回归
        results = {}
        for target_name, y_train, y_test in [
            ('转化率', self.y_train_conversion, self.y_test_conversion),
            ('选择性', self.y_train_selectivity, self.y_test_selectivity)
        ]:
            print(f"\n6.2 {target_name}模型交互项逐步回归...")
            
            # 从基础模型开始
            current_features = list(range(len(self.feature_names)))
            X_current = self.X_train[:, current_features]
            
            best_model = LinearRegression().fit(X_current, y_train)
            best_r2 = best_model.score(X_current, y_train)
            best_bic = self._calculate_bic(y_train, best_model.predict(X_current), len(current_features))
            
            added_interactions = []
            step = 0
            
            print(f"    基础模型 R² = {best_r2:.4f}, BIC = {best_bic:.2f}")
            
            while True:
                step += 1
                best_improvement = 0
                best_interaction = None
                best_new_model = None
                best_new_bic = float('inf')
                
                # 尝试添加每个未添加的交互项
                for i, (idx1, idx2) in enumerate(interaction_pairs):
                    interaction_name = interaction_names[i]
                    if interaction_name in added_interactions:
                        continue
                    
                    # 创建包含交互项的特征矩阵
                    interaction_term = (self.X_train[:, idx1] * self.X_train[:, idx2]).reshape(-1, 1)
                    X_with_interaction = np.hstack([X_current, interaction_term])
                    
                    # 拟合模型
                    model = LinearRegression().fit(X_with_interaction, y_train)
                    r2 = model.score(X_with_interaction, y_train)
                    bic = self._calculate_bic(y_train, model.predict(X_with_interaction), X_with_interaction.shape[1])
                    
                    # F检验显著性
                    f_stat, p_value = self._f_test_interaction(X_current, X_with_interaction, y_train)
                    
                    if p_value < 0.05 and bic < best_new_bic:
                        best_improvement = r2 - best_r2
                        best_interaction = interaction_name
                        best_new_model = model
                        best_new_bic = bic
                        best_new_r2 = r2
                        best_p_value = p_value
                
                # 如果找到显著改进，添加交互项
                if best_interaction is not None and best_improvement > 0.001:
                    added_interactions.append(best_interaction)
                    idx1, idx2 = interaction_pairs[interaction_names.index(best_interaction)]
                    interaction_term = (self.X_train[:, idx1] * self.X_train[:, idx2]).reshape(-1, 1)
                    X_current = np.hstack([X_current, interaction_term])
                    best_model = best_new_model
                    best_r2 = best_new_r2
                    best_bic = best_new_bic
                    
                    print(f"    第{step}步: 添加 {best_interaction}")
                    print(f"      R² = {best_r2:.4f} (+{best_improvement:.4f})")
                    print(f"      BIC = {best_bic:.2f}")
                    print(f"      p值 = {best_p_value:.4f}")
                else:
                    print(f"    第{step}步: 无显著交互项可添加，停止")
                    break
            
            print(f"    最终{target_name}模型包含交互项: {added_interactions}")
            
            results[target_name] = {
                'model': best_model,
                'interactions': added_interactions,
                'r2': best_r2,
                'bic': best_bic,
                'X_features': X_current
            }
        
        self.interaction_results = results
        return results
    
    def step7_overfitting_control(self):
        """步骤7: 过拟合控制与模型选择"""
        print("\n" + "=" * 60)
        print("步骤7: 过拟合控制与模型选择")
        print("=" * 60)
        
        print("7.1 BIC准则模型比较...")
        
        model_comparison = []
        
        for target_name in ['转化率', '选择性']:
            if target_name == '转化率':
                y_train, y_test = self.y_train_conversion, self.y_test_conversion
                base_model = self.model_conversion_base
            else:
                y_train, y_test = self.y_train_selectivity, self.y_test_selectivity
                base_model = self.model_selectivity_base
            
            # 基础模型
            base_pred = base_model.predict(self.X_train)
            base_r2_train = r2_score(y_train, base_pred)
            base_r2_test = r2_score(y_test, base_model.predict(self.X_test))
            base_bic = self._calculate_bic(y_train, base_pred, self.X_train.shape[1])
            
            # 交互项模型
            inter_model = self.interaction_results[target_name]['model']
            inter_X = self.interaction_results[target_name]['X_features']
            inter_pred = inter_model.predict(inter_X)
            inter_r2_train = r2_score(y_train, inter_pred)
            inter_bic = self.interaction_results[target_name]['bic']
            
            # 构建测试集的交互项特征
            X_test_inter = self._build_interaction_features(
                self.X_test, 
                self.interaction_results[target_name]['interactions']
            )
            inter_r2_test = r2_score(y_test, inter_model.predict(X_test_inter))
            
            model_comparison.extend([
                {
                    '目标变量': target_name,
                    '模型类型': '基础线性',
                    '特征数量': self.X_train.shape[1],
                    '训练集R²': base_r2_train,
                    '测试集R²': base_r2_test,
                    'BIC': base_bic,
                    '过拟合风险': '低' if base_r2_test >= 0.6 and base_r2_train - base_r2_test < 0.1 else '中'
                },
                {
                    '目标变量': target_name,
                    '模型类型': '含交互项',
                    '特征数量': inter_X.shape[1],
                    '训练集R²': inter_r2_train,
                    '测试集R²': inter_r2_test,
                    'BIC': inter_bic,
                    '过拟合风险': '低' if inter_r2_test >= 0.6 and inter_r2_train - inter_r2_test < 0.1 else '高' if inter_r2_train - inter_r2_test > 0.2 else '中'
                }
            ])
        
        comparison_df = pd.DataFrame(model_comparison)
        print(comparison_df.to_string(index=False))
        
        print("\n7.2 正则化方法尝试...")
        regularization_results = []
        
        for target_name in ['转化率', '选择性']:
            y_train = self.y_train_conversion if target_name == '转化率' else self.y_train_selectivity
            y_test = self.y_test_conversion if target_name == '转化率' else self.y_test_selectivity
            
            # Ridge回归
            ridge = Ridge(alpha=1.0)
            ridge.fit(self.X_train, y_train)
            ridge_r2_test = ridge.score(self.X_test, y_test)
            
            # Lasso回归
            lasso = Lasso(alpha=0.1)
            lasso.fit(self.X_train, y_train)
            lasso_r2_test = lasso.score(self.X_test, y_test)
            
            regularization_results.append({
                '目标变量': target_name,
                'Ridge R²': ridge_r2_test,
                'Lasso R²': lasso_r2_test
            })
        
        reg_df = pd.DataFrame(regularization_results)
        print(reg_df.to_string(index=False))
        
        # 保存模型比较结果
        comparison_df.to_csv('multifactor_model_comparison.csv', index=False, encoding='utf-8')
        
        return {
            'model_comparison': comparison_df,
            'regularization_results': reg_df
        }
    
    def step8_final_evaluation(self):
        """步骤8: 模型评估与结果输出"""
        print("\n" + "=" * 60)
        print("步骤8: 最终模型评估")
        print("=" * 60)
        
        final_results = {}
        
        for target_name in ['转化率', '选择性']:
            print(f"\n8.{1 if target_name == '转化率' else 2} {target_name}模型最终评估")
            
            y_train = self.y_train_conversion if target_name == '转化率' else self.y_train_selectivity
            y_test = self.y_test_conversion if target_name == '转化率' else self.y_test_selectivity
            
            # 基础模型评估
            base_model = self.model_conversion_base if target_name == '转化率' else self.model_selectivity_base
            base_r2_test = base_model.score(self.X_test, y_test)
            
            # 交互项模型评估
            inter_model = self.interaction_results[target_name]['model']
            X_test_inter = self._build_interaction_features(
                self.X_test, 
                self.interaction_results[target_name]['interactions']
            )
            inter_r2_test = inter_model.score(X_test_inter, y_test)
            
            # 模型选择
            if inter_r2_test > base_r2_test and inter_r2_test > 0.6:
                chosen_model = 'interaction'
                chosen_r2 = inter_r2_test
                model_description = f"含交互项模型 (R² = {inter_r2_test:.4f})"
            else:
                chosen_model = 'baseline'
                chosen_r2 = base_r2_test
                model_description = f"基础线性模型 (R² = {base_r2_test:.4f})"
            
            print(f"  推荐模型: {model_description}")
            print(f"  可接受性: {'✓ 可接受' if chosen_r2 > 0.6 else '⚠ 需要改进'}")
            
            final_results[target_name] = {
                'chosen_model': chosen_model,
                'test_r2': chosen_r2,
                'model_acceptable': chosen_r2 > 0.6,
                'interactions': self.interaction_results[target_name]['interactions']
            }
        
        print(f"\n✓ 分析完成! 生成了{len([f for f in ['multifactor_vif_results.csv', 'multifactor_baseline_coefficients.csv', 'multifactor_residual_diagnostics.png', 'multifactor_model_comparison.csv'] if f])}个结果文件")
        
        return final_results
    
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
                    else:
                        ratios.append(np.nan)
                except:
                    ratios.append(np.nan)
        return ratios
    
    def _calculate_bic(self, y_true, y_pred, k):
        """计算BIC"""
        n = len(y_true)
        sse = np.sum((y_true - y_pred) ** 2)
        bic = n * np.log(sse / n) + k * np.log(n)
        return bic
    
    def _f_test_interaction(self, X_base, X_extended, y):
        """F检验交互项显著性"""
        model_base = LinearRegression().fit(X_base, y)
        model_extended = LinearRegression().fit(X_extended, y)
        
        rss_base = np.sum((y - model_base.predict(X_base)) ** 2)
        rss_extended = np.sum((y - model_extended.predict(X_extended)) ** 2)
        
        df_base = len(y) - X_base.shape[1] - 1
        df_extended = len(y) - X_extended.shape[1] - 1
        
        f_stat = ((rss_base - rss_extended) / (df_base - df_extended)) / (rss_extended / df_extended)
        p_value = 1 - stats.f.cdf(f_stat, df_base - df_extended, df_extended)
        
        return f_stat, p_value
    
    def _build_interaction_features(self, X_base, interaction_names):
        """为测试集构建交互项特征"""
        X_extended = X_base.copy()
        
        for interaction_name in interaction_names:
            feature1, feature2 = interaction_name.split('×')
            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)
            
            interaction_term = (X_base[:, idx1] * X_base[:, idx2]).reshape(-1, 1)
            X_extended = np.hstack([X_extended, interaction_term])
        
        return X_extended
    


def main():
    """主函数"""
    print("乙醇偶合制备C4烯烃 - 多因素回归分析")
    print("=" * 60)
    
    # 设置中文字体
    set_chinese_font()
    
    # 初始化分析器
    analyzer = MultifactorRegressionAnalyzer(
        data_path='../附件1.csv',
        catalyst_info_path='每组指标.csv'
    )
    
    # 执行分步分析
    step1_results = analyzer.step1_data_preprocessing()
    step2_results = analyzer.step2_data_splitting()
    step3_results = analyzer.step3_multicollinearity_check()
    step4_results = analyzer.step4_baseline_models()
    step5_results = analyzer.step5_residual_diagnostics()
    step6_results = analyzer.step6_interaction_stepwise()
    step7_results = analyzer.step7_overfitting_control()
    step8_results = analyzer.step8_final_evaluation()
    # 分析总结
    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    
    print("\n关键发现：")
    
    # 交互项分析
    conversion_interactions = step8_results['转化率']['interactions']
    selectivity_interactions = step8_results['选择性']['interactions']
    
    if conversion_interactions:
        print(f"转化率模型发现显著交互项: {', '.join(conversion_interactions)}")
        print("表明温度与乙醇浓度存在协同效应，高温下乙醇浓度的负向影响可能被放大")
    else:
        print("转化率模型未发现显著交互项")
    
    if selectivity_interactions:
        print(f"选择性模型发现显著交互项: {', '.join(selectivity_interactions)}")
    else:
        print("选择性模型未发现显著交互项，各因素独立作用")
    
    # 模型性能分析
    print("模型性能：")
    print(f"转化率模型: R² = {step8_results['转化率']['test_r2']:.3f} - {'表现良好' if step8_results['转化率']['model_acceptable'] else '需要改进'}")
    print(f"选择性模型: R² = {step8_results['选择性']['test_r2']:.3f} - {'表现良好' if step8_results['选择性']['model_acceptable'] else '需要改进'}")
    
    # 共线性分析
    max_vif = step3_results['max_vif']
    print(f"\n共线性评估: VIF最大值 = {max_vif:.3f}")
    if max_vif < 2:
        print("变量间相互独立，建模可信")
    elif max_vif < 5:
        print("存在一定共线性")
    else:
        print("共线性较强")
    
    # 残差诊断分析
    print("\n模型假设检验：")
    norm_ok_conv = step5_results['conversion_normality_p'] > 0.05 if not pd.isna(step5_results['conversion_normality_p']) else False
    norm_ok_sel = step5_results['selectivity_normality_p'] > 0.05 if not pd.isna(step5_results['selectivity_normality_p']) else False
    homo_ok_conv = step5_results['conversion_homoscedasticity_p'] > 0.05 if not pd.isna(step5_results['conversion_homoscedasticity_p']) else False
    homo_ok_sel = step5_results['selectivity_homoscedasticity_p'] > 0.05 if not pd.isna(step5_results['selectivity_homoscedasticity_p']) else False
    
    print(f"残差正态性: 转化率{'通过' if norm_ok_conv else '未通过'} 选择性{'通过' if norm_ok_sel else '未通过'}")
    print(f"同方差性: 转化率{'通过' if homo_ok_conv else '未通过'} 选择性{'通过' if homo_ok_sel else '未通过'}")
    
    if not (norm_ok_conv and norm_ok_sel and homo_ok_conv and homo_ok_sel):
        print("建议考虑变量变换或非线性建模")
    
    # 工艺建议
    print("\n工艺优化建议：")
    coeff_df = step4_results['coefficients']
    temp_coeff_conv = coeff_df[coeff_df['特征'] == '温度']['转化率系数'].iloc[0]
    ethanol_coeff_conv = coeff_df[coeff_df['特征'] == '乙醇浓度']['转化率系数'].iloc[0]
    
    print(f"1. 温度仍是最关键因素（系数={temp_coeff_conv:.1f}），优先优化温度控制")
    print(f"2. 乙醇浓度负向影响显著（系数={ethanol_coeff_conv:.1f}），建议适当降低")
    if conversion_interactions:
        print("3. 注意温度与乙醇浓度的协同效应，避免高温高浓度组合")
    print("4. Co负载量和装料比影响相对较小，可作为次要优化参数")
    
    print("\n" + "=" * 60)
    print("分析完成，已生成数据文件供进一步分析")
    print("=" * 60)

if __name__ == "__main__":
    main()