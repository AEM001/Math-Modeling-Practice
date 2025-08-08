#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPR模型构建模块 - GPR实验设计系统
功能：高斯过程回归模型构建、训练、验证
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 更稳健的中文字体设置
import platform, os
from matplotlib import font_manager as fm

def _set_chinese_font():
    system = platform.system()
    if system == 'Darwin':  # macOS 常见中文字体
        candidates = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'Songti SC', 'STHeiti', 'STSong', 'Arial Unicode MS']
        font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        file_keywords = ['PingFang', 'Hiragino', 'Heiti', 'Songti', 'STHeiti', 'STSong']
    elif system == 'Windows':
        candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun']
        font_dirs = [os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')]
        file_keywords = ['yahei', 'simhei', 'simsun']
    else:  # Linux
        candidates = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Source Han Sans SC', 'Droid Sans Fallback', 'DejaVu Sans']
        font_dirs = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.local/share/fonts')]
        file_keywords = ['NotoSansCJK', 'WenQuanYi', 'SourceHanSans', 'DroidSansFallback']

    # 1) 优先通过 family 名称严格查找
    for name in candidates:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path and os.path.exists(path):
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f'使用中文字体: {name} -> {path}')
                return name
        except Exception:
            continue

    # 2) 扫描系统字体目录，尝试动态注册（含 .ttf/.otf/.ttc）
    try:
        for d in font_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                lower = fname.lower()
                if any(k.lower() in lower for k in file_keywords) and (lower.endswith('.ttf') or lower.endswith('.otf') or lower.endswith('.ttc')):
                    fpath = os.path.join(d, fname)
                    try:
                        fm.fontManager.addfont(fpath)
                    except Exception:
                        # 某些 .ttc 可能无法直接 addfont，忽略错误继续
                        pass
        fm._rebuild()  # 刷新字体缓存
        installed = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f'使用中文字体(动态注册): {name}')
                return name
    except Exception:
        pass

    # 3) 再次尝试宽松匹配任一已安装 CJK 字体
    installed_fonts = [(f.name, getattr(f, 'fname', '')) for f in fm.fontManager.ttflist]
    for fam, fpath in installed_fonts:
        if any(k.lower() in fam.lower() for k in ['pingfang', 'hiragino', 'heiti', 'song', 'noto', 'source han', 'wqy', 'cjk', '汉', '黑体', '宋体']):
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [fam]
            plt.rcParams['axes.unicode_minus'] = False
            print(f'使用中文字体(宽松匹配): {fam} -> {fpath}')
            return fam

    # 4) 最后兜底
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print('未找到合适中文字体，使用 DejaVu Sans 兜底（可能无法显示中文）')
    return 'DejaVu Sans'

# 模块导入时先设置一次
_set_chinese_font()

class GPRModel:
    """高斯过程回归模型类"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.validation_results = {}
        
    def build_and_train(self, X: np.ndarray, y: np.ndarray, 
                       optimize_hyperparams: bool = True) -> Dict:
        """
        构建并训练GPR模型
        
        Args:
            X: 标准化后的特征矩阵
            y: 目标变量
            optimize_hyperparams: 是否优化超参数
            
        Returns:
            training_results: 训练结果字典
        """
        print(" 开始构建GPR模型...")
        
        # 1. 定义核函数
        print("   配置核函数...")
        kernel = self._build_kernel(X.shape[1])
        
        # 2. 创建GPR模型
        print("   创建GPR模型...")
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.5,  # 修正：增加正则化项，对抗过拟合
            normalize_y=True,  # 标准化目标变量
            n_restarts_optimizer=10 if optimize_hyperparams else 0,
            random_state=self.random_state
        )
        
        # 3. 训练模型
        print("   训练模型...")
        self.model.fit(X, y)
        self.is_fitted = True
        
        # 4. 模型验证
        print("   模型验证...")
        validation_results = self._validate_model(X, y)
        
        # 5. 整理训练结果
        training_results = {
            'kernel_params': self._extract_kernel_params(),
            'log_marginal_likelihood': self.model.log_marginal_likelihood_value_,
            'validation_results': validation_results,
            'model_summary': self._get_model_summary(X, y)
        }
        
        print(f" GPR模型训练完成！")
        print(f"    交叉验证R²: {validation_results['cv_r2_mean']:.4f} (±{validation_results['cv_r2_std']:.4f})")
        print(f"    训练集R²: {validation_results['train_r2']:.4f}")
        print(f"    对数边际似然: {self.model.log_marginal_likelihood_value_:.4f}")
        
        return training_results
    
    def _build_kernel(self, n_features: int):
        """构建RBF核函数"""
        # 为每个特征设置独立的长度尺度
        length_scale = np.ones(n_features)
        length_scale_bounds = (1e-2, 1e2)
        
        # RBF核 + 常数核 + 白噪声核
        kernel = (C(1.0, (1e-3, 1e3)) * 
                 RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + 
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
        
        return kernel
    
    def _validate_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """模型验证"""
        results = {}
        
        # 训练集性能
        y_pred_train = self.model.predict(X)
        results['train_r2'] = r2_score(y, y_pred_train)
        results['train_rmse'] = np.sqrt(mean_squared_error(y, y_pred_train))
        results['train_mae'] = mean_absolute_error(y, y_pred_train)
        
        # 留一交叉验证
        loo = LeaveOneOut()
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=loo, scoring='r2')
            # 过滤掉nan值
            cv_scores_clean = cv_scores[~np.isnan(cv_scores)]
            if len(cv_scores_clean) > 0:
                results['cv_r2_mean'] = cv_scores_clean.mean()
                results['cv_r2_std'] = cv_scores_clean.std()
                results['cv_scores'] = cv_scores_clean.tolist()
            else:
                # 如果所有值都是nan，回退到5折交叉验证
                cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
                results['cv_r2_mean'] = cv_scores.mean()
                results['cv_r2_std'] = cv_scores.std()
                results['cv_scores'] = cv_scores.tolist()
                print("   留一交叉验证出现数值问题，回退到5折交叉验证")
        except:
            # 如果留一交叉验证失败，使用5折交叉验证
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            results['cv_r2_mean'] = cv_scores.mean()
            results['cv_r2_std'] = cv_scores.std()
            results['cv_scores'] = cv_scores.tolist()
            print("   留一交叉验证失败，使用5折交叉验证")
        
        # RMSE交叉验证
        try:
            cv_rmse_scores = cross_val_score(self.model, X, y, cv=loo, 
                                            scoring='neg_mean_squared_error')
            cv_rmse_scores_clean = cv_rmse_scores[~np.isnan(cv_rmse_scores)]
            if len(cv_rmse_scores_clean) > 0:
                results['cv_rmse_mean'] = np.sqrt(-cv_rmse_scores_clean.mean())
                results['cv_rmse_std'] = np.sqrt(cv_rmse_scores_clean.std())
            else:
                cv_rmse_scores = cross_val_score(self.model, X, y, cv=5, 
                                                scoring='neg_mean_squared_error')
                results['cv_rmse_mean'] = np.sqrt(-cv_rmse_scores.mean())
                results['cv_rmse_std'] = np.sqrt(cv_rmse_scores.std())
        except:
            cv_rmse_scores = cross_val_score(self.model, X, y, cv=5, 
                                            scoring='neg_mean_squared_error')
            results['cv_rmse_mean'] = np.sqrt(-cv_rmse_scores.mean())
            results['cv_rmse_std'] = np.sqrt(cv_rmse_scores.std())
        
        # 模型质量评估
        results['quality_assessment'] = self._assess_model_quality(results)
        
        return results
    
    def _assess_model_quality(self, results: Dict) -> Dict:
        """评估模型质量"""
        assessment = {}
        
        cv_r2 = results['cv_r2_mean']
        cv_r2_std = results['cv_r2_std']
        
        # R²评估
        if cv_r2 > 0.8:
            assessment['r2_level'] = 'excellent'
        elif cv_r2 > 0.7:
            assessment['r2_level'] = 'good'
        elif cv_r2 > 0.5:
            assessment['r2_level'] = 'acceptable'
        else:
            assessment['r2_level'] = 'poor'
        
        # 稳定性评估
        if cv_r2_std < 0.05:
            assessment['stability'] = 'high'
        elif cv_r2_std < 0.1:
            assessment['stability'] = 'medium'
        else:
            assessment['stability'] = 'low'
        
        # 过拟合检查
        train_r2 = results['train_r2']
        overfitting_gap = train_r2 - cv_r2
        if overfitting_gap < 0.05:
            assessment['overfitting_risk'] = 'low'
        elif overfitting_gap < 0.15:
            assessment['overfitting_risk'] = 'medium'
        else:
            assessment['overfitting_risk'] = 'high'
        
        # 总体评估
        if (assessment['r2_level'] in ['excellent', 'good'] and 
            assessment['stability'] in ['high', 'medium'] and
            assessment['overfitting_risk'] in ['low', 'medium']):
            assessment['overall'] = 'suitable_for_optimization'
        else:
            assessment['overall'] = 'needs_improvement'
        
        return assessment
    
    def _extract_kernel_params(self) -> Dict:
        """提取核函数参数"""
        if not self.is_fitted:
            return {}
        
        kernel = self.model.kernel_
        params = {}
        
        # 提取各组件参数
        if hasattr(kernel, 'k1') and hasattr(kernel, 'k2'):
            # 复合核函数
            k1 = kernel.k1  # ConstantKernel * RBF
            k2 = kernel.k2  # WhiteKernel
            
            if hasattr(k1, 'k1') and hasattr(k1, 'k2'):
                # ConstantKernel
                params['constant_value'] = float(k1.k1.constant_value)
                # RBF
                params['length_scale'] = k1.k2.length_scale.tolist()
            
            # WhiteKernel
            if hasattr(k2, 'noise_level'):
                params['noise_level'] = float(k2.noise_level)
        
        return params
    
    def _get_model_summary(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """获取模型摘要"""
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_range': [float(y.min()), float(y.max())],
            'target_mean': float(y.mean()),
            'target_std': float(y.std())
        }
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测
        
        Args:
            X: 输入特征（标准化后）
            return_std: 是否返回预测标准差
            
        Returns:
            y_pred: 预测均值
            y_std: 预测标准差（如果return_std=True）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用build_and_train方法")
        
        if return_std:
            y_pred, y_std = self.model.predict(X, return_std=True)
            return y_pred, y_std
        else:
            y_pred = self.model.predict(X, return_std=False)
            return y_pred, None
    
    def plot_validation_results(self, X: np.ndarray, y: np.ndarray, save_path: str = None):
        """绘制验证结果"""
        if not self.is_fitted:
            print(" 模型尚未训练，无法绘制验证结果")
            return
        
        # 预测
        y_pred, y_std = self.predict(X, return_std=True)
        
        # 创建图形
        _set_chinese_font()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('GPR模型验证结果', fontsize=16, fontweight='bold')
        
        # 1. 预测vs实际
        axes[0, 0].scatter(y, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('实际值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title(f'预测vs实际 (R²={r2_score(y, y_pred):.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = y - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 预测不确定性
        sorted_indices = np.argsort(y_pred)
        axes[1, 0].fill_between(range(len(y)), 
                               (y_pred - 1.96*y_std)[sorted_indices],
                               (y_pred + 1.96*y_std)[sorted_indices],
                               alpha=0.3, color='gray', label='95%置信区间')
        axes[1, 0].scatter(range(len(y)), y[sorted_indices], alpha=0.6, color='blue', label='实际值')
        axes[1, 0].plot(range(len(y)), y_pred[sorted_indices], color='red', label='预测值')
        axes[1, 0].set_xlabel('样本索引（按预测值排序）')
        axes[1, 0].set_ylabel('C4烯烃收率')
        axes[1, 0].set_title('预测不确定性')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 交叉验证结果
        cv_scores = self.validation_results.get('cv_scores', [])
        if cv_scores:
            axes[1, 1].bar(range(1, len(cv_scores)+1), cv_scores, color='orange', alpha=0.7)
            axes[1, 1].axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                              label=f'平均R²={np.mean(cv_scores):.3f}')
            axes[1, 1].set_xlabel('交叉验证折数')
            axes[1, 1].set_ylabel('R²分数')
            axes[1, 1].set_title('交叉验证结果')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" 验证结果图已保存: {save_path}")
        
        plt.show()
    
    def save_model_info(self, filepath: str, training_results: Dict):
        """保存模型信息"""
        model_info = {
            'model_type': 'GaussianProcessRegressor',
            'kernel_type': 'RBF + Constant + WhiteKernel',
            'training_results': training_results,
            'is_fitted': self.is_fitted
        }
        
        # 转换为DataFrame并保存
        info_df = pd.DataFrame([model_info])
        info_df.to_csv(filepath, index=False)
        print(f" 模型信息已保存: {filepath}")

def main():
    """测试GPR模型模块"""
    from data_processor import DataProcessor
    
    print(" 测试GPR模型模块...")
    
    # 1. 加载数据
    processor = DataProcessor()
    try:
        X, y, data_info = processor.load_and_prepare_data('附件1.csv', '每组指标.csv')
        
        # 2. 构建和训练GPR模型
        gpr = GPRModel(random_state=42)
        training_results = gpr.build_and_train(X, y)
        
        # 3. 测试预测
        y_pred, y_std = gpr.predict(X[:5], return_std=True)
        print(f"\n🔮 预测测试:")
        print(f"前5个样本预测值: {y_pred}")
        print(f"预测标准差: {y_std}")
        
        # 4. 绘制验证结果
        gpr.validation_results = training_results['validation_results']
        gpr.plot_validation_results(X, y, 'gpr_validation_results.png')
        
        # 5. 保存模型信息
        gpr.save_model_info('gpr_model_info.csv', training_results)
        
        print("\n GPR模型模块测试通过！")
        
    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()