import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataAuditor:
    
    def __init__(self, config_path='config/config.json'):
        """初始化审计器"""
        self.config = self.load_config(config_path)
        self.data_paths = self.config['data_paths']
        self.output_paths = self.config['output_paths']
        self.clean_config = self.config['data_cleaning']
        self.audit_stats = {}
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_and_audit_data(self):
        """加载并审计数据"""
        print("正在加载数据...")
        
        # 加载原始数据
        # Load original data
        data_path = self.data_paths['raw_item_data']
        raw_data = pd.read_csv(data_path)
        raw_data['销售日期'] = pd.to_datetime(raw_data['销售日期'])
        
        print(f"原始数据: {len(raw_data):,} 条记录")
        
        # 基础统计
        self.audit_stats = {
            'original_records': len(raw_data),
            'date_range': f"{raw_data['销售日期'].min()} 至 {raw_data['销售日期'].max()}",
            'unique_items': raw_data['单品编码'].nunique(),
            'unique_categories': raw_data['分类名称'].nunique()
        }
        
        return raw_data
    
    def clean_data(self, data):
        """清洗数据"""
        print("开始数据清洗...")
        
        clean_data = data.copy()
        cleaning_log = []
        
        # 1. 移除零价格/零销量
        before = len(clean_data)
        clean_data = clean_data[
            (clean_data['正常销售单价(元/千克)'] > 0) & 
            (clean_data['正常销量(千克)'] > 0)
        ]
        excluded = before - len(clean_data)
        cleaning_log.append(f"排除零价格/零销量: {excluded} 条")
        
        # 2. 移除负成本加成率
        before = len(clean_data)
        clean_data = clean_data[clean_data['成本加成率'] >= 0]
        excluded = before - len(clean_data)
        cleaning_log.append(f"排除负成本加成率: {excluded} 条")
        
        # 3. 移除极端价格（超过批发价3倍）
        before = len(clean_data)
        clean_data = clean_data[
            clean_data['正常销售单价(元/千克)'] <= 
            self.clean_config['price_multiplier_threshold'] * clean_data['批发价格(元/千克)']
        ]
        excluded = before - len(clean_data)
        cleaning_log.append(f"排除极端价格: {excluded} 条")
        
        # 4. 移除销量异常值（简化版MAD检测）
        before = len(clean_data)
        quantity_outliers = 0
        for category in clean_data['分类名称'].unique():
            cat_data = clean_data[clean_data['分类名称'] == category]
            if len(cat_data) > 20:
                q75, q25 = np.percentile(cat_data['正常销量(千克)'], [75, 25])
                iqr = q75 - q25
                upper_bound = q75 + 3 * iqr  # 使用3倍IQR作为阈值
                outlier_mask = (clean_data['分类名称'] == category) & \
                              (clean_data['正常销量(千克)'] > upper_bound)
                clean_data = clean_data[~outlier_mask]
                quantity_outliers += outlier_mask.sum()
        
        excluded = before - len(clean_data)
        cleaning_log.append(f"排除销量异常值: {excluded} 条")
        
        # 标记促销
        clean_data['is_promotion'] = (clean_data['打折销量(千克)'] > 0).astype(int)
        
        print(f"清洗完成: {len(clean_data):,} 条记录保留")
        
        self.audit_stats.update({
            'cleaned_records': len(clean_data),
            'retention_rate': len(clean_data) / self.audit_stats['original_records'],
            'cleaning_log': cleaning_log
        })
        
        return clean_data
    
    def split_and_save_data(self, clean_data):
        """分割并保存数据"""
        print("分割并保存数据...")
        
        # 按时间排序并分割
        clean_data_sorted = clean_data.sort_values('销售日期')
        split_ratio = self.clean_config['train_split_ratio']
        split_date = clean_data_sorted['销售日期'].quantile(split_ratio)
        
        train_data = clean_data_sorted[clean_data_sorted['销售日期'] <= split_date]
        test_data = clean_data_sorted[clean_data_sorted['销售日期'] > split_date]
        
        # 确保目录存在
        os.makedirs(self.data_paths['processed_data_dir'], exist_ok=True)
        
        # 保存文件
        train_path = self.data_paths['train_data']
        test_path = self.data_paths['test_data']
        
        train_data.to_csv(train_path, index=False, encoding='utf-8')
        test_data.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"训练集: {len(train_data):,} 条记录 -> {train_path}")
        print(f"测试集: {len(test_data):,} 条记录 -> {test_path}")
        
        return train_data, test_data
    
    def generate_report(self):
        """生成简化报告"""
        report_content = [
            "# 数据质量审计报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 数据概览",
            f"- 原始记录数: {self.audit_stats['original_records']:,}",
            f"- 数据时间范围: {self.audit_stats['date_range']}",
            f"- 单品数量: {self.audit_stats['unique_items']}",
            f"- 品类数量: {self.audit_stats['unique_categories']}",
            "",
            "## 清洗结果",
            f"- 清洗后记录数: {self.audit_stats['cleaned_records']:,}",
            f"- 数据保留率: {self.audit_stats['retention_rate']:.1%}",
            "",
            "## 清洗过程",
        ]
        
        for log_entry in self.audit_stats['cleaning_log']:
            report_content.append(f"- {log_entry}")
        
        # 保存报告
        os.makedirs(self.output_paths['reports_dir'], exist_ok=True)
        report_path = os.path.join(self.output_paths['reports_dir'], 'data_audit_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"审计报告已保存: {report_path}")
    
    def run_full_audit(self):
        """运行完整审计流程"""
        print("=== 开始数据质量审计 ===")
        
        # 加载和审计
        raw_data = self.load_and_audit_data()
        
        # 清洗数据
        clean_data = self.clean_data(raw_data)
        
        # 分割保存
        self.split_and_save_data(clean_data)
        
        # 生成报告
        self.generate_report()
        
        print("=== 数据质量审计完成 ===\n")
        return True

if __name__ == "__main__":
    auditor = DataAuditor()
    auditor.run_full_audit()
