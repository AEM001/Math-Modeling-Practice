#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的测线长度脚本
基于现有成功的测线生成，对测线长度进行智能边界扩展优化
确保测线覆盖宽度能够完全覆盖边界内的区域
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_depth_interpolator():
    """加载深度插值器"""
    df = pd.read_csv('../_00_source_data/output.csv')
    x_nm = np.array(df['横坐标'].values, dtype=float)
    y_nm = np.array(df['纵坐标'].values, dtype=float) 
    depth = np.array(df['深度'].values, dtype=float)
    return LinearNDInterpolator(list(zip(x_nm, y_nm)), depth)

def calculate_swath_width(depth, beam_angle=120):
    """根据水深计算条带宽度"""
    swath_width_m = 2 * depth * np.tan(np.radians(beam_angle / 2))
    swath_width_nm = swath_width_m / 1852
    return min(swath_width_nm, 0.2)  # 限制最大宽度0.2海里

def find_line_region_intersections_extended(x_start, y_start, x_end, y_end, 
                                          x_min, x_max, y_min, y_max, depth_interpolator):
    """
    计算测线与矩形区域边界的交点，并根据覆盖宽度进行智能延伸
    确保测线的覆盖宽度能够完全覆盖边界内的区域
    采用更保守的延伸策略以最小化漏测
    """
    # 首先找到基本的边界交点
    def get_boundary_intersections(x1, y1, x2, y2):
        """计算直线与矩形边界的所有交点"""
        intersections = []
        
        # 与各边界的交点
        # 左边界 x = x_min
        if x2 != x1:
            t = (x_min - x1) / (x2 - x1)
            if 0 <= t <= 1:
                y = y1 + t * (y2 - y1)
                if y_min <= y <= y_max:
                    intersections.append((x_min, y, t, 'left'))
        
        # 右边界 x = x_max  
        if x2 != x1:
            t = (x_max - x1) / (x2 - x1)
            if 0 <= t <= 1:
                y = y1 + t * (y2 - y1)
                if y_min <= y <= y_max:
                    intersections.append((x_max, y, t, 'right'))
        
        # 下边界 y = y_min
        if y2 != y1:
            t = (y_min - y1) / (y2 - y1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if x_min <= x <= x_max:
                    intersections.append((x, y_min, t, 'bottom'))
        
        # 上边界 y = y_max
        if y2 != y1:
            t = (y_max - y1) / (y2 - y1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if x_min <= x <= x_max:
                    intersections.append((x, y_max, t, 'top'))
        
        return sorted(intersections, key=lambda x: x[2])  # 按参数t排序
    
    # 获取边界交点
    intersections = get_boundary_intersections(x_start, y_start, x_end, y_end)
    
    if len(intersections) < 2:
        return None
    
    # 取前两个交点作为基本的裁剪结果
    x1, y1 = intersections[0][0], intersections[0][1]
    x2, y2 = intersections[-1][0], intersections[-1][1]
    
    # 计算测线方向向量
    line_length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    if line_length == 0:
        return None
        
    dx = (x_end - x_start) / line_length
    dy = (y_end - y_start) / line_length
    
    # 计算测线与边界的夹角（用于调整延伸距离）
    def calculate_angle_factor(boundary_type, dx, dy):
        """根据测线与边界的夹角计算延伸系数"""
        if boundary_type in ['left', 'right']:
            # 与垂直边界的夹角，dx越小角度越小，需要更大延伸
            angle_factor = 1.0 / max(abs(dx), 0.1)  # 防止除零
        else:  # top, bottom
            # 与水平边界的夹角，dy越小角度越小，需要更大延伸  
            angle_factor = 1.0 / max(abs(dy), 0.1)
        
        return min(angle_factor, 3.0)  # 限制最大系数为3
    
    # 在交点处采样深度，计算需要的延伸距离
    def get_extension_distance(x, y, boundary_type):
        """计算在某个边界点需要的延伸距离 - 保守策略"""
        # 查询该点的深度
        depth = depth_interpolator(x, y)
        if np.isnan(depth):
            depth = 110  # 使用平均深度作为备选
            
        # 计算覆盖宽度
        swath_width = calculate_swath_width(depth)
        
        # 角度修正系数
        angle_factor = calculate_angle_factor(boundary_type, dx, dy)
        
        # 保守延伸策略：
        # 1. 基础延伸 = 完整覆盖宽度
        # 2. 角度修正：根据入射角增加延伸
        # 3. 边界安全边距：额外增加20%
        base_extension = swath_width  # 使用完整宽度而不是一半
        angle_extension = base_extension * (angle_factor - 1) * 0.5  # 角度修正
        safety_margin = swath_width * 0.2  # 20%安全边距
        
        total_extension = base_extension + angle_extension + safety_margin
        
        return total_extension
    
    # 计算起点延伸距离
    boundary_type_start = intersections[0][3]
    ext_dist_start = get_extension_distance(x1, y1, boundary_type_start)
    
    # 计算终点延伸距离  
    boundary_type_end = intersections[-1][3]
    ext_dist_end = get_extension_distance(x2, y2, boundary_type_end)
    
    # 向外延伸测线（更保守的延伸）
    x1_ext = x1 - dx * ext_dist_start
    y1_ext = y1 - dy * ext_dist_start
    x2_ext = x2 + dx * ext_dist_end  
    y2_ext = y2 + dy * ext_dist_end
    
    return x1_ext, y1_ext, x2_ext, y2_ext

def optimize_survey_lines(lines_df, region_boundaries, depth_interpolator):
    """
    对现有的测线进行智能长度优化
    """
    optimized_lines = []
    
    for _, line in lines_df.iterrows():
        region_id = line['region_id']
        
        # 获取区域边界
        region_bounds = region_boundaries[region_boundaries['区域编号'] == region_id].iloc[0]
        x_min, x_max = region_bounds['X_min'], region_bounds['X_max']
        y_min, y_max = region_bounds['Y_min'], region_bounds['Y_max']
        
        # 对测线进行智能边界延伸
        result = find_line_region_intersections_extended(
            line['x_start_nm'], line['y_start_nm'],
            line['x_end_nm'], line['y_end_nm'],
            x_min, x_max, y_min, y_max, depth_interpolator
        )
        
        if result is not None:
            x1, y1, x2, y2 = result
            # 计算优化后的长度
            optimized_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            optimized_line = line.copy()
            optimized_line['x_start_nm'] = x1
            optimized_line['y_start_nm'] = y1
            optimized_line['x_end_nm'] = x2
            optimized_line['y_end_nm'] = y2
            optimized_line['length_optimized_nm'] = optimized_length
            
            optimized_lines.append(optimized_line)
    
    return pd.DataFrame(optimized_lines)

def visualize_survey_plan(lines_df, region_boundaries, grid_data, 
                         output_path="survey_plan_q4.png"):
    """创建优化后的测线方案可视化图."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # 绘制海深数据点
    ax.scatter(grid_data['横坐标'], grid_data['纵坐标'], c=grid_data['深度'], 
                cmap='ocean_r', s=1, alpha=0.6, label='海深数据点')
    
    # 绘制测线
    for _, line in lines_df.iterrows():
        ax.plot([line['x_start_nm'], line['x_end_nm']], [line['y_start_nm'], line['y_end_nm']], 
                'r-', lw=0.8, alpha=0.9)

    # 绘制区域边界
    for _, region in region_boundaries.iterrows():
        x_min, x_max = region['X_min'], region['X_max']
        y_min, y_max = region['Y_min'], region['Y_max']
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x_min + 0.05, y_min + 0.05, str(region['区域编号']), 
                fontsize=14, weight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel('东西方向坐标 (海里)', fontsize=12)
    ax.set_ylabel('南北方向坐标 (海里)', fontsize=12)
    ax.set_title('优化测线方案', fontsize=16, weight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='测线'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='区域边界')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"测线方案可视化图像已保存至 {output_path}")

def main():
    """Main function to run the optimized survey line planning pipeline."""
    # 1. Load data
    print("正在加载数据...")
    try:
        lines_df_original = pd.read_csv("survey_lines_q4.csv")
        grid_df = pd.read_csv('../_00_source_data/output.csv')
        depth_interpolator = load_depth_interpolator()
    except FileNotFoundError as e:
        print(f"错误: 必需的数据文件未找到 - {e}")
        return

    # Region boundaries
    region_boundaries_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'X_min': [0.00, 0.98, 0.00, 1.99, 2.99, 1.99, 2.99],
        'X_max': [0.98, 1.99, 1.99, 2.99, 4.00, 2.99, 4.00],
        'Y_min': [0.00, 0.00, 2.49, 0.00, 0.00, 2.49, 2.49],
        'Y_max': [2.49, 2.49, 5.00, 2.49, 2.49, 5.00, 5.00]
    }
    region_boundaries_df = pd.DataFrame(region_boundaries_data)

    # 2. Optimize survey lines
    print("正在进行智能测线长度优化...")
    lines_df_optimized = optimize_survey_lines(lines_df_original, region_boundaries_df, depth_interpolator)
    lines_df_optimized.to_csv("survey_lines_q4_optimized.csv", index=False, float_format='%.4f')
    print("优化的测线数据已保存至 survey_lines_q4_optimized.csv")

    # 3. Generate optimization report
    print("正在生成优化报告...")
    total_length = lines_df_optimized['length_optimized_nm'].sum()

    report_content = "# 智能测线边界优化方案报告（保守延伸版本）\n\n"
    report_content += "## 优化策略说明\n\n"
    report_content += "采用高度保守的智能边界延伸算法，根据测线覆盖宽度和入射角度动态调整测线长度。"
    report_content += "测线在区域边界处会大幅延伸，确保其覆盖宽度能够完全覆盖边界内的区域，"
    report_content += "最大程度减少边界处的漏测问题。\n\n"
    report_content += "**核心技术升级**：\n"
    report_content += "- 基础延伸距离 = 完整覆盖宽度（而非半宽）\n"
    report_content += "- 角度修正：测线与边界夹角越小，延伸越多\n" 
    report_content += "- 安全边距：额外增加20%覆盖宽度作为安全缓冲\n"
    report_content += "- 综合延伸策略确保在各种角度下的完全覆盖\n\n"
    
    summary_data = []
    for region_id in lines_df_optimized['region_id'].unique():
        region_lines = lines_df_optimized[lines_df_optimized['region_id'] == region_id]
        
        length_total = region_lines['length_optimized_nm'].sum()
        
        summary_data.append({
            '区域编号': region_id,
            '测线数量': len(region_lines),
            '总长度(海里)': length_total,
            '总长度(公里)': length_total * 1.852
        })
    
    summary_df = pd.DataFrame(summary_data)
    table_md = summary_df.to_markdown(index=False, floatfmt='.2f')
    if table_md:
        report_content += table_md

    report_content += f"\n\n**整体统计**:\n"
    report_content += f"- 总测线数量: {len(lines_df_optimized)} 条\n"
    report_content += f"- 总测线长度: {total_length:.2f} 海里 ({total_length * 1.852:.2f} 公里)\n\n"
    
    report_content += "**技术优势**: \n"
    report_content += "1. 根据实际水深动态调整延伸距离\n"
    report_content += "2. 有效解决边界角度导致的漏测问题\n"
    report_content += "3. 保持测线平行性和覆盖效果\n"
    report_content += "4. 优化实际作业效率和覆盖质量\n\n"
    
    with open("survey_optimization_report.md", "w", encoding='utf-8') as f:
        f.write(report_content)
    print("优化报告已生成: survey_optimization_report.md")

    # 4. Create visualization
    print("正在生成可视化...")
    visualize_survey_plan(lines_df_optimized, region_boundaries_df, grid_df)
    
    print(f"\n=== 保守延伸策略优化方案总结 ===")
    print(f"总测线数量: {len(lines_df_optimized)} 条")
    print(f"总测线长度: {total_length:.2f} 海里 ({total_length * 1.852:.2f} 公里)")
    print(f"采用保守边界延伸策略，最大化减少漏测")

if __name__ == '__main__':
    main() 