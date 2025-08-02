## 问题三代码 - 智能优化版
"""
舞龙队伍调头空间螺距优化分析
目标：找到在调头空间外不会发生碰撞的最小螺距
"""

import numpy as np

def binary_search_zero(func, left_bound, right_bound, tolerance, *args):
    """二分法求零点 - 高效数值求解"""
    while abs(right_bound - left_bound) > tolerance:
        mid_point = (left_bound + right_bound) / 2
        if func(mid_point, *args) == 0:
            return mid_point
        elif func(left_bound, *args) * func(mid_point, *args) < 0:
            right_bound = mid_point
        else:
            left_bound = mid_point
    return (left_bound + right_bound) / 2

def calculate_dragon_positions(start_theta, spiral_pitch, max_segments=223):
    """
    计算舞龙队伍各节点的位置轨迹
    使用阿基米德螺线模型和几何约束求解
    """
    # 物理参数
    DRAGON_RADIUS = 0.275
    FIRST_SEGMENT_DISTANCE = 3.41 - DRAGON_RADIUS * 2
    REGULAR_SEGMENT_DISTANCE = 2.2 - DRAGON_RADIUS * 2
    
    theta_positions = [start_theta]
    
    for i in range(max_segments):
        segment_distance = FIRST_SEGMENT_DISTANCE if i == 0 else REGULAR_SEGMENT_DISTANCE
        previous_theta = theta_positions[-1]
        
        # 几何约束方程：相邻节点间的距离约束
        def distance_constraint(theta):
            return (theta**2 + previous_theta**2 - 
                   2*theta*previous_theta*np.cos(theta - previous_theta) - 
                   4*np.pi**2*segment_distance**2/spiral_pitch**2)
        
        # 求解下一个节点位置
        next_theta = binary_search_zero(
            distance_constraint, 
            previous_theta, 
            previous_theta + np.pi/2, 
            1e-8
        )
        theta_positions.append(next_theta)
        
        # 三圈终止条件
        if next_theta - start_theta >= 3*np.pi:
            break
    
    return theta_positions

def convert_to_cartesian(theta_positions, spiral_pitch):
    """将极坐标转换为笛卡尔坐标"""
    coordinates = []
    for theta in theta_positions:
        radius = theta * spiral_pitch / (2 * np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        coordinates.append((x, y))
    return coordinates

def calculate_path_slopes(coordinates):
    """计算路径各段的斜率"""
    slopes = []
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        slope = (y1 - y2) / (x1 - x2)
        slopes.append(slope)
    return slopes

def check_collision_with_benches(start_theta, coordinates, slopes):
    """
    检测与板凳的碰撞 - 使用几何光学反射模型
    检查四个关键反射路径
    """
    DRAGON_RADIUS = 0.275
    BENCH_RADIUS = 0.15
    
    def calculate_reflection_path(base_slope, base_x, base_y, reflection_type, target_x, target_y):
        """计算反射路径"""
        # 反射系数计算
        if reflection_type == "positive":
            reflected_slope = (BENCH_RADIUS/DRAGON_RADIUS + base_slope) / (1 - BENCH_RADIUS*base_slope/DRAGON_RADIUS)
        else:
            reflected_slope = (base_slope - BENCH_RADIUS/DRAGON_RADIUS) / (1 + BENCH_RADIUS*base_slope/DRAGON_RADIUS)
        
        # 计算截距
        intercept = BENCH_RADIUS*np.sqrt(base_slope**2 + 1) + base_y - base_slope*base_x
        if np.abs(intercept) <= np.abs(base_y - base_slope*base_x):
            intercept = -BENCH_RADIUS*np.sqrt(base_slope**2 + 1) + base_y - base_slope*base_x
        
        # 计算交点
        if reflection_type == "positive":
            intersection_x = (base_y - reflected_slope*base_x - intercept) / (base_slope - reflected_slope)
            intersection_y = (base_slope*base_y - base_slope*reflected_slope*base_x - reflected_slope*intercept) / (base_slope - reflected_slope)
        else:
            intersection_x = (target_y - reflected_slope*target_x - intercept) / (base_slope - reflected_slope)
            intersection_y = (base_slope*target_y - base_slope*reflected_slope*target_x - intercept*reflected_slope) / (base_slope - reflected_slope)
        
        return intersection_x, intersection_y
    
    # 检查四个关键反射路径
    collision_cases = [
        (0, 0, "positive", 1),  # 第一节点正反射到第二节点
        (0, 0, "negative", 1),  # 第一节点负反射到第二节点
        (1, 1, "positive", 2),  # 第二节点正反射到第三节点  
        (1, 1, "negative", 2),  # 第二节点负反射到第三节点
    ]
    
    for base_idx, slope_idx, reflection_type, target_idx in collision_cases:
        if len(coordinates) <= target_idx or len(slopes) <= slope_idx:
            continue
            
        base_x, base_y = coordinates[base_idx]
        target_x, target_y = coordinates[target_idx]
        base_slope = slopes[slope_idx]
        
        intersection_x, intersection_y = calculate_reflection_path(
            base_slope, base_x, base_y, reflection_type, target_x, target_y
        )
        
        # 检查后续路径是否与板凳发生碰撞
        for i, (current_x, current_y) in enumerate(coordinates[:-1]):
            if len(theta_positions) > i+1 and theta_positions[i+1] - start_theta >= np.pi:
                current_slope = slopes[i]
                
                # 点到直线距离公式
                distance_to_bench = (abs(current_slope*(intersection_x - current_x) + current_y - intersection_y) / 
                                   np.sqrt(current_slope**2 + 1))
                
                if distance_to_bench < BENCH_RADIUS:
                    return True
    
    return False

def optimize_spiral_pitch(initial_velocity=1.0, turning_diameter=9.0):
    """
    智能优化螺距 - 多级精度搜索算法
    """
    print("=" * 60)
    print("舞调头空间螺距优化开始")
    print(f"初始参数: 龙头速度={initial_velocity} m/s, 调头空间直径={turning_diameter} m")
    print("=" * 60)
    
    # 多级精度搜索配置
    search_levels = [
        {"name": "粗略搜索", "step": 0.01, "range": 0.01},
        {"name": "精细搜索", "step": 0.0001, "range": 0.01},
        {"name": "高精度搜索", "step": 0.00001, "range": 0.0001},
        {"name": "超精度搜索", "step": 0.000001, "range": 0.00001},
        {"name": "极限精度搜索", "step": 0.0000001, "range": 0.000001}
    ]
    
    optimal_pitch = 0.55  # 初始搜索起点
    
    for level_idx, config in enumerate(search_levels):
        print(f"\n🔍 第{level_idx + 1}轮 {config['name']} (步长: {config['step']})")
        
        if level_idx == 0:
            # 第一轮从0.55开始向下搜索
            search_range = np.arange(0.55, 0.4, -config['step'])
        else:
            # 后续轮次在最优值附近精细搜索
            search_range = np.arange(
                optimal_pitch + config['range'], 
                optimal_pitch - config['range'], 
                -config['step']
            )
        
        collision_found = False
        tested_count = 0
        
        for pitch in search_range:
            tested_count += 1
            min_theta = turning_diameter * np.pi / pitch
            
            # 测试该螺距下的多个极角
            for theta in np.arange(min_theta + 6, min_theta, -0.1):
                # 计算龙身轨迹
                global theta_positions  # 为了在碰撞检测中使用
                theta_positions = calculate_dragon_positions(theta, pitch)
                coordinates = convert_to_cartesian(theta_positions, pitch)
                slopes = calculate_path_slopes(coordinates)
                
                # 检测碰撞
                if check_collision_with_benches(theta, coordinates, slopes):
                    collision_found = True
                    break
            
            if collision_found:
                optimal_pitch = pitch
                print(f"发现碰撞螺距: {pitch:.7f} (测试了 {tested_count} 个值)")
                break
        
        if not collision_found:
            print(f"未发现碰撞 (测试了 {tested_count} 个值)")
            break
    
    return optimal_pitch

# ========== 主程序执行 ==========
if __name__ == "__main__":
    try:
        # 开始优化计算
        result = optimize_spiral_pitch()
        
        print("\n" + "=" * 60)
        print("🎯 优化结果")
        print("=" * 60)
        print(f"在调头空间外不会发生碰撞的最小螺距: {result:.7f} m")
        
        # 验证结果
        print(f"\n结果验证:")
        min_theta = 9.0 * np.pi / result
        print(f"   对应的最小极角: {min_theta:.4f} rad ({min_theta*180/np.pi:.2f}°)")
        print(f"   螺线紧密度: {2*np.pi/result:.4f} rad/m")
        
        print("\n优化完成!")
        
    except Exception as e:
        print(f"计算过程中发生错误: {e}")
        raise