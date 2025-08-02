import numpy as np
from tqdm import tqdm

# ==============================================================================
# SECTION 0: 统一的系统常量定义（与前四问保持一致）
# ==============================================================================
NUM_LINKS = 223  # 连杆数量
NUM_POINTS = NUM_LINKS + 1  # 铰接点数量（224个）
HINGE_OFFSET = 0.275  # 铰接点距连杆端部的偏移量 (m)

# 连杆物理长度
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41  # 第一个连杆（龙头段）特殊长度

# 铰接点之间的有效长度
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)

# 龙形路径常数
SPIRAL_PITCH = 1.7
TURN_DIAMETER = 9
THETA0_START = 16.6319611
ARC_RADIUS = 1.5027088
ARC_ANGLE = 3.0214868

# 圆弧中心坐标
ARC1_CENTER = (-0.7600091, -1.3057264)
ARC2_CENTER = (1.7359325, 2.4484020)
ARC1_THETA = 4.0055376
ARC2_THETA = 0.8639449

# 关键位置参数
DRAGON_HEAD_PARAMS = {
    'd0': effective_link_lengths[0],
    'theta_1': 0.9917636,
    'theta_2': 2.5168977,
    'theta_3': 14.1235657
}

DRAGON_BODY_PARAMS = {
    'd0': effective_link_lengths[1],
    'theta_1': 0.5561483,
    'theta_2': 1.1623551,
    'theta_3': 13.8544471
}

# ==============================================================================
# SECTION 1: 核心数学函数（从问题4复用）
# ==============================================================================

def spiral_arc_length_integral(theta):
    """螺线弧长积分函数"""
    return theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))

def spiral_tangent_slope(theta):
    """螺线切线斜率"""
    return (np.sin(theta) + theta*np.cos(theta)) / (np.cos(theta) - theta*np.sin(theta))

def distance_equation_inward(theta, theta_ref, link_length):
    """盘入螺线上两点距离方程"""
    return (theta**2 + theta_ref**2 - 2*theta*theta_ref*np.cos(theta - theta_ref) 
            - 4*np.pi**2*link_length**2/SPIRAL_PITCH**2)

def distance_equation_outward(theta, theta_ref, link_length):
    """盘出螺线上两点距离方程"""
    theta_adj = theta + np.pi
    theta_ref_adj = theta_ref + np.pi
    return (theta_adj**2 + theta_ref_adj**2 - 2*theta_adj*theta_ref_adj*np.cos(theta_adj - theta_ref_adj)
            - 4*np.pi**2*link_length**2/SPIRAL_PITCH**2)

def distance_equation_transition(theta, theta_ref, link_length, l, gamma):
    """过渡区距离方程"""
    return (l**2 + SPIRAL_PITCH**2*theta**2/(4*np.pi**2) 
            - SPIRAL_PITCH*l*theta*np.cos(theta - theta_ref + gamma)/np.pi - link_length**2)

def bisection_solve(func, a, b, eps=1e-8, *args):
    """二分法求解器"""
    while abs(b - a) > eps:
        c = (a + b) / 2
        if func(c, *args) == 0:
            return c
        elif func(a, *args) * func(c, *args) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def get_coordinates_from_theta(theta, curve_flag):
    """根据theta和曲线类型计算坐标"""
    if curve_flag == 1:  # 盘入螺线
        p = SPIRAL_PITCH * theta / (2 * np.pi)
        return p * np.cos(theta), p * np.sin(theta)
    elif curve_flag == 2:  # 第一段圆弧
        return (ARC1_CENTER[0] + 2*ARC_RADIUS*np.cos(ARC1_THETA - theta),
                ARC1_CENTER[1] + 2*ARC_RADIUS*np.sin(ARC1_THETA - theta))
    elif curve_flag == 3:  # 第二段圆弧
        return (ARC2_CENTER[0] + ARC_RADIUS*np.cos(ARC2_THETA + theta - ARC_ANGLE),
                ARC2_CENTER[1] + ARC_RADIUS*np.sin(ARC2_THETA + theta - ARC_ANGLE))
    else:  # 盘出螺线
        p = SPIRAL_PITCH * (theta + np.pi) / (2 * np.pi)
        return p * np.cos(theta), p * np.sin(theta)

# ==============================================================================
# SECTION 2: 二分法求解器
# ==============================================================================

def bisection_solver(f, a, b, eps, *args):
    """通用二分法求解器"""
    while abs(b - a) > eps:
        c = (a + b) / 2
        if f(c, *args) == 0:
            return c
        elif f(a, *args) * f(c, *args) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# ==============================================================================
# SECTION 3: 位置和速度计算函数
# ==============================================================================

def calculate_next_link_position(theta_prev, flag_prev, is_head_segment):
    """计算下一节连杆的位置参数"""
    params = DRAGON_HEAD_PARAMS if is_head_segment else DRAGON_BODY_PARAMS
    d0, theta_1, theta_2, theta_3 = params['d0'], params['theta_1'], params['theta_2'], params['theta_3']
    
    if flag_prev == 1:  # 前一节在盘入螺线
        theta = bisection_solve(distance_equation_inward, theta_prev, theta_prev + np.pi/2, 1e-8, theta_prev, d0)
        return theta, 1
        
    elif flag_prev == 2:  # 前一节在第一段圆弧
        if theta_prev < theta_1:
            # 过渡到盘入螺线
            b = np.sqrt(2 - 2*np.cos(theta_prev)) * ARC_RADIUS * 2
            beta = (ARC_ANGLE - theta_prev) / 2
            l = np.sqrt(b**2 + TURN_DIAMETER**2/4 - b*TURN_DIAMETER*np.cos(beta))
            gamma = np.arcsin(np.clip(b*np.sin(beta)/l, -1.0, 1.0))
            theta = bisection_solve(distance_equation_transition, THETA0_START, THETA0_START + np.pi/2, 
                                  1e-8, THETA0_START, d0, l, gamma)
            return theta, 1
        else:
            # 继续在第一段圆弧
            return theta_prev - theta_1, 2
            
    elif flag_prev == 3:  # 前一节在第二段圆弧
        if theta_prev < theta_2:
            # 过渡到第一段圆弧
            a = np.sqrt(10 - 6*np.cos(theta_prev)) * ARC_RADIUS
            phi = np.arccos(np.clip((4*ARC_RADIUS**2 + a**2 - d0**2)/(4*a*ARC_RADIUS), -1.0, 1.0))
            beta = np.arcsin(np.clip(ARC_RADIUS*np.sin(theta_prev)/a, -1.0, 1.0))
            return ARC_ANGLE - phi + beta, 2
        else:
            # 继续在第二段圆弧
            return theta_prev - theta_2, 3
            
    else:  # flag_prev == 4, 前一节在盘出螺线
        if theta_prev < theta_3:
            # 过渡到第二段圆弧
            p = SPIRAL_PITCH * (theta_prev + np.pi) / (2*np.pi)
            sqrt_arg = max(0, p**2 + TURN_DIAMETER**2/4 - p*TURN_DIAMETER*np.cos(theta_prev - THETA0_START + np.pi))
            a = np.sqrt(sqrt_arg)
            
            if a == 0:
                return theta_prev, flag_prev
                
            sin_arg = np.clip(p*np.sin(theta_prev - THETA0_START + np.pi)/a, -1.0, 1.0)
            beta = np.arcsin(sin_arg)
            gamma = beta - (np.pi - ARC_ANGLE)/2
            b = np.sqrt(a**2 + ARC_RADIUS**2 - 2*a*ARC_RADIUS*np.cos(gamma))
            sigma = np.arcsin(np.clip(a*np.sin(gamma)/b, -1.0, 1.0))
            phi = np.arccos(np.clip((ARC_RADIUS**2 + b**2 - d0**2)/(2*ARC_RADIUS*b), -1.0, 1.0))
            return ARC_ANGLE - phi + sigma, 3
        else:
            # 继续在盘出螺线
            theta = bisection_solve(distance_equation_outward, theta_prev - np.pi/2, theta_prev, 1e-8, theta_prev, d0)
            return theta, 4

def calculate_link_velocity(v_prev, pos_prev, pos_curr):
    """计算连杆速度"""
    theta_prev, flag_prev = pos_prev
    theta_curr, flag_curr = pos_curr
    
    x_prev, y_prev = get_coordinates_from_theta(theta_prev, flag_prev)
    x_curr, y_curr = get_coordinates_from_theta(theta_curr, flag_curr)
    
    # 避免除零和特殊情况
    if abs(x_prev - x_curr) < 1e-10:
        return v_prev
    
    k_link = (y_prev - y_curr) / (x_prev - x_curr)  # 连杆斜率
    
    # 计算速度方向斜率
    if flag_prev == 1 and flag_curr == 1:
        k_v_prev, k_v_curr = spiral_tangent_slope(theta_prev), spiral_tangent_slope(theta_curr)
    elif flag_prev == 2 and flag_curr == 1:
        k_v_prev = -(x_prev - ARC1_CENTER[0]) / (y_prev - ARC1_CENTER[1])
        k_v_curr = spiral_tangent_slope(theta_curr)
    elif flag_prev == 2 and flag_curr == 2:
        return v_prev  # 圆弧上速度不变
    elif flag_prev == 3 and flag_curr == 2:
        k_v_prev = -(x_prev - ARC2_CENTER[0]) / (y_prev - ARC2_CENTER[1])
        k_v_curr = -(x_curr - ARC1_CENTER[0]) / (y_curr - ARC1_CENTER[1])
    elif flag_prev == 3 and flag_curr == 3:
        return v_prev  # 圆弧上速度不变
    elif flag_prev == 4 and flag_curr == 3:
        k_v_prev = spiral_tangent_slope(theta_prev + np.pi)
        k_v_curr = -(x_curr - ARC2_CENTER[0]) / (y_curr - ARC2_CENTER[1])
    else:  # flag_prev == 4 and flag_curr == 4
        k_v_prev = spiral_tangent_slope(theta_prev + np.pi)
        k_v_curr = spiral_tangent_slope(theta_curr + np.pi)
    
    # 速度传递计算
    denom_prev = 1 + k_v_prev * k_link
    denom_curr = 1 + k_v_curr * k_link
    
    if abs(denom_prev) < 1e-10 or abs(denom_curr) < 1e-10:
        return v_prev
    
    angle_prev = np.arctan(abs((k_v_prev - k_link) / denom_prev))
    angle_curr = np.arctan(abs((k_v_curr - k_link) / denom_curr))
    
    if abs(np.cos(angle_curr)) < 1e-10:
        return np.inf
    
    return v_prev * np.cos(angle_prev) / np.cos(angle_curr)

# ==============================================================================
# SECTION 4: 问题5的优化求解 - 基于theta参数搜索
# ==============================================================================

def find_max_velocity_in_chain(theta_head, v_head=1.0, num_links=3):
    """计算龙头在指定位置时，前num_links节的最大速度"""
    positions = [(theta_head, 4)]  # 龙头在盘出螺线
    velocities = [v_head]
    
    for i in range(num_links):
        # 计算下一节位置
        theta_next, flag_next = calculate_next_link_position(
            positions[-1][0], positions[-1][1], i == 0
        )
        positions.append((theta_next, flag_next))
        
        # 计算下一节速度
        v_next = calculate_link_velocity(velocities[-1], positions[-2], positions[-1])
        velocities.append(abs(v_next) if not np.isinf(v_next) else 1e6)
    
    return max(velocities[1:])  # 返回除龙头外的最大速度

def optimize_dragon_head_speed():
    """优化求解龙头最大速度"""
    print("="*60)
    print("问题5：龙头最大速度优化求解")
    print("="*60)
    
    # 搜索范围：龙头在盘出螺线上的有效区间
    theta_start = THETA0_START - np.pi
    theta_end = 14.1235657  # 第一节龙身前把手到达盘出螺线时龙头的位置
    
    print(f"\n搜索范围: theta ∈ [{theta_start:.3f}, {theta_end:.3f}]")
    print("正在进行粗扫描...")
    
    # 第一步：粗扫描找到最大值区域
    theta_coarse = np.arange(theta_start, theta_end, 0.001)
    max_velocity_global = 0
    optimal_theta = theta_start
    
    for theta in tqdm(theta_coarse, desc="粗扫描"):
        try:
            max_v = find_max_velocity_in_chain(theta, 1.0, 3)
            if max_v > max_velocity_global and max_v < 1e5:
                max_velocity_global = max_v
                optimal_theta = theta
        except:
            continue
    
    print(f"粗扫描结果: 最大速度 = {max_velocity_global:.6f} m/s (theta = {optimal_theta:.6f})")
    
    # 第二步：精细搜索
    print("正在进行精细搜索...")
    theta_fine = np.arange(optimal_theta - 0.001, optimal_theta + 0.001, 1e-6)
    
    for theta in tqdm(theta_fine, desc="精细搜索"):
        try:
            max_v = find_max_velocity_in_chain(theta, 1.0, 3)
            if max_v > max_velocity_global and max_v < 1e5:
                max_velocity_global = max_v
                optimal_theta = theta
        except:
            continue
    
    # 计算最终结果
    v_limit = 2.0  # 速度限制
    v_head_max = v_limit / max_velocity_global
    
    print(f"\n" + "="*60)
    print("最终优化结果:")
    print(f"最优龙头位置: theta = {optimal_theta:.8f} rad")
    print(f"单位龙头速度下的最大速度: {max_velocity_global:.8f} m/s")
    print(f"速度约束: {v_limit} m/s")
    print(f"龙头最大允许速度: {v_head_max:.6f} m/s")
    print("="*60)
    
    # 验证计算
    print("\n验证计算结果...")
    positions_verify = [(optimal_theta, 4)]
    velocities_verify = [v_head_max]
    
    print(f"龙头速度: {v_head_max:.6f} m/s")
    for i in range(5):
        theta_next, flag_next = calculate_next_link_position(
            positions_verify[-1][0], positions_verify[-1][1], i == 0
        )
        positions_verify.append((theta_next, flag_next))
        
        v_next = calculate_link_velocity(velocities_verify[-1], positions_verify[-2], positions_verify[-1])
        velocities_verify.append(abs(v_next))
        
        print(f"第{i+1}节速度: {abs(v_next):.6f} m/s")
    
    actual_max = max(velocities_verify[1:])
    print(f"\n实际最大速度: {actual_max:.6f} m/s (应接近 {v_limit} m/s)")
    
    return v_head_max

if __name__ == "__main__":
    result = optimize_dragon_head_speed()
    print(f"\n最终答案：龙头最大行进速度 = {result:.6f} m/s")