# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve
import warnings

# 抑制 fsolve 的警告
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# --- 1. 系统常量和参数 ---
LINK_WIDTH = 0.30  # 连杆的物理宽度 (m)
V0 = 1.0  # 第一个铰接点的初始速度 (m/s)
NUM_LINKS = 223  # 连杆数量
NUM_POINTS = NUM_LINKS + 1  # 铰接点数量
HINGE_OFFSET = 0.275  # 铰接点距离连杆两端的偏移量
TURNAROUND_DIAMETER = 9.0  # 调头空间直径 (m)
TURNAROUND_RADIUS = TURNAROUND_DIAMETER / 2.0  # 调头空间半径 (m)

# 连杆物理长度
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41
# 铰接点之间的有效长度
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)

# 初始极角
THETA_INITIAL = 16 * 2 * np.pi

# --- 2. 核心运动学函数（从问题二复制） ---

def bisection_solver(func, low, high, tolerance=1e-9, max_iter=100):
    """二分法求解器"""
    f_low, f_high = func(low), func(high)
    if f_low * f_high > 0: return np.nan
    for _ in range(max_iter):
        mid = (low + high) / 2
        if high - low < tolerance: return mid
        f_mid = func(mid)
        if f_low * f_mid < 0: high = mid
        else: low = mid
    return (low + high) / 2

def arc_length_func(theta, a):
    """计算螺旋线弧长"""
    if theta <= 1e-9: return 0
    val = theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))
    return 0.5 * a * val

def get_theta0(t, a, v0, s_initial):
    """计算t时刻第一个铰接点的极角"""
    s_target = s_initial - v0 * t
    if s_target < 0: return -1
    def func_to_solve(theta): 
        return arc_length_func(theta, a) - s_target
    return bisection_solver(func_to_solve, 0, THETA_INITIAL)

def get_next_theta(theta_prev, link_len, a):
    """计算下一个铰接点的极角"""
    if np.isnan(theta_prev): return np.nan
    
    def func_to_solve(theta_next_arr):
        theta_next = theta_next_arr[0]
        dist_sq = (a**2) * (theta_prev**2 + theta_next**2 - 
                           2 * theta_prev * theta_next * np.cos(theta_next - theta_prev))
        return dist_sq - link_len**2
    
    if theta_prev > 1e-4:
        d_theta_guess = link_len / (a * np.sqrt(1 + theta_prev**2))
    else:
        d_theta_guess = link_len / a
    initial_guess = theta_prev + d_theta_guess
    
    sol, _, ier, _ = fsolve(func_to_solve, [initial_guess], full_output=True)
    return sol[0] if ier == 1 else np.nan

def get_cartesian_coords(thetas, a):
    """极坐标转笛卡尔坐标"""
    rhos = a * thetas
    x = rhos * np.cos(thetas)
    y = rhos * np.sin(thetas)
    return x, y

def get_all_positions_at_t(t, a, s_initial):
    """计算t时刻所有铰接点的位置"""
    thetas = np.full(NUM_POINTS, np.nan)
    theta_0 = get_theta0(t, a, V0, s_initial)
    if theta_0 < 0: return None, None
    
    thetas[0] = theta_0
    for i in range(1, NUM_POINTS):
        thetas[i] = get_next_theta(thetas[i-1], effective_link_lengths[i-1], a)
        if np.isnan(thetas[i]): return None, None
    
    x, y = get_cartesian_coords(thetas, a)
    return np.column_stack((x, y)), thetas

# --- 3. 碰撞检测函数（使用问题二的思路） ---

def point_to_segment_distance(p, a, b):
    """计算点 p 到线段 ab 的最短距离"""
    ab, ap = b - a, p - a
    len_sq_ab = np.dot(ab, ab)
    if len_sq_ab < 1e-14: return np.linalg.norm(ap)
    t = np.dot(ap, ab) / len_sq_ab
    if t < 0.0: closest = a
    elif t > 1.0: closest = b
    else: closest = a + t * ab
    return np.linalg.norm(p - closest)

def check_collision(positions):
    """
    使用问题二的碰撞检测思路
    返回 (是否碰撞, 最小间隙)
    """
    if positions is None: return False, float('inf')
    
    p0, p1 = positions[0], positions[1]
    
    # 计算第一个连杆的方向向量
    v_axis_p1_to_p0 = p0 - p1
    u_axis = v_axis_p1_to_p0 / np.linalg.norm(v_axis_p1_to_p0)
    v_perp = np.array([-u_axis[1], u_axis[0]])
    
    # 确保垂直向量指向外侧
    if np.linalg.norm(p0 + v_perp) < np.linalg.norm(p0): 
        v_perp = -v_perp
    
    # 第一个连杆的两个关键点
    front_end_point = p0 + u_axis * HINGE_OFFSET
    critical_point_front = front_end_point + v_perp * (LINK_WIDTH / 2.0)
    
    rear_end_point = p1 - u_axis * HINGE_OFFSET
    critical_point_rear = rear_end_point + v_perp * (LINK_WIDTH / 2.0)
    
    min_clearance = float('inf')
    
    # 检查与其他连杆的碰撞（从第三个连杆开始）
    for i in range(2, NUM_LINKS):
        if i >= len(positions) - 1: break
        
        target_p_start, target_p_end = positions[i], positions[i+1]
        
        # 检查前端点
        dist_front = point_to_segment_distance(critical_point_front, target_p_start, target_p_end)
        clearance_front = float(dist_front - (LINK_WIDTH / 2.0))
        min_clearance = min(min_clearance, clearance_front)
        
        # 检查后端点
        dist_rear = point_to_segment_distance(critical_point_rear, target_p_start, target_p_end)
        clearance_rear = float(dist_rear - (LINK_WIDTH / 2.0))
        min_clearance = min(min_clearance, clearance_rear)
    
    return min_clearance <= 0, min_clearance

# --- 4. 检查特定螺距下是否发生碰撞 ---

def check_collision_for_pitch(pitch, max_time=1000):
    """
    对于给定螺距，模拟连杆运动直到第一个铰接点到达调头空间边界
    返回是否发生碰撞
    """
    a = pitch / (2 * np.pi)
    s_initial = arc_length_func(THETA_INITIAL, a)
    
    # 时间步长
    dt = 0.1
    
    for t in np.arange(0, max_time, dt):
        positions, thetas = get_all_positions_at_t(t, a, s_initial)
        
        if positions is None:
            continue
        
        # 检查第一个铰接点是否到达调头空间边界
        first_point_radius = np.linalg.norm(positions[0])
        if first_point_radius <= TURNAROUND_RADIUS:
            # 到达调头空间，未发生碰撞
            return False
        
        # 检查碰撞
        has_collision, _ = check_collision(positions)
        if has_collision:
            return True
    
    # 超时未到达调头空间
    return False

# --- 5. 优化搜索最小螺距 ---

def optimize_spiral_pitch():
    """
    使用二分法搜索最小螺距
    """
    print("=" * 60)
    print("问题三：最小螺距优化（使用问题二碰撞检测思路）")
    print(f"初始参数: 连杆前端速度={V0} m/s, 调头空间直径={TURNAROUND_DIAMETER} m")
    print("=" * 60)
    
    # 初始搜索范围
    pitch_min = 0.3  # 最小可能螺距
    pitch_max = 0.6  # 最大可能螺距
    tolerance = 1e-6  # 搜索精度
    
    print("\n开始二分法搜索...")
    
    # 首先验证边界
    print(f"测试下界 {pitch_min:.6f} m...")
    if not check_collision_for_pitch(pitch_min):
        print(f"螺距 {pitch_min:.6f} m 时无碰撞，搜索更小值...")
        pitch_min = 0.1
        
    print(f"测试上界 {pitch_max:.6f} m...")
    if check_collision_for_pitch(pitch_max):
        print(f"螺距 {pitch_max:.6f} m 时有碰撞，搜索更大值...")
        pitch_max = 1.0
    
    # 二分法搜索
    iteration = 0
    while pitch_max - pitch_min > tolerance:
        iteration += 1
        pitch_mid = (pitch_min + pitch_max) / 2
        
        print(f"\n第 {iteration} 次迭代: 测试螺距 {pitch_mid:.6f} m")
        
        if check_collision_for_pitch(pitch_mid):
            # 发生碰撞，需要更大的螺距
            pitch_min = pitch_mid
            print(f"  发生碰撞，搜索更大螺距")
        else:
            # 未发生碰撞，可以尝试更小的螺距
            pitch_max = pitch_mid
            print(f"  未发生碰撞，搜索更小螺距")
    
    # 最终结果取略大于最小值的安全值
    optimal_pitch = pitch_max + tolerance
    
    return optimal_pitch

# --- 主程序执行 ---

if __name__ == "__main__":
    try:
        # 开始优化计算
        result = optimize_spiral_pitch()
        
        print("\n" + "=" * 60)
        print("                    优化结果")
        print("=" * 60)
        print(f"在调头空间外不会发生碰撞的最小螺距: {result:.6f} m")
        
        # 验证结果
        print(f"\n结果验证:")
        a = result / (2 * np.pi)
        print(f"  螺旋线参数 a: {a:.6f} m/rad")
        print(f"  螺线紧密度: {2*np.pi/result:.4f} rad/m")
        
        # 计算到达调头空间需要的时间
        s_initial = arc_length_func(THETA_INITIAL, a)
        # 粗略估算
        theta_turnaround = TURNAROUND_RADIUS / a
        s_turnaround = arc_length_func(theta_turnaround, a)
        time_to_turnaround = (s_initial - s_turnaround) / V0
        print(f"  预计到达调头空间时间: {time_to_turnaround:.2f} s")
        
        print("\n优化完成!")
        
    except Exception as e:
        print(f"计算过程中发生错误: {e}")
        raise