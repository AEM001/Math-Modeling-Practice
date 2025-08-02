# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve
import warnings

# --- 抑制 fsolve 产生的警告，使输出更整洁 ---
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# --- 1. 系统常量和参数 ---
LINK_WIDTH = 0.30 # 连杆的物理宽度 (m)
V0 = 1.0  # 第一个铰接点的初始速度 (m/s)
NUM_LINKS = 223  # 连杆数量
NUM_POINTS = NUM_LINKS + 1  # 铰接点数量
HINGE_OFFSET = 0.275 # 铰接点距离连杆两端的偏移量
TURNAROUND_RADIUS = 4.5 # 调头空间半径 (m)

# 连杆物理长度
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41
# 铰接点之间的有效长度
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)

# 初始极角
THETA_INITIAL = 16 * 2 * np.pi

# --- 2. 核心数学和求解函数  ---

def get_kinematics_functions(d):
    """根据螺距d，生成一套对应的计算函数，避免重复计算常数。"""
    A_SPIRAL = d / (2 * np.pi)
    
    memo_arc_length = {}
    def memoized_arc_length_func(theta):
        if theta in memo_arc_length: return memo_arc_length[theta]
        if theta <= 1e-9: return 0
        val = theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))
        result = 0.5 * A_SPIRAL * val
        memo_arc_length[theta] = result
        return result

    S_INITIAL = memoized_arc_length_func(THETA_INITIAL)

    def get_theta0(t):
        s_target = S_INITIAL - V0 * t
        if s_target < 0: return -1
        def func_to_solve(theta): return memoized_arc_length_func(theta) - s_target
        low, high = 0, THETA_INITIAL
        for _ in range(100): # Standard bisection solver
            mid = (low + high) / 2
            if high - low < 1e-12: return mid
            f_mid = func_to_solve(mid)
            if np.isnan(f_mid) or f_mid < 0: high = mid
            else: low = mid
        return (low + high) / 2

    def get_next_theta(theta_prev, link_len):
        if np.isnan(theta_prev): return np.nan
        def func_to_solve(theta_next_arr):
            theta_next = theta_next_arr[0]
            dist_sq = (A_SPIRAL**2) * (theta_prev**2 + theta_next**2 - 2*theta_prev*theta_next*np.cos(theta_next - theta_prev))
            return dist_sq - link_len**2
        d_theta_guess = link_len / (A_SPIRAL*np.sqrt(1+theta_prev**2)) if theta_prev > 1e-4 else link_len/A_SPIRAL
        sol, _, ier, _ = fsolve(func_to_solve, [theta_prev + d_theta_guess], full_output=True)
        return sol[0] if ier == 1 else np.nan

    def get_all_positions_and_thetas_at_t(t):
        thetas = np.full(NUM_POINTS, np.nan)
        theta_0 = get_theta0(t)
        if theta_0 < 0: return None, None
        thetas[0] = theta_0
        for i in range(1, NUM_POINTS):
            thetas[i] = get_next_theta(thetas[i-1], effective_link_lengths[i-1])
            if np.isnan(thetas[i]): return None, None
        rhos = A_SPIRAL * thetas
        positions = np.column_stack((rhos * np.cos(thetas), rhos * np.sin(thetas)))
        return positions, thetas

    return get_all_positions_and_thetas_at_t, memoized_arc_length_func

# --- 3. 精确碰撞检测逻辑---

def _get_clearance_for_point(point, target_p_start, target_p_end):
    """辅助函数：计算一个攻击点到目标连杆侧面的间隙（仅当投影有效时）。"""
    v_axis_target = target_p_end - target_p_start
    v_point_to_start = point - target_p_start
    len_sq_axis = np.dot(v_axis_target, v_axis_target)
    if len_sq_axis < 1e-14: return float('inf')

    t_proj = np.dot(v_point_to_start, v_axis_target) / len_sq_axis
    if 0 <= t_proj <= 1:
        lateral_distance = np.linalg.norm(v_point_to_start - t_proj * v_axis_target)
        return lateral_distance - (LINK_WIDTH / 2.0)
    return float('inf')

def check_collision_and_get_clearance(get_positions_thetas_func, t):
    """碰撞检测核心函数，采用精确的"双攻击点"模型。"""
    positions, _ = get_positions_thetas_func(t)
    if positions is None: return -1.0
    
    p0, p1 = positions[0], positions[1]
    v_axis_1 = p1 - p0
    norm_axis_1 = np.linalg.norm(v_axis_1)
    if norm_axis_1 < 1e-9: return float('inf')
    u_axis_1 = v_axis_1 / norm_axis_1
    
    u_perp_1 = np.array([-u_axis_1[1], u_axis_1[0]])
    if np.linalg.norm(p0 + u_perp_1) < np.linalg.norm(p0):
        u_perp_1 = -u_perp_1
    
    crit_p_front = p0 - u_axis_1 * HINGE_OFFSET + u_perp_1 * (LINK_WIDTH / 2.0)
    crit_p_rear = p1 + u_axis_1 * HINGE_OFFSET + u_perp_1 * (LINK_WIDTH / 2.0)
    
    min_clearance = float('inf')
    for i in range(2, NUM_LINKS):
        target_p_start, target_p_end = positions[i], positions[i+1]
        clearance_front = _get_clearance_for_point(crit_p_front, target_p_start, target_p_end)
        clearance_rear = _get_clearance_for_point(crit_p_rear, target_p_start, target_p_end)
        min_clearance = min(min_clearance, float(clearance_front), float(clearance_rear))
    return min_clearance

# --- 4. 核心求解逻辑的辅助函数 ---

def find_first_collision_time(get_positions_thetas_func, time_search_range=[0, 800]):
    t_low, t_high = time_search_range
    SAFETY_MARGIN = 1e-7
    if check_collision_and_get_clearance(get_positions_thetas_func, t_high) > SAFETY_MARGIN:
        return float('inf')
    
    for _ in range(100): # Bisection search for critical collision time
        if t_high - t_low < 1e-9: break
        t_mid = (t_low + t_high) / 2
        if check_collision_and_get_clearance(get_positions_thetas_func, t_mid) > SAFETY_MARGIN:
            t_low = t_mid
        else:
            t_high = t_mid
    return t_high

def calculate_time_to_boundary(d, arc_length_calculator):
    if d <= 0: return float('inf')
    a = d / (2 * np.pi)
    if TURNAROUND_RADIUS / a >= THETA_INITIAL: return float('inf')
    theta_boundary = TURNAROUND_RADIUS / a
    s_initial = arc_length_calculator(THETA_INITIAL)
    s_boundary = arc_length_calculator(theta_boundary)
    arc_len_traveled = s_initial - s_boundary
    return arc_len_traveled / V0

# --- 5. 主执行模块 ---
print("开始优化搜索最小有效螺距 d_min (变步长网格搜索)...")

d_start, d_end = 0.55, 0.40
step_coarse = 0.01
step_fine = 0.0001
d_safe = d_start
d_unsafe = d_end
found_bracket = False

# --- 阶段一: 粗略搜索 ---
print("\n阶段一: 粗略搜索，快速定位临界区间...")
search_points_coarse = np.arange(d_start, d_end - step_coarse, -step_coarse)
with tqdm(total=len(search_points_coarse), desc="粗略搜索") as pbar:
    for d_current in search_points_coarse:
        get_pos_thetas_func, arc_calc_func = get_kinematics_functions(d_current)
        t_boundary = calculate_time_to_boundary(d_current, arc_calc_func)
        if t_boundary == float('inf'):
            pbar.update(1)
            continue

        t_collision = find_first_collision_time(get_pos_thetas_func, time_search_range=[0, t_boundary + 50])

        is_safe = t_collision >= t_boundary
        if not is_safe:
            d_unsafe = d_current
            found_bracket = True
            pbar.update(pbar.total - pbar.n)
            break
        
        d_safe = d_current
        pbar.update(1)

if not found_bracket:
    print(f"错误：在 [{d_end}, {d_start}] 范围内所有螺距均有效。请扩大搜索范围或检查模型。")
    d_min = d_end
else:
    print(f"阶段一完成。已将临界区间缩小至 [{d_unsafe:.4f}, {d_safe:.4f}]。")

    # --- 阶段二: 高精度精细搜索---
    print("\n阶段二: 在临界区间内进行精细搜索以应对非单调性...")
    d_min = d_safe # 初始值设为已知的安全上界
    
    search_points_fine = np.arange(d_safe, d_unsafe - step_fine, -step_fine)
    with tqdm(total=len(search_points_fine), desc="精确搜索") as pbar:
        # 由于函数非单调，我们需要完整遍历整个临界区间，
        # 寻找最小的有效d值，而不是碰到第一个无效d就停止。
        for d_current in search_points_fine:
            get_pos_thetas_func, arc_calc_func = get_kinematics_functions(d_current)
            t_boundary = calculate_time_to_boundary(d_current, arc_calc_func)
            
            if t_boundary == float('inf'):
                pbar.update(1)
                continue
                
            t_collision = find_first_collision_time(get_pos_thetas_func, time_search_range=[0, t_boundary + 50])

            is_safe = t_collision >= t_boundary
            if is_safe:
                # 因为是从大到小搜索，我们不断更新d_min。
                # 循环结束后，d_min将是找到的最后一个（即最小的）安全值。
                d_min = d_current
            
            pbar.update(1)
    
    print("阶段二完成。")

print("\n" + "="*70)
print(" " * 25 + "问题3：最小螺距优化总结")
print("="*70)
print(f"计算完成。")
print(f"为保证系统能安全到达直径为 {2*TURNAROUND_RADIUS}m 的掉头区，")
print(f"所需的最小螺距 d_min 为: {d_min:.6f} m")
print("="*70)


