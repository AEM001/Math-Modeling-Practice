import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from scipy.optimize import fsolve

# --- 抑制 fsolve 的警告，使输出更整洁 ---
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# --- 1. 系统常数和参数 ---
D_PITCH = 0.55  # 螺旋桨螺距 (m)
LINK_WIDTH = 0.30 # 连杆的物理宽度，用于碰撞检测 (m)
A_SPIRAL = D_PITCH / (2 * np.pi)  # 螺旋方程常数 a (ρ = a * θ)
V0 = 1.0  # 第一个铰接点的初始速度 (m/s)
NUM_LINKS = 223  # 连杆数量
NUM_POINTS = NUM_LINKS + 1  # 铰接点数量
HINGE_OFFSET = 0.275 # 铰接点距连杆末端的偏移量

# 计算用于运动学的铰接点之间的有效距离
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)

# 初始条件：t=0 时，第一个铰接点在第16圈
THETA_INITIAL = 16 * 2 * np.pi

# 报告中关键铰接点及其时间
report_indices_desc = {
    "第1个连杆前端铰接点": 0, "第1个连杆后端铰接点": 1,
    "第51个连杆前端铰接点": 51, "第101个连杆前端铰接点": 101,
    "第151个连杆前端铰接点": 151, "第201个连杆前端铰接点": 201,
    "第223个连杆后端铰接点": 223,
}
report_indices = list(report_indices_desc.values())

# --- 2. 核心运动学和求解函数  ---

def bisection_solver(func, low, high, tolerance=1e-9, max_iter=100):
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
    if theta <= 1e-9: return 0
    val = theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))
    return 0.5 * a * val
# 初始情况下的弧长
S_INITIAL = arc_length_func(THETA_INITIAL, A_SPIRAL)
# 得到第一个铰接点的角度
def get_theta0(t, a, v0):
    s_target = S_INITIAL - v0 * t
    if s_target < 0: return -1
    def func_to_solve(theta): return arc_length_func(theta, a) - s_target
    return bisection_solver(func_to_solve, 0, THETA_INITIAL)
# 得到下一个铰接点的角度
def get_next_theta(theta_prev, link_len, a):
    if np.isnan(theta_prev): return np.nan
    def func_to_solve(theta_next_arr):
        theta_next = theta_next_arr[0]
        dist_sq = (a**2) * (theta_prev**2 + theta_next**2 - 2 * theta_prev * theta_next * np.cos(theta_next - theta_prev))
        return dist_sq - link_len**2
    if theta_prev > 1e-4:
        d_theta_guess = link_len / (a * np.sqrt(1 + theta_prev**2))
    else:
        d_theta_guess = link_len / a
    initial_guess = theta_prev + d_theta_guess
    sol, _, ier, _ = fsolve(func_to_solve, [initial_guess], full_output=True)
    return sol[0] if ier == 1 else np.nan

def get_all_kinematics_at_t(t):
    """计算位置和角度。返回 (位置, 角度)。"""
    thetas = np.full(NUM_POINTS, np.nan)
    theta_0 = get_theta0(t, A_SPIRAL, V0)
    if theta_0 < 0: return None, None
    thetas[0] = theta_0
    for i in range(1, NUM_POINTS):
        thetas[i] = get_next_theta(thetas[i-1], effective_link_lengths[i-1], A_SPIRAL)
        if np.isnan(thetas[i]): return None, None
    x, y = get_cartesian_coords(thetas, A_SPIRAL)
    return np.column_stack((x, y)), thetas

def get_cartesian_coords(thetas, a):
    rhos = a * thetas
    x = rhos * np.cos(thetas)
    y = rhos * np.sin(thetas)
    return x, y

def get_velocities(thetas, v_start, a):
    speeds = np.full(len(thetas), np.nan)
    if np.isnan(thetas[0]): return speeds, np.full((len(thetas), 2), np.nan)
    speeds[0] = v_start
    x_coords, y_coords = get_cartesian_coords(thetas, a)
    for i in range(len(thetas) - 1):
        if np.isnan(speeds[i]) or np.isnan(thetas[i+1]): continue
        # 计算当前铰接点的切向量
        norm_i = np.sqrt(1 + thetas[i]**2)
        tx_i = (np.cos(thetas[i]) - thetas[i] * np.sin(thetas[i])) / norm_i
        ty_i = (np.sin(thetas[i]) + thetas[i] * np.cos(thetas[i])) / norm_i
        # 计算下一个铰接点的切向量
        norm_i1 = np.sqrt(1 + thetas[i+1]**2)
        tx_i1 = (np.cos(thetas[i+1]) - thetas[i+1] * np.sin(thetas[i+1])) / norm_i1
        ty_i1 = (np.sin(thetas[i+1]) + thetas[i+1] * np.cos(thetas[i+1])) / norm_i1
        link_vec = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
        dot_i = tx_i * link_vec[0] + ty_i * link_vec[1]
        dot_i1 = tx_i1 * link_vec[0] + ty_i1 * link_vec[1]
        if np.abs(dot_i1) < 1e-9: speeds[i+1] = np.inf
        else: speeds[i+1] = speeds[i] * (dot_i / dot_i1)
    
    tangent_norm = np.sqrt(1 + thetas**2)
    vx = speeds * (np.cos(thetas) - thetas * np.sin(thetas)) / tangent_norm
    vy = speeds * (np.sin(thetas) + thetas * np.cos(thetas)) / tangent_norm
    return speeds, np.column_stack((vx, vy))

# --- 3. 问题2：碰撞检测函数 (已更正) ---

def point_to_segment_distance(p, a, b):
    """计算点 p 到线段 ab 的最短距离。"""
    ab, ap = b - a, p - a
    len_sq_ab = np.dot(ab, ab)
    if len_sq_ab < 1e-14: return np.linalg.norm(ap) # a 和 b 是同一个点
    t = np.dot(ap, ab) / len_sq_ab
    if t < 0.0: closest = a
    elif t > 1.0: closest = b
    else: closest = a + t * ab
    return np.linalg.norm(p - closest)

def check_collision_and_get_clearance(t):
    """
    核心碰撞检测函数。它计算第一个连杆的关键角点与所有其他连杆之间的最小间隙。
    非正间隙 (<= 0) 表示发生碰撞。
    """
    positions, _ = get_all_kinematics_at_t(t)
    if positions is None: return -1.0, "运动学计算失败"

    p0, p1 = positions[0], positions[1]
    
    # --- 【关键修正】：为第一个连杆定义两个“攻击点” ---
    v_axis_p1_to_p0 = p0 - p1
    u_axis = v_axis_p1_to_p0 / np.linalg.norm(v_axis_p1_to_p0)
    
    v_perp = np.array([-u_axis[1], u_axis[0]])
    if np.linalg.norm(p0 + v_perp) < np.linalg.norm(p0): v_perp = -v_perp

    # 1. 前端关键点 
    front_end_point = p0 + u_axis * HINGE_OFFSET
    critical_point_front = front_end_point + v_perp * (LINK_WIDTH / 2.0)
    
    # 2. 后端关键点 (新添加的那个)
    rear_end_point = p1 - u_axis * HINGE_OFFSET
    critical_point_rear = rear_end_point + v_perp * (LINK_WIDTH / 2.0)


    min_clearance = float('inf')
    collision_info = ""

    # 从第三个连杆 (索引 2) 开始检查碰撞
    for i in range(2, NUM_LINKS):
        target_p_start, target_p_end = positions[i], positions[i+1]
        
        # 检查前端点
        dist_front = point_to_segment_distance(critical_point_front, target_p_start, target_p_end)
        clearance_front = dist_front - (LINK_WIDTH / 2.0)
        if clearance_front < min_clearance:
            min_clearance = clearance_front
            collision_info = f"第1个连杆的前端外角与第{i+1}个连杆发生碰撞"
            
        # 检查后端点
        dist_rear = point_to_segment_distance(critical_point_rear, target_p_start, target_p_end)
        clearance_rear = dist_rear - (LINK_WIDTH / 2.0)
        if clearance_rear < min_clearance:
            min_clearance = clearance_rear
            collision_info = f"第1个连杆的后端外角与第{i+1}个连杆发生碰撞"

    return min_clearance, collision_info

# --- 4. t_max 的主二分法搜索 ---

print("正在使用二分法搜索临界碰撞时间 t_max ...")
t_low, t_high = 0.0, 450.0 # 使用一个安全的上限
precision = 1e-7

# 检查在搜索时间内是否发生碰撞
clearance_high, _ = check_collision_and_get_clearance(t_high)
if clearance_high > 0:
    print(f"警告：在 t={t_high}s 时仍未检测到碰撞。")
    t_max = t_high
else:
    iterations = int(np.ceil(np.log2((t_high - t_low) / precision)))
    with tqdm(total=iterations, desc="搜索进度") as pbar:
        for _ in range(iterations):
            t_mid = (t_low + t_high) / 2
            clearance, _ = check_collision_and_get_clearance(t_mid)
            if clearance > 0: t_low = t_mid
            else: t_high = t_mid
            pbar.update(1)
    t_max = t_high # t_max 是碰撞发生的第一个瞬间

print(f"\n搜索完成。临界碰撞时间 t_max = {t_max:.6f} s")

# --- 5. 在 t_max 时的最终分析和输出 ---

print(f"正在计算 t = {t_max:.6f} s 时的最终状态...")
final_positions, final_thetas = get_all_kinematics_at_t(t_max)
final_clearance, final_collision_info = check_collision_and_get_clearance(t_max)

if final_positions is None or final_thetas is None:
    print("错误：在 t_max 时无法计算系统状态。")
else:
    final_speeds, final_velocity_vectors = get_velocities(final_thetas, V0, A_SPIRAL)

    results_data = []
    for i in range(NUM_POINTS):
        results_data.append({
            "铰接点索引": i,
            "x (m)": final_positions[i, 0], "y (m)": final_positions[i, 1], "z (m)": 0.0,
            "速度大小 (m/s)": final_speeds[i],
            "vx (m/s)": final_velocity_vectors[i, 0], "vy (m/s)": final_velocity_vectors[i, 1], "vz (m/s)": 0.0,
        })

    results_df = pd.DataFrame(results_data)
    for col in results_df.columns[1:]:
        results_df[col] = results_df[col].apply(lambda x: f'{float(x):.6f}' if pd.notnull(x) else 'NaN')

    print("\n正在将完整计算结果保存到 result2.xlsx ...")
    with pd.ExcelWriter('result2.xlsx', engine='openpyxl') as writer:
        info_df = pd.DataFrame({"t_max (s)": [f"{t_max:.6f}"]})
        info_df.to_excel(writer, sheet_name='碰撞时刻', index=False)
        results_df.to_excel(writer, sheet_name=f't_max={t_max:.6f}s_State', index=False)
    print("完整结果已成功保存。")

    print("\n" + "="*70)
    print(" " * 20 + "问题2：碰撞约束分析总结报告")
    print("="*70)
    print(f"1. 终止时刻确定:")
    print(f"   - 系统不发生碰撞的最大运动时间为 t_max = {t_max:.6f} 秒。")
    print(f"\n2. 终止状态分析:")
    print(f"   - 在 t = t_max 时刻，系统发生碰撞。")
    print(f"   - 碰撞原因: **{final_collision_info}**。")
    
    print("\n3. 重点铰接点在 t_max 时的状态:")
    report_df = results_df[results_df["铰接点索引"].astype(int).isin(report_indices)].copy()
    desc_mapping = {v: k for k, v in report_indices_desc.items()}
    report_df["描述"] = [desc_mapping[idx] for idx in report_df["铰接点索引"].astype(int)]
    report_df = report_df.set_index("描述")
    print(report_df[['x (m)', 'y (m)', 'vx (m/s)', 'vy (m/s)', '速度大小 (m/s)']].to_string())
    print("="*70)