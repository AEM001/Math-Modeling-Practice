import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fsolve

# --- 1. 系统常量和参数 ---
D_PITCH = 0.55  # 螺旋桨螺距 (m)
A_SPIRAL = D_PITCH / (2 * np.pi)  # 螺旋线方程常数 a (ρ = a * θ)
V0 = 1.0  # 第一个铰接点的初始速度 (m/s)
NUM_LINKS = 223  # 连杆数量
NUM_POINTS = NUM_LINKS + 1  # 铰接点数量

# **关键修正：计算铰接点之间的有效距离**
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41
HINGE_OFFSET = 0.275
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)

# 初始条件：t=0 时，第一个铰接点位于第16圈
THETA_INITIAL = 16 * 2 * np.pi

# 时间设置
T_END = 300  # 模拟结束时间 (s)
T_STEP = 1   # 时间步长 (s)
time_points = np.arange(0, T_END + T_STEP, T_STEP)

# 报告中关键铰接点和时间
report_indices_desc = {
    "第1个连杆前端铰接点": 0, "第1个连杆后端铰接点": 1,
    "第51个连杆前端铰接点": 51, "第101个连杆前端铰接点": 101,
    "第151个连杆前端铰接点": 151, "第201个连杆前端铰接点": 201,
    "第223个连杆后端铰接点": 223,
}
report_times = [0, 60, 120, 180, 240, 300]
report_indices = list(report_indices_desc.values())

# --- 2. 核心数学和求解函数 ---

def bisection_solver(func, low, high, tolerance=1e-9, max_iter=100):
    """一个鲁棒的二分法，仅用于单调的arc_length函数。"""
    f_low, f_high = func(low), func(high)
    if f_low * f_high > 0: return np.nan
    for _ in range(max_iter):
        mid = (low + high) / 2
        if high - low < tolerance: return mid
        f_mid = func(mid)
        if f_mid == 0: return mid
        if f_low * f_mid < 0: high = mid
        else: low = mid
    return (low + high) / 2

def arc_length_func(theta, a):
    """计算从原点到角度theta的螺旋线弧长。"""
    if theta <= 1e-9: return 0
    val = theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))
    return 0.5 * a * val

S_INITIAL = arc_length_func(THETA_INITIAL, A_SPIRAL)

def get_theta0(t, a, v0):
    """使用鲁棒的二分法计算前导点在时间t时的theta_0。"""
    s_target = S_INITIAL - v0 * t
    if s_target < 0: return -1
    def func_to_solve(theta): return arc_length_func(theta, a) - s_target
    return bisection_solver(func_to_solve, 0, THETA_INITIAL)

def get_next_theta(theta_prev, link_len, a):
    """
    根据圆与螺旋线相交的几何约束，使用fsolve计算下一个铰接点的theta。
    """
    if np.isnan(theta_prev): return np.nan

    def func_to_solve(theta_next_array):
        theta_next = theta_next_array[0]
        # 方程: (点之间的距离)^2 - (连杆长度)^2 = 0
        dist_sq = (a**2) * (theta_prev**2 + theta_next**2 - 2 * theta_prev * theta_next * np.cos(theta_next - theta_prev))
        return dist_sq - link_len**2

    # 提供基于弧长近似的良好初始猜测以确保稳定性
    if theta_prev > 1e-4:
        d_theta_guess = link_len / (a * np.sqrt(1 + theta_prev**2))
    else:
        d_theta_guess = link_len / a
    
    initial_guess = theta_prev + d_theta_guess

    # 使用fsolve寻找根，并检查成功状态
    theta_next_solution, _, ier, _ = fsolve(func_to_solve, [initial_guess], full_output=True)
    
    return theta_next_solution[0] if ier == 1 else np.nan

def get_cartesian_coords(thetas, a):
    """将极角数组转换为笛卡尔坐标 (x, y)。"""
    rhos = a * thetas
    x = rhos * np.cos(thetas)
    y = rhos * np.sin(thetas)
    return x, y

def get_velocities(thetas, v_start, a):
    """使用速度投影法计算所有铰接点的速度。"""
    velocities = np.full(len(thetas), np.nan)
    if np.isnan(thetas[0]): return velocities
    velocities[0] = v_start
    x_coords, y_coords = get_cartesian_coords(thetas, a)

    for i in range(len(thetas) - 1):
        if np.isnan(velocities[i]) or np.isnan(thetas[i+1]): continue
            
        norm_i = np.sqrt(1 + thetas[i]**2)
        tx_i = (np.cos(thetas[i]) - thetas[i] * np.sin(thetas[i])) / norm_i
        ty_i = (np.sin(thetas[i]) + thetas[i] * np.cos(thetas[i])) / norm_i
        
        norm_i_plus_1 = np.sqrt(1 + thetas[i+1]**2)
        tx_i_plus_1 = (np.cos(thetas[i+1]) - thetas[i+1] * np.sin(thetas[i+1])) / norm_i_plus_1
        ty_i_plus_1 = (np.sin(thetas[i+1]) + thetas[i+1] * np.cos(thetas[i+1])) / norm_i_plus_1
        
        link_vec_x = x_coords[i+1] - x_coords[i]
        link_vec_y = y_coords[i+1] - y_coords[i]
        
        dot_product_i = tx_i * link_vec_x + ty_i * link_vec_y
        dot_product_i_plus_1 = tx_i_plus_1 * link_vec_x + ty_i_plus_1 * link_vec_y

        if np.abs(dot_product_i_plus_1) < 1e-9: velocities[i+1] = np.inf
        else: velocities[i+1] = velocities[i] * (dot_product_i / dot_product_i_plus_1)
             
    return velocities

# --- 3. 主计算循环 ---
full_results = []

for t in tqdm(time_points, desc="正在模拟运动学过程"):
    current_thetas = np.full(NUM_POINTS, np.nan)
    theta_0 = get_theta0(t, A_SPIRAL, V0)
    
    if theta_0 < 0:
        print(f"\n在 t={t}s 时, 系统已完全收缩，模拟终止。")
        break
    current_thetas[0] = theta_0
    
    for i in range(1, NUM_POINTS):
        current_thetas[i] = get_next_theta(current_thetas[i-1], effective_link_lengths[i-1], A_SPIRAL)
        
    x, y = get_cartesian_coords(current_thetas, A_SPIRAL)
    v = get_velocities(current_thetas, V0, A_SPIRAL)
    
    for i in range(NUM_POINTS):
        full_results.append({
            "t (s)": t, "铰接点索引": i, "x (m)": x[i],
            "y (m)": y[i], "速度 (m/s)": v[i]
        })

# --- 4. 结果处理和输出 ---

print("\n计算完成。正在将完整结果保存到 result1.xlsx ...")
full_df = pd.DataFrame(full_results)
full_df.fillna(0.0, inplace=True) # 为显示目的，将NaN替换为0.0
for col in ["x (m)", "y (m)", "速度 (m/s)"]:
    full_df[col] = full_df[col].astype(float).map('{:.6f}'.format)
full_df.to_excel("result1.xlsx", index=False, engine='openpyxl')
print("完整结果已成功保存。")

print("\n正在生成摘要报告...")

# 创建位置报告表格
pos_report_data = {}
for desc, idx in report_indices_desc.items():
    pos_report_data[f"{desc} x(m)"] = {}
    pos_report_data[f"{desc} y(m)"] = {}

# 创建速度报告表格
vel_report_data = {}
for desc, idx in report_indices_desc.items():
    vel_report_data[desc] = {}

# 填充报告数据
for _, row in full_df.iterrows():
    t = int(row["t (s)"])
    idx = int(row["铰接点索引"])
    
    if t in report_times and idx in report_indices:
        # 找到对应的描述
        for desc, target_idx in report_indices_desc.items():
            if idx == target_idx:
                x_val = float(row["x (m)"])
                y_val = float(row["y (m)"])
                v_val = float(row["速度 (m/s)"])
                
                pos_report_data[f"{desc} x(m)"][t] = x_val
                pos_report_data[f"{desc} y(m)"][t] = y_val
                vel_report_data[desc][t] = v_val
                break

# 转换为DataFrame
pos_report_formatted = pd.DataFrame(pos_report_data).reindex(columns=sorted(pos_report_data.keys()))
pos_report_formatted = pos_report_formatted.reindex(index=report_times)

vel_report = pd.DataFrame(vel_report_data).T
vel_report = vel_report.reindex(columns=report_times)

print("\n" + "="*25 + " 位置报告 (单位: m) " + "="*25)
print(pos_report_formatted.to_string(float_format="%.6f"))

print("\n" + "="*25 + " 速度报告 (单位: m/s) " + "="*25)
print(vel_report.to_string(float_format="%.6f"))

try:
    with pd.ExcelWriter('report_summary.xlsx', engine='openpyxl') as writer:
        pos_report_formatted.to_excel(writer, sheet_name='位置报告')
        vel_report.to_excel(writer, sheet_name='速度报告')
    print("\n摘要报告已成功保存到 report_summary.xlsx。")
except Exception as e:
    print(f"\n保存摘要报告时出错: {e}")
