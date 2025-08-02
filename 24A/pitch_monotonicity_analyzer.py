# -*- coding: utf-8 -*-
"""
螺距 d 与最小间隙关系的单调性分析
用于判断问题三中使用二分搜索的恰当性
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm
import warnings

# --- Matplotlib 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# --- 1. 系统常量和参数 (与问题三保持一致) ---
LINK_WIDTH = 0.30
V0 = 1.0
NUM_LINKS = 223
NUM_POINTS = NUM_LINKS + 1
HINGE_OFFSET = 0.275
TURNAROUND_DIAMETER = 9.0
TURNAROUND_RADIUS = TURNAROUND_DIAMETER / 2.0
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)
THETA_INITIAL = 16 * 2 * np.pi

# --- 2. 核心运动学和碰撞检测函数 (与之前脚本相同) ---

def arc_length_func(theta, a):
    if theta <= 1e-9: return 0
    val = theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))
    return 0.5 * a * val

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

def get_theta0(t, a, v0, s_initial):
    s_target = s_initial - v0 * t
    if s_target < 0: return -1
    return bisection_solver(lambda theta: arc_length_func(theta, a) - s_target, 0, THETA_INITIAL)

def get_next_theta(theta_prev, link_len, a):
    if np.isnan(theta_prev): return np.nan
    def func_to_solve(theta_next_arr):
        theta_next = theta_next_arr[0]
        dist_sq = (a**2) * (theta_prev**2 + theta_next**2 - 2 * theta_prev * theta_next * np.cos(theta_next - theta_prev))
        return dist_sq - link_len**2
    d_theta_guess = link_len / (a * np.sqrt(1 + theta_prev**2)) if theta_prev > 1e-4 else link_len / a
    initial_guess = theta_prev + d_theta_guess
    sol, _, ier, _ = fsolve(func_to_solve, [initial_guess], full_output=True)
    return sol[0] if ier == 1 else np.nan

def get_all_positions_at_t(t, a, s_initial):
    thetas = np.full(NUM_POINTS, np.nan)
    theta_0 = get_theta0(t, a, V0, s_initial)
    if theta_0 < 0: return None
    thetas[0] = theta_0
    for i in range(1, NUM_POINTS):
        thetas[i] = get_next_theta(thetas[i - 1], effective_link_lengths[i - 1], a)
        if np.isnan(thetas[i]): return None
    rhos = a * thetas
    return np.column_stack((rhos * np.cos(thetas), rhos * np.sin(thetas)))

def point_to_segment_distance(p, a, b):
    ab, ap = b - a, p - a
    len_sq_ab = np.dot(ab, ab)
    if len_sq_ab < 1e-14: return np.linalg.norm(ap)
    t = np.dot(ap, ab) / len_sq_ab
    closest = a if t < 0.0 else (b if t > 1.0 else a + t * ab)
    return np.linalg.norm(p - closest)

def get_min_clearance_for_positions(positions):
    if positions is None or len(positions) < 3: return float('inf')
    p0, p1 = positions[0], positions[1]
    v_axis_p1_to_p0 = p0 - p1
    u_axis = v_axis_p1_to_p0 / np.linalg.norm(v_axis_p1_to_p0)
    v_perp = np.array([-u_axis[1], u_axis[0]])
    if np.linalg.norm(p0 + v_perp) < np.linalg.norm(p0): v_perp = -v_perp
    
    crit_p_front = p0 + u_axis * HINGE_OFFSET + v_perp * (LINK_WIDTH / 2.0)
    crit_p_rear = p1 - u_axis * HINGE_OFFSET + v_perp * (LINK_WIDTH / 2.0)
    
    min_clearance = float('inf')
    for i in range(2, len(positions) - 1):
        target_p_start, target_p_end = positions[i], positions[i+1]
        dist_front = point_to_segment_distance(crit_p_front, target_p_start, target_p_end)
        min_clearance = min(min_clearance, dist_front - (LINK_WIDTH / 2.0))
        dist_rear = point_to_segment_distance(crit_p_rear, target_p_start, target_p_end)
        min_clearance = min(min_clearance, dist_rear - (LINK_WIDTH / 2.0))
    return min_clearance

# --- 3. 核心分析函数 ---
def analyze_pitch_monotonicity(pitch, dt=0.1, max_time=2000):
    """
    对于给定的螺距(pitch), 计算在到达调头区前的最小连杆间隙
    """
    a = pitch / (2 * np.pi)
    s_initial = arc_length_func(THETA_INITIAL, a)
    min_clearance_overall = float('inf')

    for t in np.arange(0, max_time, dt):
        positions = get_all_positions_at_t(t, a, s_initial)
        if positions is None: continue
            
        if np.linalg.norm(positions[0]) <= TURNAROUND_RADIUS:
            break # 到达调头区，停止分析
        
        current_clearance = get_min_clearance_for_positions(positions)
        min_clearance_overall = min(min_clearance_overall, current_clearance)
        
        if min_clearance_overall <= 0:
            break # 发生碰撞，停止分析
            
    return min_clearance_overall

# --- 4. 绘图与主程序 ---

def run_and_plot(d_range, d_step, title_suffix):
    """执行分析并返回结果"""
    pitch_values = np.arange(d_range[0], d_range[1], d_step)
    clearances = []
    print(f"\n开始分析 {title_suffix}: {len(pitch_values)} 个螺距值...")
    for pitch in tqdm(pitch_values, desc=f"分析进度 {title_suffix}"):
        clearances.append(analyze_pitch_monotonicity(pitch))
    return pitch_values, np.array(clearances)

if __name__ == "__main__":
    # --- 执行分析 ---
    pitches_fine, clearances_fine = run_and_plot(
        d_range=(0.445, 0.455), d_step=0.0001, title_suffix="精细范围"
    )
    pitches_coarse, clearances_coarse = run_and_plot(
        d_range=(0.3, 0.55), d_step=0.01, title_suffix="总体范围"
    )
    
    # --- 绘图 ---
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('最小间隙 vs 螺距d 的单调性分析 (检验二分法适用性)', fontsize=16)

    # 子图1: 精细范围
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(pitches_fine, clearances_fine, 'b-', label='最小间隙')
    ax1.axhline(0, color='red', linestyle='--', label='碰撞临界线 (间隙=0)')
    ax1.fill_between(pitches_fine, clearances_fine, 0, where=clearances_fine<=0, color='red', alpha=0.3, label='碰撞区域')
    ax1.set_title('精细范围 (d: 0.445-0.455m)')
    ax1.set_xlabel('螺距 d (m)')
    ax1.set_ylabel('最小连杆间隙 (m)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 子图2: 总体范围
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(pitches_coarse, clearances_coarse, 'g-', label='最小间隙')
    ax2.axhline(0, color='red', linestyle='--', label='碰撞临界线 (间隙=0)')
    ax2.fill_between(pitches_coarse, clearances_coarse, 0, where=clearances_coarse<=0, color='red', alpha=0.3, label='碰撞区域')
    ax2.set_title('总体范围 (d: 0.3-0.55m)')
    ax2.set_xlabel('螺距 d (m)')
    ax2.set_ylabel('最小连杆间隙 (m)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 子图3: 单调性检验 (导数)
    ax3 = plt.subplot(2, 2, 3)
    # 使用平滑后的导数以减少噪声
    gradient_fine = np.gradient(np.convolve(clearances_fine, np.ones(5)/5, mode='same'))
    ax3.plot(pitches_fine, gradient_fine, 'm-', label='间隙函数导数 (平滑后)')
    ax3.axhline(0, color='black', linestyle='--')
    ax3.set_title('单调性检验 (精细范围导数)')
    ax3.set_xlabel('螺距 d (m)')
    ax3.set_ylabel('d(间隙)/dd')
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 子图4: 结论文本
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # 分析结论
    is_monotonic_fine = np.all(gradient_fine >= -1e-3) # 允许微小的负值以应对计算噪声
    conclusion_text = "核心结论：二分法是否恰当？\n\n"
    if is_monotonic_fine:
        conclusion_text += "1. 函数单调性：\n   在精细分析的区间内，最小间隙随螺距 `d` 的增大而\n   严格增大。函数表现出良好的 **单调性**。\n\n"
        conclusion_text += "2. 二分法适用性：\n   **非常恰当**。因为函数的单调性保证了：\n   - 如果一个螺距 `d` 会碰撞（间隙≤0），那么所有\n     更小的螺距也一定会碰撞。\n   - 如果一个螺距 `d` 安全（间隙>0），那么所有\n     更大的螺距也一定安全。\n\n"
        conclusion_text += "3. 算法选择：\n   对于寻找最小安全螺距 `d_min` 的问题，二分搜索\n   是最高效、最可靠的算法。"
    else:
        conclusion_text += "1. 函数单调性：\n   **警告！** 在精细分析的区间内，函数出现 **非单调** 行为。\n   即存在螺距增大了，但最小间隙反而减小的情况。\n\n"
        conclusion_text += "2. 二分法适用性：\n   **不恰当且有风险**。非单调性可能导致二分法\n   过早收敛到错误的局部解，或在两个安全区间之间\n   的"危险谷地"来回振荡，找不到真正的临界点。\n\n"
        conclusion_text += "3. 算法选择：\n   应使用 **网格搜索** (Grid Search) 来保证找到全局\n   最优解。可以先粗略网格搜索，再在临界点附近\n   进行精细网格搜索。"

    ax4.text(0.05, 0.95, conclusion_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11, family='monospace',
             bbox=dict(boxstyle="round,pad=0.5", fc='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pitch_monotonicity_analysis.png', dpi=300)
    print("\n分析图表已保存为 'pitch_monotonicity_analysis.png'")
    plt.show() 