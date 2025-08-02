# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import matplotlib.patches as patches

# 抑制 fsolve 的警告
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# --- 系统常量和参数 ---
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

# --- 核心运动学函数 ---
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

# --- 碰撞检测函数 ---
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
    """使用问题二的碰撞检测思路"""
    if positions is None: return False, float('inf')
    p0, p1 = positions[0], positions[1]
    v_axis_p1_to_p0 = p0 - p1
    u_axis = v_axis_p1_to_p0 / np.linalg.norm(v_axis_p1_to_p0)
    v_perp = np.array([-u_axis[1], u_axis[0]])
    if np.linalg.norm(p0 + v_perp) < np.linalg.norm(p0):
        v_perp = -v_perp
    
    front_end_point = p0 + u_axis * HINGE_OFFSET
    critical_point_front = front_end_point + v_perp * (LINK_WIDTH / 2.0)
    rear_end_point = p1 - u_axis * HINGE_OFFSET
    critical_point_rear = rear_end_point + v_perp * (LINK_WIDTH / 2.0)
    
    min_clearance = float('inf')
    collision_info = None
    
    for i in range(2, NUM_LINKS):
        if i >= len(positions) - 1: break
        target_p_start, target_p_end = positions[i], positions[i+1]
        
        dist_front = point_to_segment_distance(critical_point_front, target_p_start, target_p_end)
        clearance_front = float(dist_front - (LINK_WIDTH / 2.0))
        
        dist_rear = point_to_segment_distance(critical_point_rear, target_p_start, target_p_end)
        clearance_rear = float(dist_rear - (LINK_WIDTH / 2.0))
        
        if clearance_front < min_clearance:
            min_clearance = clearance_front
            collision_info = (i, 'front', clearance_front)
        if clearance_rear < min_clearance:
            min_clearance = clearance_rear
            collision_info = (i, 'rear', clearance_rear)
    
    return min_clearance <= 0, min_clearance, collision_info

def analyze_pitch_collision_detail(pitch, dt=0.1):
    """详细分析特定螺距下的碰撞情况"""
    a = pitch / (2 * np.pi)
    s_initial = arc_length_func(THETA_INITIAL, a)
    
    collision_occurred = False
    theta_c = None
    time_to_collision = None
    time_to_turnaround = None
    min_clearance_overall = float('inf')
    
    for t in np.arange(0, 1000, dt):
        positions, thetas = get_all_positions_at_t(t, a, s_initial)
        if positions is None:
            continue
            
        # 检查是否到达调头空间
        first_point_radius = np.linalg.norm(positions[0])
        if first_point_radius <= TURNAROUND_RADIUS:
            time_to_turnaround = t
            if thetas is not None:
                theta_c = thetas[0]
            break
            
        # 检查碰撞
        has_collision, clearance,_ = check_collision(positions)
        min_clearance_overall = min(min_clearance_overall, clearance)
        
        if has_collision and not collision_occurred:
            collision_occurred = True
            time_to_collision = t
    
    return {
        'pitch': pitch,
        'theta_c': theta_c,
        'collision': collision_occurred,
        'time_to_collision': time_to_collision,
        'time_to_turnaround': time_to_turnaround,
        'min_clearance': min_clearance_overall
    }

# --- 主分析程序 ---
print("="*60)
print("碰撞函数单调性分析")
print("="*60)

# 设置分析范围
d_min = 0.445
d_max = 0.455
d_step = 0.0001

pitch_values = np.arange(d_min, d_max + d_step, d_step)
results = []

print(f"\n分析范围: {d_min}m 到 {d_max}m，步长: {d_step}m")
print(f"共需分析 {len(pitch_values)} 个螺距值\n")

# 分析每个螺距
for pitch in tqdm(pitch_values, desc="分析进度"):
    result = analyze_pitch_collision_detail(pitch)
    results.append(result)

# 提取数据
pitches = np.array([r['pitch'] for r in results])
theta_c_values = np.array([r['theta_c'] if r['theta_c'] is not None else np.nan for r in results])
collisions = np.array([r['collision'] for r in results])
min_clearances = np.array([r['min_clearance'] for r in results])
product_values = pitches * theta_c_values

# 创建图表
fig = plt.figure(figsize=(15, 12))

# 子图1: d×θ_c vs d，带碰撞标记
ax1 = plt.subplot(3, 2, 1)
# 分别绘制碰撞和非碰撞的点
no_collision_mask = ~collisions
collision_mask = collisions

ax1.scatter(pitches[no_collision_mask], product_values[no_collision_mask], 
           c='green', s=10, label='无碰撞', alpha=0.6)
ax1.scatter(pitches[collision_mask], product_values[collision_mask], 
           c='red', s=10, label='有碰撞', alpha=0.6)
ax1.axhline(y=9*np.pi, color='blue', linestyle='--', label='理论值 9π')
ax1.set_xlabel('螺距 d (m)')
ax1.set_ylabel('d × θ_c (m·rad)')
ax1.set_title('螺距与到达角度乘积 (带碰撞标记)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: 最小间隙 vs 螺距
ax2 = plt.subplot(3, 2, 2)
ax2.plot(pitches, min_clearances, 'b-', linewidth=2)
ax2.axhline(y=0, color='red', linestyle='--', label='碰撞线')
ax2.set_xlabel('螺距 d (m)')
ax2.set_ylabel('最小间隙 (m)')
ax2.set_title('最小间隙 vs 螺距')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: 碰撞状态图
ax3 = plt.subplot(3, 2, 3)
# 创建碰撞区域可视化
collision_regions = []
in_collision = False
start_idx = 0

for i in range(len(collisions)):
    if collisions[i] and not in_collision:
        in_collision = True
        start_idx = i
    elif not collisions[i] and in_collision:
        in_collision = False
        collision_regions.append((pitches[start_idx], pitches[i-1]))

if in_collision:
    collision_regions.append((pitches[start_idx], pitches[-1]))

# 绘制碰撞区域
for start, end in collision_regions:
    ax3.axvspan(start, end, alpha=0.3, color='red', label='碰撞区域' if start == collision_regions[0][0] else '')

ax3.plot(pitches, collisions.astype(int), 'k-', linewidth=2)
ax3.set_xlabel('螺距 d (m)')
ax3.set_ylabel('碰撞状态 (0=安全, 1=碰撞)')
ax3.set_title('碰撞状态 vs 螺距')
ax3.set_ylim(-0.1, 1.1)
if collision_regions:
    ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4: θ_c vs 螺距
ax4 = plt.subplot(3, 2, 4)
ax4.plot(pitches, theta_c_values, 'g-', linewidth=2)
ax4.set_xlabel('螺距 d (m)')
ax4.set_ylabel('θ_c (rad)')
ax4.set_title('到达角度 θ_c vs 螺距')
ax4.grid(True, alpha=0.3)

# 子图5: 局部放大图（如果发现非单调性）
ax5 = plt.subplot(3, 2, 5)
# 计算最小间隙的导数来找到非单调区域
if len(min_clearances) > 2:
    clearance_gradient = np.gradient(min_clearances)
    # 找到导数变号的位置
    sign_changes = np.where(np.diff(np.sign(clearance_gradient)))[0]
    
    if len(sign_changes) > 0:
        # 放大第一个非单调区域
        idx = sign_changes[0]
        window = min(20, len(pitches)//10)
        start_idx = max(0, idx - window)
        end_idx = min(len(pitches), idx + window)
        
        ax5.plot(pitches[start_idx:end_idx], min_clearances[start_idx:end_idx], 'b-', linewidth=2)
        ax5.axhline(y=0, color='red', linestyle='--')
        ax5.set_xlabel('螺距 d (m)')
        ax5.set_ylabel('最小间隙 (m)')
        ax5.set_title('最小间隙局部放大（非单调区域）')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, '未发现明显非单调性', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('非单调性检测')

# 子图6: 统计信息
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

# 统计文本
stats_text = f"""分析结果统计：
螺距范围: {d_min:.4f} - {d_max:.4f} m
分析点数: {len(pitch_values)}
碰撞螺距数: {np.sum(collisions)}
安全螺距数: {np.sum(~collisions)}
碰撞比例: {np.sum(collisions)/len(collisions)*100:.1f}%

最小间隙范围: {np.min(min_clearances):.6f} - {np.max(min_clearances):.6f} m
d×θ_c 平均值: {np.nanmean(product_values):.6f} m·rad
理论值 9π: {9*np.pi:.6f} m·rad
相对误差: {abs(np.nanmean(product_values) - 9*np.pi)/(9*np.pi)*100:.2f}%"""

# 检测单调性
if len(min_clearances) > 2:
    clearance_gradient = np.gradient(min_clearances)
    monotonic = np.all(clearance_gradient >= 0) or np.all(clearance_gradient <= 0)
    stats_text += f"\n\n最小间隙函数单调性: {'是' if monotonic else '否'}"
    
    if not monotonic:
        sign_changes = np.where(np.diff(np.sign(clearance_gradient)))[0]
        stats_text += f"\n发现 {len(sign_changes)} 个导数变号点"

ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
         verticalalignment='top', fontsize=10, family='monospace')
ax6.set_title('统计信息')

plt.tight_layout()
plt.savefig('collision_monotonicity_analysis.png', dpi=300, bbox_inches='tight')
print("\n分析图表已保存为 'collision_monotonicity_analysis.png'")
plt.show()

# 输出关键发现
print("\n" + "="*60)
print("关键发现")
print("="*60)

# 找到转变点
transition_indices = np.where(np.diff(collisions.astype(int)))[0]
if len(transition_indices) > 0:
    print("\n碰撞状态转变点:")
    for idx in transition_indices:
        print(f"  螺距 {pitches[idx]:.6f} m 到 {pitches[idx+1]:.6f} m")
        print(f"  状态: {'安全→碰撞' if collisions[idx+1] else '碰撞→安全'}")

# 检查单调性
if len(min_clearances) > 2:
    clearance_gradient = np.gradient(min_clearances)
    if not (np.all(clearance_gradient >= 0) or np.all(clearance_gradient <= 0)):
        print("\n警告: 最小间隙函数关于螺距不是单调的！")
        print("这可能导致二分法搜索失效或找到局部最优解。")
        
        # 找到所有局部极值
        local_mins = []
        local_maxs = []
        for i in range(1, len(min_clearances)-1):
            if min_clearances[i] < min_clearances[i-1] and min_clearances[i] < min_clearances[i+1]:
                local_mins.append(i)
            elif min_clearances[i] > min_clearances[i-1] and min_clearances[i] > min_clearances[i+1]:
                local_maxs.append(i)
        
        if local_mins:
            print(f"\n发现 {len(local_mins)} 个局部最小值:")
            for idx in local_mins:
                print(f"  螺距: {pitches[idx]:.6f} m, 间隙: {min_clearances[idx]:.6f} m")
        
        if local_maxs:
            print(f"\n发现 {len(local_maxs)} 个局部最大值:")
            for idx in local_maxs:
                print(f"  螺距: {pitches[idx]:.6f} m, 间隙: {min_clearances[idx]:.6f} m")

print("\n分析完成！")