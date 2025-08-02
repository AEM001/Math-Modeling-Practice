# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# Mac 中文字体配置
def setup_chinese_font():
    """配置Mac系统的中文字体"""
    # Mac系统常见的中文字体
    chinese_fonts = [
        'PingFang SC',      # macOS 默认中文字体
        'Hiragino Sans GB', # 苹果丽黑字体
        'STHeiti',          # 华文黑体
        'Arial Unicode MS', # 包含中文的Arial字体
        'SimHei'            # 黑体（如果安装了）
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"使用中文字体: {font}")
            break
    else:
        print("警告: 未找到合适的中文字体，使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 设置负号正常显示
    plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
setup_chinese_font()

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
    if positions is None: return False, float('inf'), None
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
    collision_link = None
    
    for i in range(2, NUM_LINKS):
        if i >= len(positions) - 1: break
        target_p_start, target_p_end = positions[i], positions[i+1]
        
        dist_front = point_to_segment_distance(critical_point_front, target_p_start, target_p_end)
        clearance_front = float(dist_front - (LINK_WIDTH / 2.0))
        
        dist_rear = point_to_segment_distance(critical_point_rear, target_p_start, target_p_end)
        clearance_rear = float(dist_rear - (LINK_WIDTH / 2.0))
        
        if clearance_front < min_clearance:
            min_clearance = clearance_front
            collision_link = i
        if clearance_rear < min_clearance:
            min_clearance = clearance_rear
            collision_link = i
    
    return min_clearance <= 0, min_clearance, collision_link

def analyze_pitch_detailed(pitch, dt=0.05):
    """详细分析特定螺距下的运动和碰撞情况"""
    a = pitch / (2 * np.pi)
    s_initial = arc_length_func(THETA_INITIAL, a)
    
    # 初始化结果
    result = {
        'pitch': pitch,
        'reached_turnaround': False,
        'collision_occurred': False,
        'theta_at_turnaround': None,
        'theta_at_collision': None,
        'time_to_turnaround': None,
        'time_to_collision': None,
        'min_clearance': float('inf'),
        'collision_link': None
    }
    
    for t in np.arange(0, 1000, dt):
        positions, thetas = get_all_positions_at_t(t, a, s_initial)
        if positions is None:
            continue
            
        # 检查碰撞
        has_collision, clearance, collision_link = check_collision(positions)
        result['min_clearance'] = min(result['min_clearance'], clearance)
        
        # 记录首次碰撞
        if has_collision and not result['collision_occurred']:
            result['collision_occurred'] = True
            result['time_to_collision'] = t
            result['theta_at_collision'] = thetas[0] if thetas is not None else None
            result['collision_link'] = collision_link
            # 碰撞后继续检测是否能到达调头空间
        
        # 检查是否到达调头空间
        first_point_radius = np.linalg.norm(positions[0])
        if first_point_radius <= TURNAROUND_RADIUS and not result['reached_turnaround']:
            result['reached_turnaround'] = True
            result['time_to_turnaround'] = t
            result['theta_at_turnaround'] = thetas[0] if thetas is not None else None
            
        # 如果已经碰撞且到达调头空间，可以提前结束
        if result['collision_occurred'] and result['reached_turnaround']:
            break
    
    return result

# --- 主分析程序 ---
print("="*60)
print("螺距单调性综合分析")
print("="*60)

# 分析两个范围
ranges = [
    {'name': '精细范围', 'd_min': 0.445, 'd_max': 0.455, 'd_step': 0.001},
    {'name': '总体范围', 'd_min': 0.3, 'd_max': 0.55, 'd_step': 0.01}
]

all_results = {}

for range_config in ranges:
    print(f"\n分析{range_config['name']}: {range_config['d_min']}m - {range_config['d_max']}m")
    
    pitch_values = np.arange(range_config['d_min'], 
                           range_config['d_max'] + range_config['d_step'], 
                           range_config['d_step'])
    results = []
    
    for pitch in tqdm(pitch_values, desc=f"{range_config['name']}进度"):
        result = analyze_pitch_detailed(pitch)
        results.append(result)
    
    all_results[range_config['name']] = {
        'pitches': pitch_values,
        'results': results
    }

# 创建综合图表
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

# 为每个范围创建图表
for idx, (range_name, data) in enumerate(all_results.items()):
    pitches = data['pitches']
    results = data['results']
    
    # 提取数据
    collision_flags = np.array([r['collision_occurred'] for r in results])
    min_clearances = np.array([r['min_clearance'] for r in results])
    theta_turnaround = np.array([r['theta_at_turnaround'] if r['theta_at_turnaround'] else np.nan for r in results])
    theta_collision = np.array([r['theta_at_collision'] if r['theta_at_collision'] else np.nan for r in results])
    time_collision = np.array([r['time_to_collision'] if r['time_to_collision'] else np.nan for r in results])
    time_turnaround = np.array([r['time_to_turnaround'] if r['time_to_turnaround'] else np.nan for r in results])
    
    # 计算导出量
    product_turnaround = pitches * theta_turnaround
    product_collision = pitches * theta_collision
    
    # 图1: 最小间隙 vs 螺距
    ax1 = fig.add_subplot(gs[idx*2, 0])
    ax1.plot(pitches, min_clearances, 'b-', linewidth=2)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='碰撞线')
    collision_mask = (min_clearances <= 0).tolist()
    ax1.fill_between(pitches, -0.5, 0, where=collision_mask, 
                     color='red', alpha=0.2, label='碰撞区域')
    ax1.set_xlabel('螺距 d (m)')
    ax1.set_ylabel('最小间隙 (m)')
    ax1.set_title(f'{range_name}: 最小间隙 vs 螺距')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2: 碰撞时间 vs 螺距
    ax2 = fig.add_subplot(gs[idx*2, 1])
    # 分别绘制碰撞和未碰撞的情况
    collision_mask = ~np.isnan(time_collision)
    ax2.scatter(pitches[collision_mask], time_collision[collision_mask], 
               c='red', s=20, alpha=0.6, label='碰撞时间')
    ax2.plot(pitches, time_turnaround, 'g-', linewidth=2, label='到达调头空间时间')
    ax2.set_xlabel('螺距 d (m)')
    ax2.set_ylabel('时间 (s)')
    ax2.set_title(f'{range_name}: 关键时间点 vs 螺距')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 图3: d×θ 对比图
    ax3 = fig.add_subplot(gs[idx*2, 2])
    # 绘制到达调头空间的 d×θ
    valid_turnaround = ~np.isnan(product_turnaround)
    ax3.plot(pitches[valid_turnaround], product_turnaround[valid_turnaround], 
            'g-', linewidth=2, label='d×θ(调头空间)')
    # 绘制碰撞时的 d×θ
    valid_collision = ~np.isnan(product_collision)
    if np.any(valid_collision):
        ax3.scatter(pitches[valid_collision], product_collision[valid_collision], 
                   c='red', s=20, alpha=0.6, label='d×θ(碰撞)')
    ax3.axhline(y=9*np.pi, color='blue', linestyle='--', alpha=0.5, label='理论值 9π')
    ax3.set_xlabel('螺距 d (m)')
    ax3.set_ylabel('d × θ (m·rad)')
    ax3.set_title(f'{range_name}: 螺距与角度乘积')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 图4: 单调性分析
    ax4 = fig.add_subplot(gs[idx*2+1, :])
    
    # 计算导数并检测单调性
    if len(min_clearances) > 2:
        # 使用中心差分计算导数
        clearance_gradient = np.gradient(min_clearances, pitches)
        
        # 绘制导数
        ax4_twin = ax4.twinx()
        ax4.plot(pitches, min_clearances, 'b-', linewidth=2, label='最小间隙')
        ax4_twin.plot(pitches, clearance_gradient, 'r--', linewidth=1.5, 
                     alpha=0.7, label='间隙导数')
        ax4_twin.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # 标记导数变号点
        sign_changes = np.where(np.diff(np.sign(clearance_gradient)))[0]
        if len(sign_changes) > 0:
            for sc in sign_changes:
                ax4.axvline(x=pitches[sc], color='orange', linestyle='--', 
                           alpha=0.7, linewidth=1)
                ax4.text(pitches[sc], ax4.get_ylim()[1]*0.9, 
                        f'{pitches[sc]:.4f}', rotation=90, 
                        verticalalignment='bottom', fontsize=8)
        
        ax4.set_xlabel('螺距 d (m)')
        ax4.set_ylabel('最小间隙 (m)', color='b')
        ax4_twin.set_ylabel('间隙导数', color='r')
        ax4.set_title(f'{range_name}: 单调性分析（导数变号点: {len(sign_changes)}个）')
        ax4.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

# 添加总体统计信息
ax_stats = fig.add_subplot(gs[3, :])
ax_stats.axis('off')

stats_text = "="*80 + "\n统计分析结果\n" + "="*80 + "\n\n"

for range_name, data in all_results.items():
    pitches = data['pitches']
    results = data['results']
    
    collision_count = sum(r['collision_occurred'] for r in results)
    min_clearances = np.array([r['min_clearance'] for r in results])
    
    stats_text += f"{range_name}:\n"
    stats_text += f"  螺距范围: {pitches[0]:.4f} - {pitches[-1]:.4f} m\n"
    stats_text += f"  分析点数: {len(pitches)}\n"
    stats_text += f"  碰撞情况: {collision_count}/{len(results)} ({collision_count/len(results)*100:.1f}%)\n"
    stats_text += f"  最小间隙范围: [{np.min(min_clearances):.6f}, {np.max(min_clearances):.6f}] m\n"
    
    # 单调性检测
    if len(min_clearances) > 2:
        gradient = np.gradient(min_clearances, pitches)
        sign_changes = np.where(np.diff(np.sign(gradient)))[0]
        
        if len(sign_changes) == 0:
            if np.all(gradient >= -1e-6):
                stats_text += f"  单调性: 单调递增\n"
            elif np.all(gradient <= 1e-6):
                stats_text += f"  单调性: 单调递减\n"
            else:
                stats_text += f"  单调性: 近似单调\n"
        else:
            stats_text += f"  单调性: 非单调（{len(sign_changes)}个转折点）\n"
            
            # 找到关键转折点
            critical_points = []
            for sc in sign_changes:
                if sc > 0 and sc < len(pitches)-1:
                    critical_points.append(pitches[sc])
            
            if critical_points:
                stats_text += f"    转折点位置: {', '.join([f'{p:.4f}' for p in critical_points[:5]])}"
                if len(critical_points) > 5:
                    stats_text += f" ... (共{len(critical_points)}个)"
                stats_text += "\n"
    
    stats_text += "\n"

stats_text += "\n建议:\n"
# 基于分析结果给出建议
fine_results = all_results['精细范围']['results']
fine_clearances = np.array([r['min_clearance'] for r in fine_results])
if len(fine_clearances) > 2:
    gradient = np.gradient(fine_clearances)
    if len(np.where(np.diff(np.sign(gradient)))[0]) > 0:
        stats_text += "• 检测到非单调性，二分法可能找不到全局最优解\n"
        stats_text += "• 建议使用全局优化算法（如遗传算法、模拟退火）\n"
        stats_text += "• 或者使用多起点的局部搜索\n"
    else:
        stats_text += "• 函数近似单调，二分法应该有效\n"
        stats_text += "• 可以使用二分法快速找到最优解\n"

ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
             verticalalignment='top', fontsize=10, family='monospace')

plt.suptitle('螺距变化对系统行为的影响分析', fontsize=16)
plt.tight_layout()
plt.savefig('pitch_monotonicity_comprehensive.png', dpi=300, bbox_inches='tight')
print("\n综合分析图表已保存为 'pitch_monotonicity_comprehensive.png'")
plt.show()

# 输出额外的分析
print("\n" + "="*60)
print("关键发现总结")
print("="*60)

# 找出安全螺距范围
for range_name, data in all_results.items():
    print(f"\n{range_name}:")
    results = data['results']
    pitches = data['pitches']
    
    safe_ranges = []
    in_safe = False
    start_idx = 0
    
    for i, r in enumerate(results):
        if not r['collision_occurred'] and not in_safe:
            in_safe = True
            start_idx = i
        elif r['collision_occurred'] and in_safe:
            in_safe = False
            safe_ranges.append((pitches[start_idx], pitches[i-1]))
    
    if in_safe:
        safe_ranges.append((pitches[start_idx], pitches[-1]))
    
    if safe_ranges:
        print("  安全螺距区间:")
        for start, end in safe_ranges:
            print(f"    [{start:.4f}, {end:.4f}] m")
    else:
        print("  没有找到安全螺距区间")

print("\n分析完成！")