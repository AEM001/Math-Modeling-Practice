一个由223个刚性连杆组成的链式系统的运动学问题：
系统构成：
•	第1个连杆：长度3.41 m，宽度0.30 m
•	第2-222个连杆：每个长度2.20 m，宽度0.30 m
•	第223个连杆：长度2.20 m，宽度0.30 m
连接方式：
•	每个连杆上有两个铰接点，均分别位于距离连杆两端0.275 m处
•	铰接点直径0.055 m
•	相邻连杆通过铰接点连接，形成可转动的关节
•	连杆i的后铰接点与连杆i+1的前铰接点重合连接
运动约束：
•	整个链式系统沿一条等距螺旋线运动，初始位于
•	螺旋线的螺距为0.55 m（相邻圈之间的垂直距离）
•	所有铰接点的中心都严格约束在这条螺旋线上
•	第1个连杆前端的铰接点以恒定速度1.0 m/s沿螺旋线运动

## 问题1：223个刚性连杆链式系统的运动学分析
运动条件：
第1个连杆前端铰接点以恒定速度1.0 m/s沿等距螺旋线运动，螺旋线螺距为0.55 m，所有铰接点中心严格约束在螺旋线上，并且保持连杆的本身约束，初始时刻(t=0)，第1个连杆前端铰接点位于螺旋线第16圈的点A处，整体沿着顺时针进入
求解要求：
完整运动分析：计算从t=0秒到t=300秒期间，每秒时刻下整个链式系统中每个铰接点的位置和速度

# 问题2：223个刚性连杆链式系统的碰撞约束分析

**问题背景：** 
• 延续问题1的运动设定：223个刚性连杆链式系统沿等距螺旋线运动 
• 第1个连杆前端铰接点以恒定速度1.0 m/s沿螺旋线运动
• 螺旋线螺距为0.55 m，所有铰接点约束在螺旋线上

**碰撞约束条件：**
• 连杆几何尺寸：宽度0.30 m，需考虑物理占用空间 
• 碰撞判定：见下文
• 连杆形状：矩形，注意第一个的特殊性
• 碰撞检测范围：需检查所有可能的连杆间距离

**求解目标：**

1. **终止时刻确定**：找到使得连杆间不发生碰撞的最大运动时间t_max
    - 即在t ≤ t_max时，系统可以安全运动
    - 当t > t_max时，至少存在一对连杆发生碰撞
2. **终止状态分析**：在t = t_max时刻，给出系统的完整状态
    - 所有铰接点的位置坐标(x,y,z)
    - 所有铰接点的速度矢量(vx,vy,vz)


# 问题三
从盘入到盘出，整体的连杆结构将由顺时针盘入调头切换为逆时针盘出，这需要一定的调头空间。若调头空间是以螺线中心为圆心、直径为 9 m 的圆形区域，请确定最小螺距，使得第一个连杆的前端铰接点能够沿着相应的螺线盘入到调头空间的边界。

## 问题三代码
```
python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import fsolve

import warnings

  

# 抑制 fsolve 的警告

warnings.filterwarnings('ignore', 'The iteration is not making good progress')

  

# --- 1. 系统常量和参数 ---

LINK_WIDTH = 0.30 # 连杆的物理宽度 (m)

V0 = 1.0 # 第一个铰接点的初始速度 (m/s)

NUM_LINKS = 223 # 连杆数量

NUM_POINTS = NUM_LINKS + 1 # 铰接点数量

HINGE_OFFSET = 0.275 # 铰接点距离连杆两端的偏移量

TURNAROUND_DIAMETER = 9.0 # 调头空间直径 (m)

TURNAROUND_RADIUS = TURNAROUND_DIAMETER / 2.0 # 调头空间半径 (m)

  

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

pitch_min = 0.3 # 最小可能螺距

pitch_max = 0.6 # 最大可能螺距

tolerance = 1e-6 # 搜索精度

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

print(f" 发生碰撞，搜索更大螺距")

else:

# 未发生碰撞，可以尝试更小的螺距

pitch_max = pitch_mid

print(f" 未发生碰撞，搜索更小螺距")

# 最终结果取略大于最小值的安全值

optimal_pitch = pitch_max + tolerance

return optimal_pitch

  

# --- 主程序执行 ---

  

if __name__ == "__main__":

try:

# 开始优化计算

result = optimize_spiral_pitch()

print("\n" + "=" * 60)

print(" 优化结果")

print("=" * 60)

print(f"在调头空间外不会发生碰撞的最小螺距: {result:.6f} m")

# 验证结果

print(f"\n结果验证:")

a = result / (2 * np.pi)

print(f" 螺旋线参数 a: {a:.6f} m/rad")

print(f" 螺线紧密度: {2*np.pi/result:.4f} rad/m")

# 计算到达调头空间需要的时间

s_initial = arc_length_func(THETA_INITIAL, a)

# 粗略估算

theta_turnaround = TURNAROUND_RADIUS / a

s_turnaround = arc_length_func(theta_turnaround, a)

time_to_turnaround = (s_initial - s_turnaround) / V0

print(f" 预计到达调头空间时间: {time_to_turnaround:.2f} s")

print("\n优化完成!")

except Exception as e:

print(f"计算过程中发生错误: {e}")

raise
```
# 第四问
问题 4 盘入螺线的螺距为 1.7 m，盘出螺线与盘入螺线关于螺线中心呈中心对称，舞龙队在问题 3 设定的调头空间内完成调头，调头路径是由两段圆弧相切连接而成的 S 形曲线，前一段圆弧的半径是后一段的 2 倍，它与盘入、盘出螺线均相切。龙头前把手的行进速度始终保持 1 m/s。以调头开始时间为零时刻，给出从−100 s 开始到100 s 为止，每秒舞龙队的位置和速度，将结果存放到文件 result4.xlsx 中（模板文件见附件）。同时在论文中给出−100 s、−50 s、0 s、50 s、100 s 时，龙头前把手、龙头后面第 1、51、101、151、201 节龙身前把手和龙尾后把手的位置和速度。
已经提前计算好的数值，

| 分类         | 变量名 (代码中)  | 含义                      | 数值               |     |
| ---------- | ---------- | ----------------------- | ---------------- | --- |
| **基本输入**   | d          | 螺线螺距                    | 1.7 m            |     |
|            | D          | 调头空间直径                  | 9.0 m            |     |
|            | v0         | 龙头行进速度                  | 1.0 m/s          |     |
| **核心几何参数** | theta0     | 龙头在0时刻（调头开始）的螺线极角       | 16.6319611 rad   |     |
|            | r          | S形曲线中较小圆弧的半径 (r₂)       | 1.5027088 m      |     |
|            | aleph      | S形曲线中两段圆弧对应的圆心角 (α)     | 3.0214868 rad    |     |
| **衍生参数**   | r1         | S形曲线中较大圆弧的半径 (r₁ = 2r₂) | 3.0054176 m      |     |
|            | t1         | 龙头到达S形曲线中点P所需时间         | 9.0808299 s      |     |
|            | t2         | 龙头完成整个调头（到达P_out）所需时间   | 13.6212449 s     |     |
|            | x1, y1     | S形曲线第一段大圆弧的圆心坐标         | (-0.76, -1.31) m |     |
|            | x2, y2     | S形曲线第二段小圆弧的圆心坐标         | (1.74, 2.45) m   |     |
|            | theta1_arc | 大圆弧路径的起始计算角度            | 4.0055376 rad    |     |
|            | theta2_arc | 小圆弧路径的起始计算角度            | 0.8639449 rad    |     |
## 第四问代码
```python
import numpy as np
import pandas as pd

# ============ 核心计算函数 ============

def spiral_length_integral(theta):
    """计算螺线长度积分"""
    return theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))

def find_zero_point(func, a, b, tolerance=1e-8, *args):
    """通用二分法求零点"""
    while b - a >= tolerance:
        c = (a + b) / 2
        if func(a, *args) * func(c, *args) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def round_array(arr, decimals=6):
    """数组保留指定小数位数"""
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            if isinstance(arr[i], (list, np.ndarray)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j], decimals)
            else:
                arr[i] = round(arr[i], decimals)
    return arr

def velocity_slope_on_spiral(theta):
    """计算螺线上速度的斜率"""
    return (np.sin(theta) + theta * np.cos(theta)) / (np.cos(theta) - theta * np.sin(theta))

def position_iteration(theta_last, flag_last, segment_index):
    """计算位置迭代 """
    # 参数定义
    d = 1.7  # 螺距
    D = 9    # 调头空间的直径
    theta0 = 16.6319611  # 龙头0时刻时的极角
    r = 1.5027088       # 第二段圆弧的半径
    aleph = 3.0214868   # 两段圆弧的圆心角

    # 确定板长和关键位置参数
    if segment_index == 0:
        d0 = 3.41 - 0.275 * 2  # 龙头板长 (第一个连杆的有效长度)
        theta_1, theta_2, theta_3 = 0.9917636, 2.5168977, 14.1235657 # 特定过渡点的极角
    else:
        d0 = 2.2 - 0.275 * 2   # 龙身板长 (其他连杆的有效长度)
        theta_1, theta_2, theta_3 = 0.5561483, 1.1623551, 13.8544471 # 特定过渡点的极角

    # 位置迭代计算
    if flag_last == 1:  # 前把手在盘入螺线
        # 定义方程：盘入螺线上的位置约束
        def spiral_constraint(theta, d, d0, theta_last):
            a = d / (2 * np.pi)
            # 两个铰接点在螺线上，距离为 d0
            # 使用极坐标距离公式：距离^2 = rho_i^2 + rho_j^2 - 2 * rho_i * rho_j * cos(theta_i - theta_j)
            # (d0)^2 = (a*theta)^2 + (a*theta_last)^2 - 2*(a*theta)*(a*theta_last)*cos(theta - theta_last)
            # d0^2 / a^2 = theta^2 + theta_last^2 - 2*theta*theta_last*cos(theta - theta_last)
            # 4*pi^2*d0^2/d^2 = theta^2 + theta_last^2 - 2*theta*theta_last*cos(theta - theta_last)
            return (theta**2 + theta_last**2 -
                   2*theta*theta_last*np.cos(theta-theta_last) -
                   4*np.pi**2*d0**2/d**2)
        # 寻找下一个铰接点的极角 theta，保证与前一个铰接点距离为 d0
        theta = find_zero_point(spiral_constraint, theta_last, theta_last+np.pi/2, 1e-8, d, d0, theta_last)
        flag = 1 # 标记仍在盘入螺线

    elif flag_last == 2:  # 前把手在第一段圆弧 (大圆弧)
        # 根据龙头位置（theta_last）判断是处于圆弧到螺线的过渡段还是圆弧内部
        if theta_last < theta_1: # 表示当前把手位于第一段圆弧的末端附近，下一个把手可能进入盘入螺线
            # 几何计算，涉及大圆弧、调头空间和盘入螺线的几何关系，以及连杆长度d0
            # 这部分是复杂的几何求解，通过一系列三角函数和勾股定理来构建下一个铰接点在盘入螺线上的约束方程
            # b, beta, l, gamma 是中间计算量
            b = np.sqrt(2-2*np.cos(theta_last)) * r * 2 # 注意这里的 r * 2 对应大圆弧半径 R1
            beta = (aleph - theta_last) / 2
            l = np.sqrt(b**2 + D**2/4 - b*D*np.cos(beta))
            gamma = np.arcsin(b*np.sin(beta)/l)

            # 这是一个约束方程，用于在盘入螺线上找到下一个铰接点的极角 theta
            # 涉及到从圆弧上的点到螺线上的点，距离为 d0 的情况
            def transition_constraint(theta, d, d0, theta0, l, gamma):
                a = d / (2 * np.pi)
                # 这个方程可能描述了从大圆弧上一个点到盘入螺线上一个点的距离等于 d0 的几何关系
                # 具体表达式可能与S形曲线和螺线的切线连接有关
                return (l**2 + a**2*theta**2 -
                       2*l*a*theta*np.cos(theta-theta0+gamma) - d0**2)
            # 寻找零点以确定下一个铰接点的极角
            theta = find_zero_point(transition_constraint, theta0, theta0+np.pi/2, 1e-8, d, d0, theta0, l, gamma)
            flag = 1 # 标记进入盘入螺线
        else: # 当前把手在第一段圆弧内部，下一个把手也继续在第一段圆弧上
            # 只需要减去已经走过的圆弧角度 (theta_1)
            theta = theta_last - theta_1
            flag = 2 # 标记仍在第一段圆弧

    elif flag_last == 3:  # 前把手在第二段圆弧 (小圆弧)
        # 根据龙头位置判断是处于第二段圆弧到第一段圆弧的过渡段还是第二段圆弧内部
        if theta_last < theta_2: # 表示当前把手位于第二段圆弧的末端附近，下一个把手可能进入第一段圆弧
            # 复杂的几何计算，用于确定从小圆弧上的点到大圆弧上的点，距离为 d0 的情况
            # a, phi, beta 是中间计算量
            a = np.sqrt(10-6*np.cos(theta_last)) * r
            phi = np.arccos((4*r**2+a**2-d0**2)/(4*a*r)) # 4*r^2 对应大圆弧半径平方 (2r)^2
            beta = np.arcsin(r*np.sin(theta_last)/a)
            theta = aleph - phi + beta # 计算下一个铰接点在大圆弧上的角度
            flag = 2 # 标记进入第一段圆弧
        else: # 当前把手在第二段圆弧内部，下一个把手也继续在第二段圆弧上
            # 只需要减去已经走过的圆弧角度 (theta_2)
            theta = theta_last - theta_2
            flag = 3 # 标记仍在第二段圆弧

    else:  # flag_last == 4，前把手在盘出螺线
        # 根据龙头位置判断是处于盘出螺线到第二段圆弧的过渡段还是盘出螺线内部
        if theta_last < theta_3: # 表示当前把手位于盘出螺线的末端附近，下一个把手可能进入第二段圆弧
            # 复杂的几何计算，用于确定从盘出螺线上的点到小圆弧上的点，距离为 d0 的情况
            # p, a, beta, gamma, b, sigma, phi 是中间计算量
            p = d * (theta_last + np.pi) / (2 * np.pi) # 盘出螺线可以看作盘入螺线旋转 pi 角度，所以极角加 pi
            a = np.sqrt(p**2 + D**2/4 - p*D*np.cos(theta_last-theta0+np.pi))
            beta = np.arcsin(p*np.sin(theta_last-theta0+np.pi)/a)
            gamma = beta - (np.pi-aleph)/2
            b = np.sqrt(a**2 + r**2 - 2*a*r*np.cos(gamma))
            sigma = np.arcsin(a*np.sin(gamma)/b)
            phi = np.arccos((r**2+b**2-d0**2)/(2*r*b))
            theta = aleph - phi + sigma # 计算下一个铰接点在小圆弧上的角度
            flag = 3 # 标记进入第二段圆弧
        else: # 当前把手在盘出螺线内部，下一个把手也继续在盘出螺线上
            # 定义方程：盘出螺线上的位置约束
            def spiral_out_constraint(theta, d, d0, theta_last):
                a = d / (2 * np.pi)
                t = theta + np.pi # 盘出螺线极角需要加上 pi 转换为与盘入螺线一致的极角系统
                t_1 = theta_last + np.pi
                # 同样的极坐标距离公式，确保与前一个铰接点距离为 d0
                return (t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2)
            # 寻找零点以确定下一个铰接点的极角
            theta = find_zero_point(spiral_out_constraint, theta_last-np.pi/2, theta_last, 1e-8, d, d0, theta_last)
            flag = 4 # 标记仍在盘出螺线

    return theta, flag

def velocity_iteration(v_last, flag_last, flag, theta_last, theta, x_last, y_last, x, y):
    """计算速度迭代"""
    # 圆弧圆心坐标 (来自问题背景预计算值)
    x1, y1 = -0.7600091, -1.3057264  # 第一段圆弧圆心
    x2, y2 = 1.7359325, 2.4484020    # 第二段圆弧圆心

    # 计算板的斜率 (当前连杆的轴线方向)
    k_board = (y_last - y) / (x_last - x)

    # 根据不同情况计算速度斜率 (前把手和后把手在各自路径上的速度方向)
    if flag_last == 1 and flag == 1:  # 前后把手都在盘入螺线
        k_v_last = velocity_slope_on_spiral(theta_last) # 前把手在螺线上的速度斜率
        k_v = velocity_slope_on_spiral(theta)         # 后把手在螺线上的速度斜率
    elif flag_last == 2 and flag == 1:  # 前把手在第一段圆弧，后把手进入盘入螺线
        k_v_last = -(x_last - x1) / (y_last - y1) # 前把手在圆弧上的速度斜率 (切线垂直于半径)
        k_v = velocity_slope_on_spiral(theta)       # 后把手在螺线上的速度斜率
    elif flag_last == 2 and flag == 2:  # 前后把手都在第一段圆弧
        return v_last # 都在同一圆弧上，速度大小保持不变
    elif flag_last == 3 and flag == 2:  # 前把手在第二段圆弧，后把手进入第一段圆弧
        k_v_last = -(x_last - x2) / (y_last - y2) # 前把手在小圆弧上的速度斜率
        k_v = -(x - x1) / (y - y1)               # 后把手在大圆弧上的速度斜率
    elif flag_last == 3 and flag == 3:  # 前后把手都在第二段圆弧
        return v_last # 都在同一圆弧上，速度大小保持不变
    elif flag_last == 4 and flag == 3:  # 前把手在盘出螺线，后把手进入第二段圆弧
        k_v_last = velocity_slope_on_spiral(theta_last + np.pi) # 前把手在盘出螺线上的速度斜率 (极角需加pi)
        k_v = -(x - x2) / (y - y2)                             # 后把手在小圆弧上的速度斜率
    else:  # 都在盘出螺线 (flag_last == 4 and flag == 4)
        k_v_last = velocity_slope_on_spiral(theta_last + np.pi) # 前把手在盘出螺线上的速度斜率
        k_v = velocity_slope_on_spiral(theta + np.pi)           # 后把手在盘出螺线上的速度斜率

    # 计算角度和速度
    # angle1 是前把手的速度方向与连杆轴线之间的夹角
    # angle2 是后把手的速度方向与连杆轴线之间的夹角
    # tan(alpha) = |(k1 - k2) / (1 + k1*k2)| 用于计算两条直线的夹角
    angle1 = np.arctan(np.abs((k_v_last - k_board) / (1 + k_v_last * k_board)))
    angle2 = np.arctan(np.abs((k_v - k_board) / (1 + k_v * k_board)))

    # 根据连杆的刚性约束（长度不变），连杆上不同点的速度在连杆轴向上的分量必须相等。
    # 假设 v_last 是前把手的速度大小，v 是后把手的速度大小。
    # v_last * cos(angle1) = v * cos(angle2)
    # 从而得到 v = v_last * cos(angle1) / cos(angle2)
    return v_last * np.cos(angle1) / np.cos(angle2)

# ============ 主程序 ============

# 基本参数
d = 1.7           # 螺距
v0 = 1            # 龙头速度
theta0 = 16.6319611  # 龙头0时刻时的极角
r = 1.5027088     # 第二段圆弧的半径
aleph = 3.0214868 # 两段圆弧的圆心角
t1 = 9.0808299    # 龙头到达第二段圆弧的时刻
t2 = 13.6212449   # 龙头到达盘出螺线的时刻

# 圆弧圆心坐标
x1, y1 = -0.7600091, -1.3057264  # 第一段圆弧圆心
x2, y2 = 1.7359325, 2.4484020    # 第二段圆弧圆心
theta1 = 4.0055376  # 第一段圆弧的起始计算角度 (可能与龙头进入圆弧时的极角相关)
theta2 = 0.8639449  # 第二段圆弧的起始计算角度 (可能与龙头进入第二段圆弧时的极角相关)

# 计算所有时刻的位置和标志
positions_theta = []
positions_flag = []

for t in np.arange(-100, 101):
    # 计算龙头位置
    if t < 0:
        # 盘入螺线阶段
        def head_equation(theta, theta0, v0, t, d):
            return (spiral_length_integral(theta0) - spiral_length_integral(theta) - 4*v0*t*np.pi/d)
        theta_head = find_zero_point(head_equation, theta0, 100, 1e-8, theta0, v0, t, d)
        flag_head = 1
    elif t == 0:
        theta_head = theta0
        flag_head = 1
    elif t < t1:
        # 第一段圆弧阶段
        theta_head = v0 * t / (2 * r)
        flag_head = 2
    elif t < t2:
        # 第二段圆弧阶段
        theta_head = v0 * (t - t1) / r
        flag_head = 3
    else:
        # 盘出螺线阶段
        def head_equation_out(theta, theta0, v0, t, d):
            return (spiral_length_integral(theta0) - spiral_length_integral(theta) - 4*v0*t*np.pi/d)
        theta_head = find_zero_point(head_equation_out, theta0, 100, 1e-8, theta0, v0, -t + t2, d) - np.pi
        flag_head = 4
    
    # 计算所有把手位置
    current_theta = [theta_head]
    current_flag = [flag_head]
    
    for i in range(223):
        theta_new, flag_new = position_iteration(current_theta[-1], current_flag[-1], i)
        current_theta.append(theta_new)
        current_flag.append(flag_new)
    
    positions_theta.append(current_theta)
    positions_flag.append(current_flag)

positions_theta = np.array(positions_theta)
positions_flag = np.array(positions_flag)

# 计算坐标
coordinates = []
for i in range(201):
    coords = []
    for j in range(224):
        flag = positions_flag[i, j]
        theta = positions_theta[i, j]
        
        if flag == 1:  # 盘入螺线
            p = d * theta / (2 * np.pi)
            x = p * np.cos(theta)
            y = p * np.sin(theta)
        elif flag == 2:  # 第一段圆弧
            x = x1 + 2 * r * np.cos(theta1 - theta)
            y = y1 + 2 * r * np.sin(theta1 - theta)
        elif flag == 3:  # 第二段圆弧
            x = x2 + r * np.cos(theta2 + theta - aleph)
            y = y2 + r * np.sin(theta2 + theta - aleph)
        else:  # 盘出螺线
            p = d * (theta + np.pi) / (2 * np.pi)
            x = p * np.cos(theta)
            y = p * np.sin(theta)
        
        coords.extend([x, y])
    coordinates.append(coords)

coordinates = np.array(coordinates).T
coordinates = round_array(coordinates, 6)

# 保存坐标数据
df_coords = pd.DataFrame(coordinates)
df_coords.to_excel("result4_1.xlsx", index=False)

# 计算速度
velocities = []
for i in range(201):
    current_velocities = [v0]
    for j in range(223):
        # 获取相关参数
        flag_last = positions_flag[i, j]
        theta_last = positions_theta[i, j]
        flag = positions_flag[i, j + 1]
        theta = positions_theta[i, j + 1]
        
        # 获取坐标
        x_last = coordinates[j * 2, i]
        y_last = coordinates[j * 2 + 1, i]
        x = coordinates[j * 2 + 2, i]
        y = coordinates[j * 2 + 3, i]
        
        # 计算速度
        v = velocity_iteration(current_velocities[-1], flag_last, flag, theta_last, theta, 
                             x_last, y_last, x, y)
        current_velocities.append(v)
    
    velocities.append(current_velocities)

velocities = np.array(velocities).T
velocities = round_array(velocities, 6)

# 保存速度数据
df_velocities = pd.DataFrame(velocities)
df_velocities.to_excel("result4_2.xlsx", index=False)

# ============ 结果输出 ============
print("\n" + "="*60)
print("问题4：特定时刻和位置的详细数据")
print("="*60)

# 定义输出的时刻和位置
time_points = {"-100 s": 0, "-50 s": 50, "0 s": 100, "50 s": 150, "100 s": 200}
handle_points = {
    "龙头前把手": 0, "第1节龙身前把手": 1, "第51节龙身前把手": 51,
    "第101节龙身前把手": 101, "第151节龙身前把手": 151, 
    "第201节龙身前把手": 201, "龙尾后把手": 223
}

# 输出数据
for time_label, time_idx in time_points.items():
    print(f"\n--- 时间: {time_label} ---")
    for handle_label, handle_idx in handle_points.items():
        if handle_idx * 2 + 1 < len(coordinates) and handle_idx < len(velocities):
            x = coordinates[handle_idx * 2, time_idx]
            y = coordinates[handle_idx * 2 + 1, time_idx]
            v = velocities[handle_idx, time_idx]
            print(f"{handle_label}: 位置 ({x:.6f}, {y:.6f}) m, 速度 {v:.6f} m/s")
        else:
            print(f"{handle_label}: 数据索引超出范围")

print("\n" + "="*60)
print("数据输出完成")
print("="*60)

```

# 问题 5 舞龙队沿问题 4 设定的路径行进，龙头行进速度保持不变，请确定龙头的最大行进速度，使得舞龙队各把手的速度均不超过 2 m/s。
```python
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

```