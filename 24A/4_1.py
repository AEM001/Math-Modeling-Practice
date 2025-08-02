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
        d0 = 3.41 - 0.275 * 2  # 龙头板长
        theta_1, theta_2, theta_3 = 0.9917636, 2.5168977, 14.1235657
    else:
        d0 = 2.2 - 0.275 * 2   # 龙身板长
        theta_1, theta_2, theta_3 = 0.5561483, 1.1623551, 13.8544471
    
    # 位置迭代计算
    if flag_last == 1:  # 前把手在盘入螺线
        # 定义方程：盘入螺线上的位置约束
        def spiral_constraint(theta, d, d0, theta_last):
            return (theta**2 + theta_last**2 - 
                   2*theta*theta_last*np.cos(theta-theta_last) - 
                   4*np.pi**2*d0**2/d**2)
        theta = find_zero_point(spiral_constraint, theta_last, theta_last+np.pi/2, 1e-8, d, d0, theta_last)
        flag = 1
        
    elif flag_last == 2:  # 前把手在第一段圆弧
        if theta_last < theta_1:
            # 第一段圆弧到盘入螺线的过渡
            b = np.sqrt(2-2*np.cos(theta_last)) * r * 2
            beta = (aleph - theta_last) / 2
            l = np.sqrt(b**2 + D**2/4 - b*D*np.cos(beta))
            gamma = np.arcsin(b*np.sin(beta)/l)
            
            def transition_constraint(theta, d, d0, theta0, l, gamma):
                return (l**2 + d**2*theta**2/(4*np.pi**2) - 
                       d*l*theta*np.cos(theta-theta0+gamma)/np.pi - d0**2)
            theta = find_zero_point(transition_constraint, theta0, theta0+np.pi/2, 1e-8, d, d0, theta0, l, gamma)
            flag = 1
        else:
            # 第一段圆弧内部
            theta = theta_last - theta_1
            flag = 2
            
    elif flag_last == 3:  # 前把手在第二段圆弧
        if theta_last < theta_2:
            # 第二段圆弧到第一段圆弧的过渡
            a = np.sqrt(10-6*np.cos(theta_last)) * r
            phi = np.arccos((4*r**2+a**2-d0**2)/(4*a*r))
            beta = np.arcsin(r*np.sin(theta_last)/a)
            theta = aleph - phi + beta
            flag = 2
        else:
            # 第二段圆弧内部
            theta = theta_last - theta_2
            flag = 3
            
    else:  # flag_last == 4，前把手在盘出螺线
        if theta_last < theta_3:
            # 盘出螺线到第二段圆弧的过渡
            p = d * (theta_last + np.pi) / (2 * np.pi)
            a = np.sqrt(p**2 + D**2/4 - p*D*np.cos(theta_last-theta0+np.pi))
            beta = np.arcsin(p*np.sin(theta_last-theta0+np.pi)/a)
            gamma = beta - (np.pi-aleph)/2
            b = np.sqrt(a**2 + r**2 - 2*a*r*np.cos(gamma))
            sigma = np.arcsin(a*np.sin(gamma)/b)
            phi = np.arccos((r**2+b**2-d0**2)/(2*r*b))
            theta = aleph - phi + sigma
            flag = 3
        else:
            # 盘出螺线上的位置约束
            def spiral_out_constraint(theta, d, d0, theta_last):
                t = theta + np.pi
                t_1 = theta_last + np.pi
                return (t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2)
            theta = find_zero_point(spiral_out_constraint, theta_last-np.pi/2, theta_last, 1e-8, d, d0, theta_last)
            flag = 4
            
    return theta, flag

def velocity_iteration(v_last, flag_last, flag, theta_last, theta, x_last, y_last, x, y):
    """计算速度迭代"""
    # 圆弧圆心坐标
    x1, y1 = -0.7600091, -1.3057264  # 第一段圆弧圆心
    x2, y2 = 1.7359325, 2.4484020    # 第二段圆弧圆心
    
    # 计算板的斜率
    k_board = (y_last - y) / (x_last - x)
    
    # 根据不同情况计算速度斜率
    if flag_last == 1 and flag == 1:  # 都在盘入螺线
        k_v_last = velocity_slope_on_spiral(theta_last)
        k_v = velocity_slope_on_spiral(theta)
    elif flag_last == 2 and flag == 1:  # 第一段圆弧到盘入螺线
        k_v_last = -(x_last - x1) / (y_last - y1)
        k_v = velocity_slope_on_spiral(theta)
    elif flag_last == 2 and flag == 2:  # 都在第一段圆弧
        return v_last
    elif flag_last == 3 and flag == 2:  # 第二段圆弧到第一段圆弧
        k_v_last = -(x_last - x2) / (y_last - y2)
        k_v = -(x - x1) / (y - y1)
    elif flag_last == 3 and flag == 3:  # 都在第二段圆弧
        return v_last
    elif flag_last == 4 and flag == 3:  # 盘出螺线到第二段圆弧
        k_v_last = velocity_slope_on_spiral(theta_last + np.pi)
        k_v = -(x - x2) / (y - y2)
    else:  # 都在盘出螺线
        k_v_last = velocity_slope_on_spiral(theta_last + np.pi)
        k_v = velocity_slope_on_spiral(theta + np.pi)
    
    # 计算角度和速度
    angle1 = np.arctan(np.abs((k_v_last - k_board) / (1 + k_v_last * k_board)))
    angle2 = np.arctan(np.abs((k_v - k_board) / (1 + k_v * k_board)))
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
theta1 = 4.0055376  # 第一段圆弧的进入点极角
theta2 = 0.8639449  # 第二段圆弧的离开点极角

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